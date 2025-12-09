from typing import Callable, Dict, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn


__all__ = ["EmpiricalFF"]


def build_dispersion_embedding():
    C6_atomic_au = {
        1:  6.5,   # H
        6: 46.6,   # C
        7: 24.2,   # N
        8: 15.6,   # O
    }

    AU_TO_KCAL_A6_PER_MOL = 13.779293298534622
    C6_atomic_kcalA6 = {Z: c6_au * AU_TO_KCAL_A6_PER_MOL
                        for Z, c6_au in C6_atomic_au.items()}

    # Build a tensor indexed by Z; index 0 is unused.
    max_Z = max(C6_atomic_kcalA6)
    C6_atom = torch.zeros(max_Z + 1, dtype=torch.float32)
    for Z, c6 in C6_atomic_kcalA6.items():
        C6_atom[Z] = c6
    return C6_atom / 100.


class EmpiricalFF(nn.Module):
    def __init__(
        self,
        molecule_db_file: str,
        terms=['all'],
    ):
        super(EmpiricalFF, self).__init__()
        # dummy value that is only used for interpretability F_ij analysis
        self.register_buffer('cutoff', torch.tensor(999.))
        self.terms = terms

        # load bond_indices
        molecule = molecule_db_file.removeprefix("mbd_").removesuffix('.db')
        if molecule not in ('Ac-Ala3-NHMe', 'DHA', 'buckyball-catcher', 'chignolin'):
            raise Exception(f'no ff for molecule db file {molecule_db_file}')
        loaded = torch.load(f'/home/space/datasets/xai4qc/md22/empirical_ff_{molecule}.pth', weights_only=True)

        self.register_buffer('bonded_mask', loaded['bonded_mask'])
        self.register_buffer('one_three_nonbonded_mask', loaded['one_three_nonbonded_mask'])
        self.register_buffer('idx_i_full', loaded['idx_i_full'])
        self.register_buffer('idx_j_full', loaded['idx_j_full'])
        self.register_buffer('idx_i_bonded', loaded['idx_i_bonded'])
        self.register_buffer('idx_j_bonded', loaded['idx_j_bonded'])
        self.register_buffer('idx_i_triples', loaded[properties.idx_i_triples])
        self.register_buffer('idx_j_triples', loaded[properties.idx_j_triples])
        self.register_buffer('idx_k_triples', loaded[properties.idx_k_triples])
        self.register_buffer('charges', loaded['charges'])


        self.bond_distance_equilibrium = torch.nn.Parameter(torch.ones(self.idx_i_bonded.shape[0]) * 1.3)
        self.bond_distance_force_constant =  torch.nn.Parameter(torch.ones(self.idx_i_bonded.shape[0]) * 5)
        self.bond_angle_equilibrium = torch.nn.Parameter(torch.ones(self.idx_j_triples.shape[0]) * 2.0)  # 2 seems a good default based on previous runs
        self.bond_angle_force_constant =  torch.nn.Parameter(torch.ones(self.idx_j_triples.shape[0]))
        self.register_buffer('C6_embedding', build_dispersion_embedding())
        # nn.init.uniform_(self.C6_embedding.weight.data, a=-1, b=0)
        self.coulomb_trainable_constant = torch.nn.Parameter(torch.tensor([100.]))

    def set_terms(self, terms):
        self.terms = terms

    def forward(self, inputs: Dict[str, torch.Tensor]):
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        R_ij_full = inputs[properties.Rij]
        D_ij_full = torch.norm(R_ij_full, dim=1)
        R_ij_bonded = R_ij_full[self.bonded_mask]
        D_ij_bonded = D_ij_full[self.bonded_mask]
        energy_terms = []

        # bond lengths
        if 'all' in self.terms or 'two-body' in self.terms:
            E_bond_distance = 0.5 * self.bond_distance_force_constant * (D_ij_bonded - self.bond_distance_equilibrium)**2
            E_bond_distance_atomwise = \
                snn.scatter_add(E_bond_distance, self.idx_i_bonded, dim_size=len(atomic_numbers), dim=0) +\
                snn.scatter_add(E_bond_distance, self.idx_j_bonded, dim_size=len(atomic_numbers), dim=0)
            energy_terms.append(E_bond_distance_atomwise[:, None])

        # bond angles
        if 'all' in self.terms or 'three-body' in self.terms:
            bond_angles = torch.acos(torch.einsum('bi,bi->b', R_ij_bonded[self.idx_j_triples], R_ij_bonded[self.idx_k_triples]) / (torch.linalg.norm(R_ij_bonded[self.idx_j_triples], dim=1) * torch.linalg.norm(R_ij_bonded[self.idx_k_triples], dim=1)))
            E_bond_angle = 0.33 * self.bond_angle_force_constant * (self.bond_angle_equilibrium - bond_angles)**2
            # it's computationally wasteful to add all three
            # terms when one would suffice (and then removing the
            # 0.33 factor above), but we do this for the Fij
            # interpretability analysis
            E_bond_angle_atomwise = \
                snn.scatter_add(E_bond_angle, self.idx_i_triples, dim_size=len(atomic_numbers), dim=0) +\
                snn.scatter_add(E_bond_angle, self.idx_j_bonded[self.idx_j_triples], dim_size=len(atomic_numbers), dim=0) +\
                snn.scatter_add(E_bond_angle, self.idx_j_bonded[self.idx_k_triples], dim_size=len(atomic_numbers), dim=0)
            energy_terms.append(E_bond_angle_atomwise[:, None])

        # dispersion
        if 'all' in self.terms or 'dispersion' in self.terms:
            idx_i_dispersion = self.idx_i_full[self.one_three_nonbonded_mask]
            idx_j_dispersion = self.idx_j_full[self.one_three_nonbonded_mask]
            C6_at_idx_i = self.C6_embedding[atomic_numbers[idx_i_dispersion]]
            C6_at_idx_j = self.C6_embedding[atomic_numbers[idx_j_dispersion]]
            C6 = torch.sqrt(C6_at_idx_i * C6_at_idx_j) # geometric mean

            # debug printing
            if torch.rand(1).item() < 0.01:
                # print('\nc6')
                # print(self.C6_embedding(torch.tensor([1,6,7,8]).to(idx_i_dispersion.device)))
                print('\nc6 min, max')
                print(C6.min(), C6.max())
                print('elec')
                print(self.coulomb_trainable_constant)
            E_dispersion = - 0.5 * C6 / D_ij_full[self.one_three_nonbonded_mask].pow(6)
            # print(f'E_dispersion {E_dispersion[0]}')
            E_dispersion_atomwise = \
                snn.scatter_add(E_dispersion, self.idx_i_full[self.one_three_nonbonded_mask], dim_size=len(atomic_numbers), dim=0) +\
                snn.scatter_add(E_dispersion, self.idx_j_full[self.one_three_nonbonded_mask], dim_size=len(atomic_numbers), dim=0)
            energy_terms.append(E_dispersion_atomwise[:, None])

        # elec/coulomb
        if 'all' in self.terms or 'elec' in self.terms:
            E_coulomb = \
                0.5 * self.coulomb_trainable_constant \
                * self.charges[self.idx_i_full[self.one_three_nonbonded_mask]] \
                * self.charges[self.idx_j_full[self.one_three_nonbonded_mask]] \
                / D_ij_full[self.one_three_nonbonded_mask]
            E_coulomb_atomwise = \
                snn.scatter_add(E_coulomb, self.idx_i_full[self.one_three_nonbonded_mask], dim_size=len(atomic_numbers), dim=0) +\
                snn.scatter_add(E_coulomb, self.idx_j_full[self.one_three_nonbonded_mask], dim_size=len(atomic_numbers), dim=0)
            energy_terms.append(E_coulomb_atomwise[:, None])

        inputs["scalar_representation"] = torch.sum(torch.stack(energy_terms, dim=0), dim=0)
        return inputs
