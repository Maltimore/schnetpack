from typing import Callable, Dict, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn


__all__ = ["EmpiricalFF"]


class EmpiricalFF(nn.Module):
    def __init__(
        self,
        molecule_db_file: str,
    ):
        super(EmpiricalFF, self).__init__()
        # dummy value that is only used for interpretability F_ij analysis
        self.cutoff = torch.tensor(999.)

        # load bond_indices
        if molecule_db_file.startswith('Ac-Ala3-NHMe'):
            loaded = torch.load('/home/space/datasets/xai4qc/md22/empirical_ff_acal.pth', weights_only=True)
        else:
            raise Exception(f'no ff for molecule db file {molecule_db_file}')

        self.idx_i_full = loaded['idx_i_full']
        self.idx_j_full = loaded['idx_j_full']
        self.bonded_mask = loaded['bonded_mask']
        self.idx_i_bonded = loaded['idx_i_bonded']
        self.idx_j_bonded = loaded['idx_j_bonded']
        self.idx_i_triples = loaded[properties.idx_i_triples]
        self.idx_j_triples = loaded[properties.idx_j_triples]
        self.idx_k_triples = loaded[properties.idx_k_triples]



        self.bond_distance_equilibrium = torch.nn.Parameter(torch.ones(self.idx_i_bonded.shape[0]) * 1.5)
        self.bond_distance_force_constant =  torch.nn.Parameter(torch.ones(self.idx_i_bonded.shape[0]))
        self.bond_angle_equilibrium = torch.nn.Parameter(torch.ones(self.idx_j_triples.shape[0]) * 3.141)
        self.bond_angle_force_constant =  torch.nn.Parameter(torch.ones(self.idx_j_triples.shape[0]))
        self.C6_embedding = torch.nn.Embedding(9, 1)
        nn.init.uniform_(self.C6_embedding.weight.data, a=0.1, b=2.0)



    def forward(self, inputs: Dict[str, torch.Tensor]):
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        R_ij_full = inputs[properties.Rij]
        D_ij_full = torch.norm(R_ij_full, dim=1)
        R_ij_bonded = R_ij_full[self.bonded_mask]
        D_ij_bonded = D_ij_full[self.bonded_mask]
        energy_terms = []

        # bond lengths
        E_bond_distance = 0.5 * self.bond_distance_force_constant * (D_ij_bonded - self.bond_distance_equilibrium)**2
        E_bond_distance_atomwise = snn.scatter_add(E_bond_distance, self.idx_i_bonded, dim_size=len(atomic_numbers), dim=0)
        energy_terms.append(E_bond_distance_atomwise[:, None])

        # bond angles
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
        C6_at_idx_i = self.C6_embedding(atomic_numbers[self.idx_i_full])[:, 0]
        C6_at_idx_j = self.C6_embedding(atomic_numbers[self.idx_j_full])[:, 0]
        C6 = torch.sqrt(C6_at_idx_i * C6_at_idx_j) # geometric mean
        E_dispersion = 0.5 * C6 / D_ij_full.pow(6)
        E_dispersion_atomwise = snn.scatter_add(E_dispersion, self.idx_i_full, dim_size=len(atomic_numbers), dim=0)
        energy_terms.append(E_dispersion_atomwise[:, None])

        inputs["scalar_representation"] = torch.sum(torch.stack(energy_terms, dim=0), dim=0)
        return inputs

