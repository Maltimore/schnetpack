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
        # load bond_indices
        if molecule_db_file.startswith('Ac-Ala3-NHMe'):
            loaded = torch.load('/home/space/datasets/xai4qc/md22/empirical_ff_acal.pth', weights_only=True)

        self.idx_i_full = loaded['idx_i_full']
        self.idx_j_full = loaded['idx_j_full']
        self.bonded_mask = loaded['bonded_mask']
        self.idx_i_bonded = loaded['idx_i_bonded']
        self.idx_j_bonded = loaded['idx_j_bonded']

        self.pairwise_equilibrium = torch.nn.Parameter(torch.ones(self.idx_i_bonded.shape[0]) * 1.5)
        self.pairwise_force_constant =  torch.nn.Parameter(torch.ones(self.idx_i_bonded.shape[0]))



    def forward(self, inputs: Dict[str, torch.Tensor]):
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        R_ij_full = inputs[properties.Rij]
        D_ij_full = torch.norm(R_ij_full, dim=1)
        R_ij_bonded = R_ij_full[self.bonded_mask]
        D_ij_bonded = D_ij_full[self.bonded_mask]

        # bond lengths
        E_bond = 0.5 * self.pairwise_force_constant * (D_ij_bonded - self.pairwise_equilibrium)**2
        E_bond_atomwise = snn.scatter_add(E_bond, self.idx_i_bonded, dim_size=len(atomic_numbers), dim=0)

        inputs["scalar_representation"] = E_bond_atomwise[:, None]
        return inputs

