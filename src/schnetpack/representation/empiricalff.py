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
        if molecule_db_file.startswith('Ac-Ala3-NHMe'):
            pass # TODO load

        # self.pairwise_equilibrium = torch.ones(self.bonded_i.shape[0]) * 1.5
        # self.pairwise_force_constant =  torch.ones(self.bonded_i.shape[0])



    def forward(self, inputs: Dict[str, torch.Tensor]):
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        R = inputs[properties.R]
        breakpoint()

        d_ij = torch.norm(R_ij, dim=1, keepdim=True)


        inputs["scalar_representation"] = TODO
        return inputs

