from typing import Dict, Optional

import torch
from ase.data import atomic_masses

import schnetpack.properties as structure
from .base import Transform
from schnetpack.nn import scatter_add

__all__ = [
    "SubtractCenterOfMass",
    "SubtractCenterOfGeometry",
    "AddOffsets",
    "RemoveOffsets",
    "ScaleProperty",
    "Maltes_neighboring_elements_labels",
    "Maltes_repelling_forces",
]


class Maltes_repelling_forces(Transform):
    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self):
        super().__init__()
        self.elem_pair_to_repel_distance = {
            (1, 1): 1.65,  # H-H covalent bond would be .74, but the H's are non-bonded
            (1, 6): 1.038,  # 1.06 - 1.12
            (1, 7): 0.966,
            (1, 8): 1.71,
            (6, 6): 1.45,  # 1.20 - 1.55 (lower range is for triple bond)
            (6, 7): 1.31,  # 1.47 - 2.10
            (6, 8): 1.19,  # 1.43 - 2.15
        }

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        Z = inputs['_atomic_numbers']

        if '_maltes_r_ij' not in inputs.keys():
            r_ij = inputs['_positions'][idx_i] - inputs['_positions'][idx_j]
            inputs['_maltes_r_ij'] = r_ij
        else:
            r_ij = inputs['_maltes_r_ij']
        d_ij = torch.norm(r_ij, dim=1)

        repel_idxes_i = []
        repel_idxes_j = []
        for elem_pair, repel_distance in self.elem_pair_to_repel_distance.items():
            mask = (
                (
                 ((Z[idx_i] == elem_pair[0]) & (Z[idx_j] == elem_pair[1]))
                 |
                 ((Z[idx_j] == elem_pair[0]) & (Z[idx_i] == elem_pair[1]))
                )
                &
                (
                    d_ij < repel_distance
                )
            )
            repel_idxes_i.append(idx_i[mask])
            repel_idxes_j.append(idx_j[mask])
        inputs['_repel_idxes_i_pleasecollate'] = torch.concatenate(repel_idxes_i, dim=0)
        inputs['_repel_idxes_j_pleasecollate'] = torch.concatenate(repel_idxes_j, dim=0)

        return inputs


class Maltes_neighboring_elements_labels(Transform):
    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self):
        super().__init__()
        self.element_mapping = torch.tensor([
            -1,
            0, # H
            -1,
            -1,
            -1,
            -1,
            1, # C
            2, # N
            3, # O
            4, # F
        ])

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if '_maltes_r_ij' not in inputs.keys():
            r_ij = inputs['_positions'][inputs['_idx_i']] - inputs['_positions'][inputs['_idx_j']]
            inputs['_maltes_r_ij'] = r_ij
        else:
            r_ij = inputs['_maltes_r_ij']

        elements = inputs[structure.Z]
        inputs['elements_as_labels'] = self.element_mapping[inputs[structure.Z]]
        inputs['idx_j_elements_as_labels'] = self.element_mapping[inputs[structure.Z][inputs['_idx_j']]]
        if torch.any(inputs['elements_as_labels'] < 0) or torch.any(inputs['idx_j_elements_as_labels'] < 0):
            raise Exception('neighbor elems computation went wrong')
        return inputs


class SubtractCenterOfMass(Transform):
    """
    Subtract center of mass from positions.
    """
    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self):
        super().__init__()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        masses = torch.tensor(atomic_masses[inputs[structure.Z]])
        inputs[structure.position] -= (
            masses.unsqueeze(-1) * inputs[structure.position]
        ).sum(0) / masses.sum()
        return inputs


class SubtractCenterOfGeometry(Transform):
    """
    Subtract center of geometry from positions.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs[structure.position] -= inputs[structure.position].mean(0)
        return inputs


class RemoveOffsets(Transform):
    """
    Remove offsets from property based on the mean of the training data and/or the
    single atom reference calculations.

    The `mean` and/or `atomref` are automatically obtained from the AtomsDataModule,
    when it is used. Otherwise, they have to be provided in the init manually.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def __init__(
        self,
        property,
        remove_mean: bool = False,
        remove_atomrefs: bool = False,
        is_extensive: bool = True,
        zmax: int = 100,
        atomrefs: torch.Tensor = None,
        property_mean: torch.Tensor = None,
        estimate_atomref: bool = False,
    ):
        """
        Args:
            property: The property to add the offsets to.
            remove_mean: If true, remove mean of the dataset from property.
            remove_atomrefs: If true, remove single-atom references.
            is_extensive: Set true if the property is extensive.
            zmax: Set the maximum atomic number, to determine the size of the atomref
                tensor.
            atomrefs: Provide single-atom references directly.
            property_mean: Provide mean property value / n_atoms.
        """
        super().__init__()
        self._property = property
        self.remove_mean = remove_mean
        self.remove_atomrefs = remove_atomrefs
        self.is_extensive = is_extensive
        self.estimate_atomref = estimate_atomref

        assert not (
            estimate_atomref and atomrefs is not None
        ), "You can not set `atomrefs` and use `estimate_atomrefs=True!`"

        if atomrefs is not None:
            self._atomrefs_initialized = True
        else:
            self._atomrefs_initialized = False

        if property_mean is not None:
            self._mean_initialized = True
        else:
            self._mean_initialized = False

        if self.remove_atomrefs:
            atomrefs = atomrefs or torch.zeros((zmax,))
            self.register_buffer("atomref", atomrefs)
        if self.remove_mean:
            property_mean = property_mean or torch.zeros((1,))
            self.register_buffer("mean", property_mean)

    def datamodule(self, _datamodule):
        """
        Sets mean and atomref automatically when using PyTorchLightning integration.
        """
        if self.remove_atomrefs and not self._atomrefs_initialized:
            if self.estimate_atomref:
                atrefs = _datamodule.get_atomrefs(
                    property=self._property, is_extensive=self.is_extensive
                )
            else:
                atrefs = _datamodule.train_dataset.atomrefs
            self.atomref = atrefs[self._property].detach()

        if self.remove_mean and not self._mean_initialized:
            stats = _datamodule.get_stats(
                self._property, self.is_extensive, self.remove_atomrefs
            )
            self.mean = stats[0].detach()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.remove_mean:
            mean = (
                self.mean * inputs[structure.n_atoms]
                if self.is_extensive
                else self.mean
            )
            inputs[self._property] -= mean
        if self.remove_atomrefs:
            atomref_bias = torch.sum(self.atomref[inputs[structure.Z]])
            if not self.is_extensive:
                atomref_bias /= inputs[structure.n_atoms].item()
            inputs[self._property] -= atomref_bias

        return inputs


class ScaleProperty(Transform):
    """
    Scale an entry of the input or results dioctionary.

    The `scale` can be automatically obtained from the AtomsDataModule,
    when it is used. Otherwise, it has to be provided in the init manually.

    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def __init__(
        self,
        input_key: str,
        target_key: str = None,
        output_key: str = None,
        scale_by_mean: bool = False,
        scale: torch.Tensor = None,
    ):
        """
        Args:
            input_key: dict key of input to be scaled
            target_key: dict key of target to derive scaling from
                (either its mean or std dev)
            output_key: dict key for scaled output
            scale_by_mean: if true, use the mean of the target variable for scaling,
                otherwise use its standard deviation
            scale: provide the scale of the property manually.
        """
        super().__init__()
        self.input_key = input_key
        self._target_key = target_key or input_key
        self.output_key = output_key or input_key
        self._scale_by_mean = scale_by_mean
        self.model_outputs = [self.output_key]

        if scale is not None:
            self._initialized = True
        else:
            self._initialized = False

        scale = scale or torch.ones((1,))
        self.register_buffer("scale", scale)

    def datamodule(self, _datamodule):
        if not self._initialized:
            stats = _datamodule.get_stats(self._target_key, True, False)
            scale = stats[0] if self._scale_by_mean else stats[1]
            self.scale = torch.abs(scale).detach()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs[self.output_key] = inputs[self.input_key] * self.scale
        return inputs


class AddOffsets(Transform):
    """
    Add offsets to property based on the mean of the training data and/or the single
    atom reference calculations.

    The `mean` and/or `atomref` are automatically obtained from the AtomsDataModule,
    when it is used. Otherwise, they have to be provided in the init manually.

    Hint:
        Place this postprocessor after casting to float64 for higher numerical
        precision.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True
    atomref: torch.Tensor

    def __init__(
        self,
        property,
        add_mean: bool = False,
        add_atomrefs: bool = False,
        is_extensive: bool = True,
        zmax: int = 100,
        atomrefs: torch.Tensor = None,
        property_mean: torch.Tensor = None,
        estimate_atomref: bool = False,
    ):
        """
        Args:
            property: The property to add the offsets to.
            add_mean: If true, add mean of the dataset.
            add_atomrefs: If true, add single-atom references.
            is_extensive: Set true if the property is extensive.
            zmax: Set the maximum atomic number, to determine the size of the atomref
                tensor.
            atomrefs: Provide single-atom references directly.
            property_mean: Provide mean property value / n_atoms.
        """
        super().__init__()
        self._property = property
        self.add_mean = add_mean
        self.add_atomrefs = add_atomrefs
        self.is_extensive = is_extensive
        self._aggregation = "sum" if self.is_extensive else "mean"
        self.estimate_atomref = estimate_atomref

        assert not (
            estimate_atomref and atomrefs is not None
        ), "You can not set `atomrefs` and use `estimate_atomrefs=True!`"

        if atomrefs is not None:
            self._atomrefs_initialized = True
        else:
            self._atomrefs_initialized = False

        if property_mean is not None:
            self._mean_initialized = True
        else:
            self._mean_initialized = False

        atomrefs = atomrefs or torch.zeros((zmax,))
        property_mean = property_mean or torch.zeros((1,))
        self.register_buffer("atomref", atomrefs)
        self.register_buffer("mean", property_mean)

    def datamodule(self, _datamodule):
        if self.add_atomrefs and not self._atomrefs_initialized:
            if self.estimate_atomref:
                atrefs = _datamodule.get_atomrefs(
                    property=self._property, is_extensive=self.is_extensive
                )
            else:
                atrefs = _datamodule.train_dataset.atomrefs
            self.atomref = atrefs[self._property].detach()

        if self.add_mean and not self._mean_initialized:
            stats = _datamodule.get_stats(
                self._property, self.is_extensive, self.add_atomrefs
            )
            self.mean = stats[0].detach()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.add_mean:
            mean = (
                self.mean * inputs[structure.n_atoms]
                if self.is_extensive
                else self.mean
            )
            inputs[self._property] += mean

        if self.add_atomrefs:
            idx_m = inputs[structure.idx_m]
            y0i = self.atomref[inputs[structure.Z]]
            maxm = int(idx_m[-1]) + 1

            y0 = scatter_add(y0i, idx_m, dim_size=maxm)

            if not self.is_extensive:
                y0 /= inputs[structure.n_atoms]

            inputs[self._property] += y0

        return inputs
