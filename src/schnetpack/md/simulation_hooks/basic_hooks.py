from __future__ import annotations
import torch.nn as nn

from schnetpack.md.utils import UninitializedMixin
from schnetpack import properties

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schnetpack.md import Simulator

__all__ = ["RemoveCOMMotion", "SimulationHook", "WrapPositions", "Maltes_disabling"]


class SimulationHook(UninitializedMixin, nn.Module):
    """
    Basic class for simulator hooks
    """

    def on_step_begin(self, simulator: Simulator):
        pass

    def on_step_middle(self, simulator: Simulator):
        pass

    def on_step_end(self, simulator: Simulator):
        pass

    def on_step_finalize(self, simulator: Simulator):
        pass

    def on_step_failed(self, simulator: Simulator):
        pass

    def on_simulation_start(self, simulator: Simulator):
        pass

    def on_simulation_end(self, simulator: Simulator):
        pass


class RemoveCOMMotion(SimulationHook):
    """
    Periodically remove motions of the center of mass from the system.

    Args:
        every_n_steps (int): Frequency with which motions are removed.
        remove_rotation (bool): Also remove rotations.
    """

    def __init__(self, every_n_steps: int, remove_rotation: bool):
        super(RemoveCOMMotion, self).__init__()
        self.every_n_steps = every_n_steps
        self.remove_rotation = remove_rotation

    def on_step_finalize(self, simulator: Simulator):
        if simulator.step % self.every_n_steps == 0:
            simulator.system.remove_center_of_mass()
            simulator.system.remove_translation()

            if self.remove_rotation:
                simulator.system.remove_com_rotation()


class WrapPositions(SimulationHook):
    """
    Periodically wrap atoms back into simulation cell.

    Args:
        every_n_steps (int): Frequency with which atoms should be wrapped.
    """

    def __init__(self, every_n_steps: int):
        super(WrapPositions, self).__init__()
        self.every_n_steps = every_n_steps

    def on_step_finalize(self, simulator: Simulator):
        if simulator.step % self.every_n_steps == 0:
            simulator.system.wrap_positions()


def check_md_stability(positions_start=None, positions_end=None, positions=None, too_many_bonds_criterion=6, bond_cutoff=2):
    if positions is not None:
        positions_start = positions[0]
        positions_end = positions[-1]
    if np.any(np.abs(positions_end) > 1e15):
        print('Instable! Positions encountered overflow')
        return False
    connectivity_start = np.linalg.norm(positions_start[:, None, :] - positions_start[None, :, :], axis=2)
    connectivity_start = connectivity_start < bond_cutoff
    np.fill_diagonal(connectivity_start, 0)
    if np.any(connectivity_start.sum(axis=0) == 0):
        raise Exception('Already in the starter config, no bonds')
    if np.any(connectivity_start.sum(axis=0) >= too_many_bonds_criterion):
        raise Exception('Already in the starter config, too many bonds')

    D_end = np.linalg.norm(positions_end[:, None, :] - positions_end[None, :, :], axis=2)
    connectivity_end = D_end < bond_cutoff
    np.fill_diagonal(connectivity_end, 0)

    connectivity_both = connectivity_start & connectivity_end
    connections_per_atom = connectivity_both.sum(axis=0)  # which axis doesn't matter

    if positions is not None:
        D_all = np.linalg.norm(positions[:, None, :, :] - positions[:, :, None, :], axis=3)
        D_all[:, np.arange(D_all.shape[1]), np.arange(D_all.shape[2])] = 1.0

    if np.any(connections_per_atom == 0):
        print('Instable! Bond breaking')
        return False
    elif np.any(connections_per_atom >= too_many_bonds_criterion):
        print('Instable! Too many bonds')
        return False
    elif positions is not None and np.any(D_all > 21):  # THIS IS A BAD HACK
        print('Instable! A distance was > 21 A. HACK!')
        return False
    elif positions is None and np.any(D_end > 21):
        print('Instable! A distance was > 21 A. HACK!')
        return False
    # elif positions is not None and np.any(D_all < 0.6):
    #     print('Instable! Two atoms got too close (<0.6A)')
    else:
        return True


class Maltes_disabling(SimulationHook):
    def __init__(self, every_n_steps: int):
        super().__init__()
        self.every_n_steps = every_n_steps

    def on_step_finalize(self, simulator: Simulator):
        if simulator.step == 0:
            molecules = simulator.calculator._get_system_molecules(simulator.system)
            self.positions_start = molecules[properties.R].split(list(molecules[properties.n_atoms]), dim=0)
        if simulator.step % self.every_n_steps == 0:
            disabled_molecules = simulator.calculator.neighbor_list.disabled_molecules
            molecules = simulator.calculator._get_system_molecules(simulator.system)
            cur_pos = molecules[properties.R].split(list(molecules[properties.n_atoms]), dim=0)
            for mol_idx in range(len(cur_pos)):
                if mol_idx not in disabled_molecules:
                    if not check_md_stability(positions_start=self.positions_start[mol_idx].detach().cpu().numpy(),
                                              positions_end=cur_pos[mol_idx].detach().cpu().numpy()):
                        print(f'Disabling molecule {mol_idx} on step {simulator.step}')
                        simulator.calculator.neighbor_list.disable_molecule(mol_idx)
