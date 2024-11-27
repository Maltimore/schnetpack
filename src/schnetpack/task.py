import warnings
from typing import Optional, Dict, List, Type, Any

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torchmetrics import Metric

from schnetpack.model.base import AtomisticModel

__all__ = ["ModelOutput", "AtomisticTask", "UnsupervisedModelOutput", "Maltes_partial_forces_loss", "Maltes_norm_loss"]


class Maltes_partial_forces_loss(nn.Module):
    def __call__(self, pred, batch):
        if 'partial_forces' not in pred.keys():
            return torch.tensor(0.0)

        partial_forces_list = torch.split(pred['partial_forces'], batch['_n_atoms'].tolist(), dim=0)
        positions_list = torch.split(batch['_positions'], batch['_n_atoms'].tolist(), dim=0)
        diag_zeroing_masks = [
            torch.ones(
                (batch['_n_atoms'][molecule_idx].item(), batch['_n_atoms'][molecule_idx]),
                device=positions_list[0].device,
                dtype=positions_list[0].dtype
            ) - torch.eye(batch['_n_atoms'][molecule_idx].item(), device=positions_list[0].device)
            for molecule_idx in range(len(batch['_n_atoms']))
        ]
        loss_terms_list = []
        for molecule_idx in range(len(batch['_n_atoms'])):
            partials = partial_forces_list[molecule_idx][:, 0:batch['_n_atoms'][molecule_idx].item(), :]
            positions = positions_list[molecule_idx]
            r_ij = positions[:, None, :] - positions[None, :, :]
            D = torch.norm(r_ij, dim=2)
            cosine_sim_force_pairs = torch.nn.functional.cosine_similarity(
                partials,
                partials.transpose(1, 0).clone().detach(),
                dim=2,
                eps=1e-18,
            ).div((D + 1e-3)).sum(axis=0).mean()
            # the cosine on the diagonal should be 1, where the gradient is 0,
            # but I don't trust it so we multiply the diagonal by 0 to be sure
            cosine_sim_force_pairs = (cosine_sim_force_pairs * diag_zeroing_masks[molecule_idx]).mean()
            # loss for making sure nurms of pairs are equal
            squared_distance_force_norms = ((
                partials.norm(dim=2) - partials.clone().detach().transpose(1, 0).norm(dim=2)
            )**2).sum(axis=0).mean()
#            self_force_norm = (torch.diag(torch.norm(partials, dim=2))**2).mean()
            # note that on the diagonal of r_ij we get cosines of 0
            cosine_sim_force_r_ij = torch.nn.functional.cosine_similarity(
                partials,
                r_ij,
                dim=2,
                eps=1e-18,
            )
            # just to be sure that the diagonal elements do not contribute to the grad
            cosine_sim_force_r_ij_masked = cosine_sim_force_r_ij * diag_zeroing_masks[molecule_idx]
            force_to_rij_cosine_loss = (-torch.abs(cosine_sim_force_r_ij_masked)).div((D + 1e-3)).sum(axis=0).mean()
            loss_terms_list.append(
                cosine_sim_force_pairs.mean()
                + squared_distance_force_norms
#                + 0.01 * self_force_norm
                + force_to_rij_cosine_loss
            )
        final_loss = torch.stack(loss_terms_list).mean()
        return final_loss


class Maltes_norm_loss(nn.Module):
    def __call__(self, pred, batch):
        return pred['activations'].norm(dim=-1).mean()


class ModelOutput(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
    ):
        """
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
            constraints:
                constraint class for specifying the usage of model output in the loss function and logged metrics,
                while not changing the model output itself. Essentially, constraints represent postprocessing transforms
                that do not affect the model output but only change the loss value. For example, constraints can be used
                to neglect or weight some atomic forces in the loss function. This may be useful when training on
                systems, where only some forces are crucial for its dynamics.
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        if metrics is not None:
            self.train_metrics = nn.ModuleDict(metrics)
            self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
            self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
            self.metrics = {
                "train": self.train_metrics,
                "val": self.val_metrics,
                "test": self.test_metrics,
            }
        else:
            self.metrics = {}

        self.constraints = constraints or []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name], target[self.target_property])



class UnsupervisedModelOutput(ModelOutput):
    """
    Defines an unsupervised output of a model, i.e. an unsupervised loss or a regularizer
    that do not depend on label data. It includes mappings to the loss function,
    a weight for training and metrics to be logged.
    """
    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
    ):
        nn.Module.__init__(self)
        self.name = name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        if metrics is not None:
            self.train_metrics = nn.ModuleDict(metrics)
            self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
            self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
            self.metrics = {
                "train": self.train_metrics,
                "val": self.val_metrics,
                "test": self.test_metrics,
            }
        else:
            self.metrics = {}

        self.constraints = constraints or []

    def calculate_loss(self, pred, batch):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        loss = self.loss_weight * self.loss_fn(pred, batch)
        return loss

    def update_metrics(self, pred, batch, subset):
        for metric in self.metrics[subset].values():
            metric(pred, batch)


class AtomisticTask(pl.LightningModule):
    """
    The basic learning task in SchNetPack, which ties model, loss and optimizer together.

    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        """
        Args:
            model: the neural network model
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def add_datamodule(self, datamodule):
        self.datamodule = datamodule

    def setup(self, stage=None):
        if stage == "fit":
            self.model.initialize_transforms(self.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    def supervised_loss_fn(self, pred, batch):
        loss = 0.0
        for output in self.outputs:
            if isinstance(output, UnsupervisedModelOutput):
                continue
            loss += output.calculate_loss(pred, batch)
        return loss

    def unsupervised_loss_fn(self, pred, batch):
        loss = 0.0
        for output in self.outputs:
            if not isinstance(output, UnsupervisedModelOutput):
                continue
            loss += output.calculate_loss(pred, batch)
        return loss

    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            if subset not in output.metrics.keys():
                continue
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}_{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )

    def apply_constraints(self, pred, targets):
        for output in self.outputs:
            for constraint in output.constraints:
                pred, targets = constraint(pred, targets, output)
        return pred, targets

    def training_step(self, batch, batch_idx):
#        try:
#            targets["considered_atoms"] = batch["considered_atoms"]
#        except:
#            pass

        # SUPERVISED
        if 'supervised' in batch.keys():
            batch_ = batch['supervised']
            batch_new_reference = {k: v for k, v in batch_.items()}
            pred = self.predict_without_postprocessing(batch_new_reference)
            pred, targets = self.apply_constraints(pred, batch_)
            supervised_loss = self.supervised_loss_fn(pred, targets)
            self.log("train_loss", supervised_loss, on_step=True, on_epoch=False, prog_bar=False)
            self.log_metrics(pred, targets, "train")
        else:
            supervised_loss = 0.

        # UNSUPERVISED
        if 'unsuperbised' in batch.keys():
            batch_ = batch['unsupervised']
            batch_new_reference = {k: v for k, v in batch_.items()}
            pred = self.predict_without_postprocessing(batch_new_reference)
            pred, targets = self.apply_constraints(pred, batch_)
            unsupervised_loss = self.unsupervised_loss_fn(pred, targets)
            self.log("unsupervised_train_loss", supervised_loss, on_step=True, on_epoch=False, prog_bar=False)
        else:
            unsupervised_loss = 0.
        return supervised_loss + unsupervised_loss


    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        batch_new_reference = {k: v for k, v in batch.items()}
        pred = self.predict_without_postprocessing(batch_new_reference)
        pred, targets = self.apply_constraints(pred, batch)

        loss = self.supervised_loss_fn(pred, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        self.log_metrics(pred, targets, "val")

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        batch_new_reference = {k: v for k, v in batch.items()}
        pred = self.predict_without_postprocessing(batch_new_reference)
        pred, targets = self.apply_constraints(pred, batch)

        loss = self.supervised_loss_fn(pred, targets)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        self.log_metrics(pred, targets, "test")
        return {"test_loss": loss}

    def predict_without_postprocessing(self, batch):
        pp = self.model.do_postprocessing
        self.model.do_postprocessing = False
        pred = self(batch)
        self.model.do_postprocessing = pp
        return pred

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            # incase model is validated before epoch end (not recommended use of val_check_interval)
            if self.trainer.val_check_interval < 1.0:
                warnings.warn(
                    "Learning rate is scheduled after epoch end. To enable scheduling before epoch end, "
                    "please specify val_check_interval by the number of training epochs after which the "
                    "model is validated."
                )
            # incase model is validated before epoch end (recommended use of val_check_interval)
            if self.trainer.val_check_interval > 1.0:
                optimconf["interval"] = "step"
                optimconf["frequency"] = self.trainer.val_check_interval
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_closure=None,
    ):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: Optional[bool] = None):
        if self.global_rank == 0:
            pp_status = self.model.do_postprocessing
            if do_postprocessing is not None:
                self.model.do_postprocessing = do_postprocessing
            torch.save(self.model, path)
            self.model.do_postprocessing = pp_status


class ConsiderOnlySelectedAtoms(nn.Module):
    """
    Constraint that allows to neglect some atomic targets (e.g. forces of some specified atoms) for model optimization,
    while not affecting the actual model output. The indices of the atoms, which targets to consider in the loss
    function, must be provided in the dataset for each sample in form of a torch tensor of type boolean
    (True: considered, False: neglected).
    """

    def __init__(self, selection_name):
        """
        Args:
            selection_name: string associated with the list of considered atoms in the dataset
        """
        super().__init__()
        self.selection_name = selection_name

    def forward(self, pred, targets, output_module):
        """
        A torch tensor is loaded from the dataset, which specifies the considered atoms. Only the
        predictions of those atoms are considered for training, validation, and testing.

        :param pred: python dictionary containing model outputs
        :param targets: python dictionary containing targets
        :param output_module: torch.nn.Module class of a particular property (e.g. forces)
        :return: model outputs and targets of considered atoms only
        """

        considered_atoms = targets[self.selection_name].nonzero()[:, 0]

        # drop neglected atoms
        pred[output_module.name] = pred[output_module.name][considered_atoms]
        targets[output_module.target_property] = targets[output_module.target_property][
            considered_atoms
        ]

        return pred, targets
