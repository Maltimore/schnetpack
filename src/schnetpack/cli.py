import logging
import os
import uuid
import tempfile
import socket
from typing import List
import random

import torch
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import CombinedLoader

import schnetpack as spk
from schnetpack.utils import str2class
from schnetpack.utils.script import log_hyperparameters, print_config
from schnetpack.data import BaseAtomsData, AtomsLoader
from schnetpack.train import PredictionWriter
from schnetpack import properties
from schnetpack.utils import load_model


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)

header = """
   _____      __    _   __     __  ____             __
  / ___/_____/ /_  / | / /__  / /_/ __ \____ ______/ /__
  \__ \/ ___/ __ \/  |/ / _ \/ __/ /_/ / __ `/ ___/ //_/
 ___/ / /__/ / / / /|  /  __/ /_/ ____/ /_/ / /__/ ,<
/____/\___/_/ /_/_/ |_/\___/\__/_/    \__,_/\___/_/|_|
"""


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def train(config: DictConfig):
    """
    General training routine for all models defined by the provided hydra configs.

    """
    print(header)
    log.info("Running on host: " + str(socket.gethostname()))

    if OmegaConf.is_missing(config, "run.data_dir"):
        log.error(
            f"Config incomplete! You need to specify the data directory `data_dir`."
        )
        return

    if not "model" in config: 
        log.error(
            f"""
        Config incomplete! You did not specify "model"
        For an example, try one of our pre-defined experiments:
        > spktrain experiment=qm9_atomwise
        """
        )
        return
    if not "data" in config and not "datasets" in config:
        log.error(
            f"""
        Config incomplete! You did not specify "data" or "datasets".
        For an example, try one of our pre-defined experiments:
        > spktrain experiment=qm9_atomwise
        """
        )
        return
    if 'data' in config.keys() and 'datasets' in config.keys() and config['datasets'] is not None:
        log.error('Config error: only one key of data or datasets may be specified')
        return

    if os.path.exists("config.yaml"):
        log.info(
            f"Config already exists in given directory {os.path.abspath('.')}."
            + " Attempting to continue training."
        )

        # save old config
        old_config = OmegaConf.load("config.yaml")
        count = 1
        while os.path.exists(f"config.old.{count}.yaml"):
            count += 1
        with open(f"config.old.{count}.yaml", "w") as f:
            OmegaConf.save(old_config, f, resolve=False)

        # resume from latest checkpoint
        if config.run.ckpt_path is None:
            if os.path.exists("checkpoints/last.ckpt"):
                config.run.ckpt_path = "checkpoints/last.ckpt"

        if config.run.ckpt_path is not None:
            log.info(
                f"Resuming from checkpoint {os.path.abspath(config.run.ckpt_path)}"
            )
    else:
        with open("config.yaml", "w") as f:
            OmegaConf.save(config, f, resolve=False)

    # Set matmul precision if specified
    if "matmul_precision" in config and config.matmul_precision is not None:
        log.info(f"Setting float32 matmul precision to <{config.matmul_precision}>")
        torch.set_float32_matmul_precision(config.matmul_precision)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
    else:
        # choose seed randomly
        with open_dict(config):
            config.seed = random.randint(0, 2**32 - 1)
        log.info(f"Seed randomly with <{config.seed}>")
    seed_everything(seed=config.seed, workers=True)

    if config.get("print_config"):
        print_config(config, resolve=False)

    if not os.path.exists(config.run.data_dir):
        os.makedirs(config.run.data_dir)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodules")
    datamodule_dict = {}
    # if `data` is given (and therefore, not `datasets`),
    # transform `data` into a dict similar to `config.datasets`
    if 'data' in config.keys():
        if 'name' in config.data.keys():
            dataset_key = config.data.pop('name')
        else:
            dataset_key = 'default'
        dataset_configs = {dataset_key: config.data}
    else:
        dataset_configs = config.datasets
    for dataset_key, dataset_config in dataset_configs.items():
        datamodule: LightningDataModule = hydra.utils.instantiate(
            dataset_config,
            train_sampler_cls=(
                str2class(dataset_config.train_sampler_cls)
                if dataset_config.train_sampler_cls
                else None
            ),
            dataset_name=dataset_key,
        )
        datamodule.setup()
        datamodule_dict[dataset_key] = datamodule

    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    # Init LightningModule
    log.info(f"Instantiating task <{config.task._target_}>")
    scheduler_cls = (
        str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
    )

    task: spk.AtomisticTask = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )
    task.add_datamodule(datamodule)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[Logger] = []

    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                l = hydra.utils.instantiate(lg_conf)

                logger.append(l)

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=os.path.join(config.run.id),
        _convert_="partial",
    )

    log.info("Logging hyperparameters.")
    log_hyperparameters(config=config, model=task, trainer=trainer)

    # Train the model
    log.info("Starting training.")
    iterables = {}
    for dataset_key, datamodule in datamodule_dict.items():
        if datamodule.disable_training:
            continue
        iterables[dataset_key] = datamodule.train_dataloader()
    train_dataloader = CombinedLoader(
        iterables=iterables,
        mode='max_size_cycle'
    )
    val_dataloader = CombinedLoader(
        iterables={dataset_key:
            datamodule.val_dataloader() for
            dataset_key, datamodule in
            datamodule_dict.items()
        },
        mode='max_size_cycle'
    )

    trainer.fit(
        model=task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=config.run.ckpt_path
    )

    # Evaluate model on test set after training
    log.info("Starting testing.")
    trainer.test(
        model=task,
        test_dataloaders=CombinedLoader(
            iterables={dataset_key: datamodule.test_dataloader() for dataset_key, datamodule in datamodule_dict.items()},
            mode='max_size_cycle'
        ),
        ckpt_path="best"
    )

    # Store best model
    best_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint path:\n{best_path}")

    log.info(f"Store best model")
    best_task = type(task).load_from_checkpoint(best_path)
    torch.save(best_task, config.globals.model_path + ".task")

    best_task.save_model(config.globals.model_path, do_postprocessing=True)
    log.info(f"Best model stored at {os.path.abspath(config.globals.model_path)}")


@hydra.main(config_path="configs", config_name="predict", version_base="1.2")
def predict(config: DictConfig):
    log.info(f"Load data from `{config.data.datapath}`")
    dataset: BaseAtomsData = hydra.utils.instantiate(config.data)
    loader = AtomsLoader(dataset, batch_size=config.batch_size, num_workers=8)

    model = load_model("best_model")

    class WrapperLM(LightningModule):
        def __init__(self, model, enable_grad=config.enable_grad):
            super().__init__()
            self.model = model
            self.enable_grad = enable_grad

        def forward(self, x):
            return model(x)

        def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
            torch.set_grad_enabled(self.enable_grad)
            results = self(batch)
            results[properties.idx_m] = batch[properties.idx][batch[properties.idx_m]]
            results = {k: v.detach().cpu() for k, v in results.items()}
            return results

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=[
            PredictionWriter(
                output_dir=config.outputdir,
                write_interval=config.write_interval,
                write_idx=config.write_idx_m,
            )
        ],
        default_root_dir=".",
        _convert_="partial",
    )
    trainer.predict(
        WrapperLM(model, config.enable_grad),
        dataloaders=loader,
        ckpt_path=config.ckpt_path,
    )
