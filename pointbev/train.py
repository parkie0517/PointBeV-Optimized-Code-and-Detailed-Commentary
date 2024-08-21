"""
    Original code from: https://github.com/valeoai/PointBeV
    Addition of comments and optimization by Heejun Park
"""

"""
    1. Environmental Setup
        - Import necesary modules
        - Configues the root directory
        - Setup logger
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import hydra # pip install hydra-core --upgrade
import pyrootutils
import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.profiler import PyTorchProfiler
from torch.profiler import ProfilerActivity

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pointbev import utils

# set up a logger for tracking and debugging
log = utils.get_pylogger(__name__)



"""
    4. train() function
        - configuration data is passed as an argument
"""
@utils.task_wrapper
def train(cfg: DictConfig) -> (Tuple[dict, dict]):
    if cfg.get("seed"): # if seed is null, then this block is not executed
        L.seed_everything(cfg.seed, workers=True)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    """
        4.1. Assign PointBeV to model variable
    """
    ckpt = utils.get_ckpt_from_path(cfg.ckpt.path)

    log.info(f"Instantiating model <{cfg.model._target_}>") # leave a log
    model: LightningModule = hydra.utils.instantiate(cfg.model) # instantiate PointBev
    model = utils.load_state_model(
        model,
        ckpt,
        cfg.ckpt.model.freeze,
        cfg.ckpt.model.load,
        verbose=1, # verbose(장황한): determines the level of information logged during the loading process
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    plugins = utils.instantiate_loggers(cfg.get("plugins"))
    if len(plugins) == 0:
        plugins = None

    if cfg.get("profile"):
        schedule = torch.profiler.schedule(
            skip_first=2, wait=1, warmup=1, active=2, repeat=1
        )
        profiler = PyTorchProfiler(
            # native args.
            filename="profile",
            export_to_chrome=False,
            # Kwargs
            schedule=schedule,
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            # Tensorboard profiling
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "./logs/profile/model"
            ),
        )
    else:
        profiler = None
    """
        4.2. Define the Trainer
    """
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        plugins=plugins,
        profiler=profiler,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        sys.stderr = open(Path(logger[0].save_dir) / "stdd.err", "a")

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)
    """
        4.3. Training
            - unlike conventional pytorch training loop, Trainer class helps make the training code cleaner
    """
    if cfg.get("train"):
        log.info("Starting training!") # leave a log
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt.path) # actual training

    train_metrics = trainer.callback_metrics # retrieve the training (and validation) results

    """
        4.4. Testing
    """
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics # retrieve the test results

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics} # merge two dictionaries

    return metric_dict, object_dict



"""
    3. main function
        - @hydra.main : decorator used to ease the execution of the main function using the specified configuration
        - the decorator passes the configuration data as a dictionary format
        - main() gets the configuration data as the cfg variable
        - main() returns float value
"""
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> (Optional[float]): # Added '(' and ')' around the return type to avoid error
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # cfg will now contain the data from 'train.yaml'

    utils.modif_config_based_on_flags(cfg) # go check pointbev/utils/launch.py
    utils.extras(cfg) # pointbev/utils/utils/extras()

    # train the model
    metric_dict, _ = train(cfg) # ignore the 2nd value

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value



"""
    2. Checking if the code is ran directly
        - only execute the code below if this file is being ran as the main file
"""
if __name__ == "__main__":
    main() # go to section 3