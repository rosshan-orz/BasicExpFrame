import argparse
import os
from pathlib import Path
from typing import Dict, Any, Union
import torch
from box import Box
from datetime import datetime

from src.utils.config_parser import ConfigParser
from src.utils.logger import BaseLogger
from src.utils.seeds import seed_setup
from src.utils.checkpoint import save_checkpoint   # for emergency save
from src.dataset.builder import DataBuilder
from src.utils.registry import MODEL_REGISTRY, METRIC_REGISTRY, CRITERION_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
from src.metrics.manager import MetricManager
from src.trainer.trainer import Trainer
from src.trainer.builder import build_optimizer, build_scheduler

import src.models
import src.metrics
import src.dataset.splitters
import src.trainer.criterion

def main(config_path: Union[str, Path]):
    """

    :param config_path:
    :return:
    """
    frozen_config = ConfigParser.load(config_path)
    config_dict = frozen_config.to_dict()
    config = Box(config_dict)
    print(f"Configuration loaded from {config_path}")
    print(f"Experiment: {config.experiment_name}")

    # setup global settings
    seed_setup(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)
    print(f"Using device: {config.device}")

    # output directory confirmation
    base_output_dir = Path(config.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Basic Output directory: {base_output_dir}")

    # Experiment loop
    for exp_name, train_loader, valid_loader, test_loader in DataBuilder.build_experiments(config=config):

        model_name = config.model.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = base_output_dir / model_name / exp_name / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)
        print("##############################################################")
        print(f"Model {model_name} Experiment {exp_name} directory: {exp_dir}")

        #checkpoint and log
        config.trainer.save_dir = str(exp_dir)
        logger = BaseLogger(exp_dir, config)
        logger.info(f"Loaded config: \n{config.to_yaml()}")

        # model
        model = MODEL_REGISTRY.build(config.model)
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # criterion
        criterion = CRITERION_REGISTRY.build(config.criterion)
        logger.info(f"Criterion: {criterion.__class__.__name__}")

        # optimizer
        optimizer = build_optimizer(config, model.parameters())
        logger.info(f"Optimizer: {optimizer.__class__.__name__}")

        # scheduler
        scheduler = build_scheduler(config, optimizer)
        if scheduler:
            logger.info(f"Scheduler: {scheduler.__class__.__name__}")
        else:
            logger.info("No learning rate scheduler configured")

        # metrics
        metrics_manager = MetricManager(config.metrics)
        logger.info(f"Metrics managed and to monitor: {list(metrics_manager.metrics.keys())}")

        # Initialize and run Trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            criterion=criterion,
            config=config,
            logger=logger,
            metrics_manager=metrics_manager,
            device=device,
        )

        try:
            trainer.run()
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user. Emergency save.")
            trainer.emergency_save(
                epoch=trainer.current_epoch if hasattr(trainer, 'current_epoch') else 0,
                file_name="emergency_save.pth"
            )
        finally:
            logger.info(f"Cleaning up resources for experiment {exp_name}")
            logger.close()
            del model, trainer, optimizer, criterion, metrics_manager
            if scheduler:
                del scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Experiment {exp_name} completed or terminated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a machine learning experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    config_file_path = Path(args.config)
    if not config_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file_path}")

    main(config_file_path)