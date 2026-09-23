"""Configuration-driven experiment entry point."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml
import lightning.pytorch as pl

# Allow ``python project/main.py`` to work without requiring package installation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import BaseLogger, load_config, set_seed
from src.data import ExperimentPlanner
from src.model import build_criterion, build_metrics, build_model
from src.trainer import (
    ExperimentDataModule,
    ExperimentLightningModule,
    build_callbacks,
    create_trainer,
)


def main(config_path: str) -> None:
    config = load_config(config_path)

    set_seed(config.seed)
    print(f"Configuration loaded from {config_path}")
    print(f"Experiment: {config.experiment_name}")

    base_output_dir = PROJECT_ROOT / config.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_output_dir / config.experiment_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = BaseLogger(exp_dir)
    logger.info(f"Loaded config:\n{yaml.safe_dump(config.to_dict(), allow_unicode=True)}")
    logger.info(f"Output directory: {exp_dir}")

    if "data" not in config:
        logger.info("Core infrastructure initialized successfully.")
        logger.info("Data/model/trainer integration is not configured.")
        logger.close()
        return

    planner = ExperimentPlanner.from_config(config)
    results = []
    trainer_config = dict(config.get("trainer", {}) or {})
    monitor = trainer_config.pop("monitor", "val/loss")
    monitor_mode = trainer_config.pop("monitor_mode", "min")
    callback_config = trainer_config.pop("callbacks", {}) or {}
    if config.get("device") == "cpu":
        trainer_config.setdefault("accelerator", "cpu")
        trainer_config.setdefault("devices", 1)
    for experiment in planner:
        experiment_dir = exp_dir / experiment.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        experiment_logger = BaseLogger(experiment_dir)
        data_config = config.data.get("loader", {}) or {}
        data_module = ExperimentDataModule(
            experiment,
            batch_size=data_config.get("batch_size", 1),
            num_workers=data_config.get("num_workers", 0),
            pin_memory=data_config.get("pin_memory"),
        )
        model = build_model(config.model)
        criterion = build_criterion(config.get("loss", {"name": "CrossEntropyLoss"}))
        metrics = build_metrics(config.get("metrics", []))
        lightning_module = ExperimentLightningModule(
            model,
            criterion,
            config.get("optimizer", {"name": "Adam", "params": {"lr": 1e-3}}),
            scheduler_config=config.get("scheduler"),
            metrics=metrics,
        )
        callback_config = dict(callback_config)
        checkpoint_config = dict(callback_config.get("checkpoint", {}) or {})
        checkpoint_config.setdefault("dirpath", str(experiment_dir / "checkpoints"))
        checkpoint_config.setdefault("save_top_k", 1)
        checkpoint_config.setdefault("monitor", monitor)
        checkpoint_config.setdefault("mode", monitor_mode)
        callback_config["checkpoint"] = checkpoint_config
        callbacks = build_callbacks(callback_config)
        lightning_logger = pl.loggers.CSVLogger(str(experiment_dir / "lightning"), name="run")
        trainer = create_trainer(
            trainer_config,
            logger=lightning_logger,
            callbacks=callbacks,
            default_root_dir=str(experiment_dir),
        )
        experiment_logger.info(f"Starting experiment {experiment.name}")
        trainer.fit(lightning_module, datamodule=data_module)
        best_checkpoint = getattr(trainer.checkpoint_callback, "best_model_path", "")
        test_kwargs = {"datamodule": data_module}
        if best_checkpoint:
            test_kwargs["ckpt_path"] = best_checkpoint
        test_result = trainer.test(lightning_module, **test_kwargs)
        results.append({"name": experiment.name, "test": test_result})
        experiment_logger.info(f"Test result: {test_result}")
        experiment_logger.close()

    logger.info(f"Completed {len(results)} experiment(s).")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a BasicExpFrame experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    main(str(config_file))
