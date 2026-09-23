# Trainer API

## `ExperimentLightningModule`

```python
ExperimentLightningModule(
    model, criterion, optimizer_config,
    scheduler_config=None, metrics=None,
)
```

包装普通 `nn.Module`；实现 `forward`、`training_step`、`validation_step`、`test_step` 和 `configure_optimizers`。Batch 支持 `{"inputs": ..., "targets": ...}` 或二元 tuple/list。模型 forward 必须返回包含 `logits` 的 mapping。记录 `{train,val,test}/loss` 和配置的 metrics。

`ReduceLROnPlateau` scheduler 默认监控 `val/loss`，可在 scheduler 配置中使用 `monitor` 覆盖。

## `ExperimentDataModule`

接受一个 `Experiment`，或关键字形式传入 `train`、`valid`、`test` Dataset；另有 `batch_size=1`、`num_workers=0` 和其他 DataLoader kwargs。只转发已切分数据，不做发现或划分。

## Trainer 工厂

`create_trainer(config=None, *, logger=None, callbacks=None, default_root_dir=None) -> lightning.pytorch.Trainer`。一般 Trainer 参数直接透传；框架别名 `epochs -> max_epochs`、`grad_clip -> gradient_clip_val`、`use_amp=True -> precision="16-mixed"`（如果未显式设置 precision）。默认关闭 progress bar。

## Callbacks

`build_callbacks(config=None) -> list[Callback]` 接受：

- `checkpoint`: 传给 `ModelCheckpoint` 的参数字典；默认启用默认 ModelCheckpoint，设为 `false` 可关闭。
- `early_stopping`: 传给 `EarlyStopping` 的参数字典；缺省/false 时不添加。
- `learning_rate_monitor`: 传给 `LearningRateMonitor` 的参数字典；默认启用，设为 `false` 可关闭。

入口会为每个 Experiment 补充独立 checkpoint 路径，并在训练完成后用 best checkpoint 测试（如果 callback 产生了 best checkpoint）。
