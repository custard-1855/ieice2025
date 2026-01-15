## 8. ExperimentDirManagerの統合

### 8.1 実行モード検出

```python
from local.experiment_dir import ExecutionMode, ExperimentConfig, ExperimentDirManager

# _init_wandb_project内
self.execution_mode = ExperimentDirManager.detect_execution_mode(
    self.hparams,
    evaluation=self.evaluation,
    test_state_dict=self._test_state_dict,
    fast_dev_run=self.fast_dev_run,
)
# 返り値: ExecutionMode.TRAIN / TEST / INFERENCE / FEATURE_EXTRACTION
```

### 8.2 実験パス構築

```python
exp_config = ExperimentConfig(
    mode="train",  # or "test"
    category="baseline",
    method="crnn_beats",
    variant="cmt_sebbs",
)

base_dir = ExperimentDirManager.build_experiment_path(
    exp_config,
    mode=self.execution_mode,
)
# 結果: experiments/train/baseline/crnn_beats/cmt_sebbs/
```

### 8.3 アーティファクトディレクトリ作成

```python
artifact_dirs = ExperimentDirManager.create_artifact_dirs(experiment_dir)
# 作成されるディレクトリ:
# - checkpoints/
# - predictions/
# - visualizations/
# - logs/
# - metrics/
```

### 8.4 マニフェスト生成

```python
ExperimentDirManager.generate_manifest(
    experiment_dir,
    run_id=wandb.run.id,
    config=self.hparams,
    mode=self.execution_mode,
)
```

**マニフェスト例 (manifest.json):**
```json
{
  "run_id": "abc123def456",
  "created_at": "2024-12-22T15:00:00",
  "execution_mode": "train",
  "config": {
    "experiment": {
      "mode": "train",
      "category": "baseline",
      "method": "crnn_beats",
      "variant": "cmt_sebbs"
    },
    "training": {...},
    "net": {...}
  },
  "artifact_directories": {
    "checkpoints": "checkpoints",
    "predictions": "predictions",
    "visualizations": "visualizations"
  }
}
```

**目的:**
- 実験の標準化されたディレクトリ構造
- マルチステージワークフローのサポート (train → test → inference)
- 実験の再現性と追跡可能性
- アーティファクトの整理
