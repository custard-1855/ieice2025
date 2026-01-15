## 7. WandB統合の変更

### 7.1 初期化処理

```python
# __init__内
if self.hparams["wandb"]["use_wandb"]:
    self._init_wandb_project()
else:
    self._wandb_checkpoint_dir: str | None = None
```

### 7.2 _init_wandb_project メソッド

```python
def _init_wandb_project(self) -> None:
    """実行モード対応のディレクトリ管理でwandbを初期化.
    
    このメソッドはモード対応のwandb初期化を実装:
    - 実行モードを検出 (train/test/inference/feature_extraction)
    - レガシーモード (--wandb_dir) が新モードより優先
    - 階層的な実験ディレクトリ構造を作成
    - アーティファクトディレクトリとマニフェスト生成を管理
    - inference/feature_extractionモードではwandb初期化をスキップ
    """
    # 実行モードの検出
    self.execution_mode = ExperimentDirManager.detect_execution_mode(
        self.hparams,
        evaluation=self.evaluation,
        test_state_dict=getattr(self, "_test_state_dict", None),
        fast_dev_run=self.fast_dev_run,
    )
    print(f"Detected execution mode: {self.execution_mode.value}")
    
    # ExperimentConfigを使用
    if "experiment" in self.hparams:
        exp_config = ExperimentConfig(**self.hparams["experiment"])
        
        # レガシーwandb_dirパスの処理
        if "wandb_dir" in self.hparams["wandb"]:
            legacy_path = Path(self.hparams["wandb"]["wandb_dir"])
            wandb.init(
                project=PROJECT_NAME,
                name=exp_config.variant,
                dir=str(legacy_path.parent),
                config=self.hparams,
            )
            self._wandb_checkpoint_dir = str(legacy_path)
            return
        
        # 推論/特徴抽出モードではwandbをスキップ
        if not ExperimentDirManager.should_initialize_wandb(
            self.execution_mode, exp_config
        ):
            base_dir = ExperimentDirManager.build_experiment_path(
                exp_config, mode=self.execution_mode
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            inference_dir = base_dir / f"run-{timestamp}"
            inference_dir.mkdir(parents=True, exist_ok=True)
            self._wandb_checkpoint_dir = str(inference_dir)
            print(f"Created inference directory: {inference_dir}")
            return
        
        # 訓練/テストモード: wandbを初期化
        base_dir = ExperimentDirManager.build_experiment_path(
            exp_config, mode=self.execution_mode
        )
        
        wandb.init(
            project=PROJECT_NAME,
            name=f"{self.execution_mode.value}/{exp_config.category}/"
                 f"{exp_config.method}/{exp_config.variant}",
            dir=str(base_dir),
            config=self.hparams,
            tags=[
                self.execution_mode.value,
                exp_config.category,
                exp_config.method,
            ],
        )
        
        # 実験ディレクトリとアーティファクトディレクトリを作成
        experiment_dir = Path(wandb.run.dir).parent
        artifact_dirs = ExperimentDirManager.create_artifact_dirs(experiment_dir)
        
        # マニフェストを生成
        ExperimentDirManager.generate_manifest(
            experiment_dir,
            run_id=wandb.run.id,
            config=self.hparams,
            mode=self.execution_mode,
        )
        
        self._wandb_checkpoint_dir = str(experiment_dir)
```

### 7.3 カスタムログメソッド

```python
def log(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:
    """wandbへのログをミラーリングするためにLightningModule.logをオーバーライド.
    
    元のpl.LightningModule.logを呼び出した後、同じkey/valueをwandb.logに
    送信 (wandbが初期化されuse_wandbが有効な場合)。
    テンソル/ndarrayのPythonスカラー/リストへの変換を処理。
    """
    res = super(SEDTask4, self).log(name, value, *args, **kwargs)
    try:
        self._maybe_wandb_log({name: value})
    except Exception:
        pass  # ログエラーがトレーニングを妨げないようにする
    return res

def log_dict(self, dictionary: dict[str, Any], *args: Any, **kwargs: Any) -> None:
    """Lightningのlog_dictに加えてメトリクス辞書をwandbにミラーリング."""
    res = super(SEDTask4, self).log_dict(dictionary, *args, **kwargs)
    try:
        self._maybe_wandb_log(dictionary)
    except Exception:
        pass
    return res

def _maybe_wandb_log(self, log_dict: dict[str, Any]) -> None:
    """wandbが有効で初期化されている場合、辞書を安全にログ.
    
    torchテンソルとnumpy配列をPythonスカラーまたはリストに変換。
    wandbがアクティブでない、またはhparamsで無効な場合、no-op。
    """
    if not self.hparams.get("wandb", {}).get("use_wandb", False):
        return
    
    try:
        if wandb.run is None:
            return
    except Exception:
        return
    
    def _to_native(x):
        """torch/numpyをネイティブPython型に変換."""
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.item()
            return x.detach().cpu().numpy().tolist()
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return x.item()
            return x.tolist()
        return x
    
    payload = {k: _to_native(v) for k, v in log_dict.items()}
    
    # デバッグ用: 複数のstep情報を追加
    try:
        payload["_debug/global_step"] = self.global_step
        payload["_debug/scheduler_step_num"] = (
            self.scheduler["scheduler"].step_num
        )
        payload["_debug/current_epoch"] = self.current_epoch
    except Exception:
        pass
    
    # stepが利用可能な場合は設定
    step = None
    try:
        step = self.global_step
    except Exception:
        pass
    try:
        wandb.log(payload, step=step)
    except Exception:
        pass  # ログエラーでトレーニングを中断しない
```

**目的:**
- 階層的な実験構成
- モード対応のディレクトリ作成 (train/test/inference)
- メトリクスのWandBへの自動同期
- マニフェストによる実験の再現性向上

**ディレクトリ構造例:**
```
experiments/
├── train/
│   └── baseline/
│       └── crnn_beats/
│           └── default/
│               └── wandb/
│                   └── run-20241222_150000-abc123/
│                       ├── manifest.json
│                       ├── checkpoints/
│                       ├── predictions/
│                       └── visualizations/
└── test/
    └── baseline/
        └── crnn_beats/
            └── default/
                └── run-20241222_160000-def456/
```
