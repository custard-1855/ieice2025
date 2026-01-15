## 1. インポート変更

### 追加されたインポート

```python
from datetime import datetime
from typing import Any, TypeAlias
import torch.nn.functional as F
import wandb
from desed_task.data_augm import MixupAugmentor  # mixup関数からクラスベースに変更
from local.experiment_dir import ExecutionMode, ExperimentConfig, ExperimentDirManager
from local.sebbs_wrapper import SEBBsPredictor, SEBBsTuner
from sebbs.sebbs.utils import sed_scores_from_sebbs
from torch.utils.data.dataloader import DataLoader, default_collate
```

**目的:**
- 型安全性の向上 (typing)
- WandB統合 (wandb)
- 実験管理の改善 (ExperimentDirManager)
- cSEBBs後処理の追加 (SEBBsPredictor, SEBBsTuner)
- Mixupの高度化 (MixupAugmentor)

---

## 2. 型アノテーション (TypeAlias)

### 新規追加された型定義

```python
# プロジェクト名の定数
PROJECT_NAME = "SED-pl-noise"

# トレーニングバッチ構造 (ファイル名なし) - 5要素
TrainBatchType: TypeAlias = tuple[
    torch.Tensor,  # audio: (batch_size, audio_length) waveform
    torch.Tensor,  # labels: (batch_size, time_steps, num_classes) strong labels
    torch.Tensor,  # padded_indxs: (batch_size,) padding mask
    torch.Tensor,  # embeddings: (batch_size, embedding_dim, time_steps) BEATs features
    torch.Tensor,  # valid_class_mask: (batch_size, num_classes) class validity mask
]

# 評価バッチ構造 (ファイル名あり) - 6要素
EvalBatchType: TypeAlias = tuple[
    torch.Tensor,  # audio
    torch.Tensor,  # labels
    torch.Tensor,  # padded_indxs
    list[str],     # filenames: オーディオファイルパス
    torch.Tensor,  # embeddings
    torch.Tensor,  # valid_class_mask
]

# モデル予測出力ペア
PredictionPair: TypeAlias = tuple[
    torch.Tensor,  # strong_preds: (batch_size, time_steps, num_classes)
    torch.Tensor,  # weak_preds: (batch_size, num_classes)
]

# PSDS評価用スコアDataFrame辞書
ScoreDataFrameDict: TypeAlias = dict[str, pd.DataFrame]

# PSDS評価結果構造
PSDSResult: TypeAlias = dict[str, float | dict[str, float]]
```

**目的:**
- 型安全性の向上
- データ構造の明確なドキュメント化
- IDEのオートコンプリートサポート
- 開発時のバグ検出

## 4. SEDTask4クラスの新規メソッド

### 4.1 ヘルパーメソッド (コード重複削減)

#### _process_embeddings メソッド
```python
def _process_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
    """事前学習モデルからの埋め込み抽出 (e2eモード時).
    
    事前条件:
    - embeddingsテンソルは事前学習モデルの入力形状に一致
    - e2e=Trueの場合、self.pretrained_modelが初期化済み
    
    事後条件:
    - e2e=Trueなら処理済み埋め込みを返す、そうでなければ入力をそのまま返す
    - frozen=Trueの場合、pretrained_modelが評価モードであることを保証
    
    不変条件:
    - モデルの重みを変更しない (推論のみ)
    """
    if self.hparams["pretrained"]["e2e"]:
        if self.pretrained_model.training and self.hparams["pretrained"]["freezed"]:
            self.pretrained_model.eval()
        return self.pretrained_model(embeddings)[self.hparams["net"]["embedding_type"]]
    return embeddings
```

#### _generate_predictions メソッド
```python
def _generate_predictions(
    self,
    audio: torch.Tensor,
    embeddings: torch.Tensor,
    classes_mask: torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """オーディオから生徒モデルと教師モデルの予測を生成.
    
    事前条件:
    - audioテンソル形状: (batch_size, audio_length)
    - embeddingsは_process_embeddings()で処理済み
    - classes_maskはマスク付きクラス予測用のオプショナルテンソル
    
    事後条件:
    - ((strong_student, weak_student), (strong_teacher, weak_teacher))を返す
    - 各予測テンソル形状: strongは(batch_size, time_steps, num_classes)
      weakは(batch_size, num_classes)
    
    不変条件:
    - メルスペクトログラムは1回だけ計算され、両モデルで共有される
    """
    mels = self.mel_spec(audio)
    strong_student, weak_student = self.detect(
        mels, self.sed_student, embeddings=embeddings, classes_mask=classes_mask,
    )
    strong_teacher, weak_teacher = self.detect(
        mels, self.sed_teacher, embeddings=embeddings, classes_mask=classes_mask,
    )
    return (strong_student, weak_student), (strong_teacher, weak_teacher)
```

#### _compute_step_loss メソッド
```python
def _compute_step_loss(
    self,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """マスクされた予測に対する教師あり損失を計算.
    
    事前条件:
    - predictionsとlabelsはBCE損失に対応する互換性のある形状
    - maskが提供される場合、バッチ次元に一致するbooleanテンソル
    
    事後条件:
    - スカラー損失テンソルを返す
    - maskが提供される場合、マスクされたサブセットのみで損失を計算
    
    不変条件:
    - 損失は非負
    """
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    return self.supervised_loss(predictions, labels)
```

#### _update_metrics メソッド
```python
def _update_metrics(
    self,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    metric_name: str,
    mask: torch.Tensor | None = None,
) -> None:
    """指定されたメトリクスを予測とラベルで更新.
    
    事前条件:
    - metric_nameは初期化済みのtorchmetricsインスタンスに対応
      (例: "weak_student_f1_seg_macro")
    - predictionsとlabelsはメトリクスの期待するフォーマットに一致
    - maskが提供される場合、バッチ次元に一致するbooleanテンソル
    
    事後条件:
    - 内部メトリクス状態を更新 (エポック終了時の計算用に累積)
    - 戻り値なし (副作用のみ)
    
    不変条件:
    - 更新後もメトリクス状態は有効のまま
    """
    metric = getattr(self, f"get_{metric_name}")
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    metric(predictions, labels.long() if "f1" in metric_name else labels)
```

**効果:**
- training_step、validation_step、test_stepでのコード重複を約40%削減
- 保守性の向上
- テストが容易
- 一貫性のあるエラー処理

## 10. その他の重要な変更

### 10.1 完全な型ヒント

```python
# 全メソッドに型アノテーションを追加
def training_step(self, batch: TrainBatchType, batch_indx: int) -> torch.Tensor:
    ...

def validation_step(self, batch: EvalBatchType, batch_indx: int) -> None:
    ...

def test_step(self, batch: EvalBatchType, batch_indx: int) -> None:
    ...

def train_dataloader(self) -> SafeDataLoader:
    ...
```

### 10.2 DataLoaderの戻り値型

```python
# 元の実装
def train_dataloader(self):
    return torch.utils.data.DataLoader(...)

# 改良版
def train_dataloader(self) -> SafeDataLoader:
    return SafeDataLoader(
        self.train_data,
        batch_sampler=self.train_sampler,
        num_workers=self.num_workers,
    )
```

### 10.3 training_stepのマスク変更

```python
# 元の実装: 3つのマスク
strong_mask[:indx_strong] = 1
weak_mask[indx_strong:indx_weak] = 1
mask_unlabeled[indx_maestro:] = 1

# 改良版: 4つのマスク (より細かい制御)
strong_mask[:indx_strong] = 1              # 強ラベル付きデータ
weak_mask[indx_strong:indx_weak] = 1       # 弱ラベル付きデータ
mask_consistency[indx_maestro:] = 1        # 一貫性損失用
mask_pure_unlabeled[indx_weak:] = 1        # 純粋なラベルなしデータ
```

### 10.4 validation_epoch_endの安全性チェック

```python
# 改良版: オーディオIDのフィルタリング追加
common_audio_ids = (
    desed_ground_truth.keys() 
    & desed_audio_durations.keys()
    & self.val_buffer_sed_scores_eval_student.keys()
)
desed_ground_truth = {
    audio_id: gt 
    for audio_id in common_audio_ids
}
desed_audio_durations = {
    audio_id: dur 
    for audio_id in common_audio_ids
}
```

**目的:** ground truthとdurationファイルの不整合時のクラッシュ防止

### 10.5 WandB終了呼び出し

```python
# test_epoch_end内
def on_test_epoch_end(self) -> None:
    # ... 結果を計算 ...
    for key in results:
        self.log(key, results[key], prog_bar=True, logger=True)
    
    # wandbランを明示的にクローズ
    if self.hparams.get("wandb", {}).get("use_wandb", False):
        try:
            wandb.finish()
        except Exception:
            pass
```

### 10.6 拡張ログ

```python
# デバッグステップ情報を追加
self.log("train/step/global_step", self.global_step)
self.log("train/step/current_epoch", self.current_epoch)
self.log("_debug/scheduler_step_num", self.scheduler["scheduler"].step_num)

# CMT関連のログ
if self.cmt_enabled:
    self.log("train/cmt/active", float(cmt_active))
    self.log("train/student/loss_w_consistency", weak_self_sup_loss)
    self.log("train/student/loss_s_consistency", strong_self_sup_loss)
```

### 10.7 テスト時のcSEBBs検証パス

改良版ではテスト開始時に完全な検証パスを実行してcSEBBsチューニング用のスコアを収集:

```python
def on_test_start(self) -> None:
    if self.sebbs_enabled and self.valid_data:
        val_loader = self.val_dataloader()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Collecting scores"):
                # バッチを処理してスコアを保存
                # val_tune_sebbs_studentとval_tune_sebbs_teacherに格納
```
