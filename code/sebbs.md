## 6. cSEBBs (change-point based Sound Event Bounding Boxes) 統合

### 6.1 インスタンス変数

```python
# __init__内
self.sebbs_enabled: bool = self.hparams.get("sebbs", {}).get("enabled", False)

# 検証セットでのチューニング用バッファ
self.val_tune_sebbs_student: dict = {}
self.val_tune_sebbs_teacher: dict = {}
```

### 6.2 on_test_start でのチューニング

```python
def on_test_start(self) -> None:
    """テスト開始時に検証パスを実行してcSEBBsチューニング用のスコアを収集."""
    if not self.sebbs_enabled:
        return
    
    # MAESTROメタデータのロード
    self.maestro_audio_durations, self.maestro_ground_truth = (
        self.load_maestro_audio_durations_and_gt()
    )
    
    # 検証ローダーの実行
    val_loader = self.val_dataloader()
    self.val_tune_sebbs_student = {}
    self.val_tune_sebbs_teacher = {}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting scores for cSEBBs tuning"):
            audio, labels, _, filenames, embeddings, valid_class_mask = batch
            # ... 予測を生成してval_tune_sebbs_*に保存 ...
```

### 6.3 validation_epoch_end でのチューニング

```python
# validation_epoch_end内
if self.sebbs_enabled:
    # DESEDクラス用にチューニング
    if not hasattr(self, "csebbs_predictor_desed"):
        self.csebbs_predictor_desed, _ = SEBBsTuner.tune_for_psds(
            self.val_tune_sebbs_student,
            desed_ground_truth,
            desed_audio_durations,
            event_classes=sorted(classes_labels_desed.keys()),
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_st=1.0,
            alpha_ct=0.0,
        )
    
    # MAESTROクラス用にチューニング
    if not hasattr(self, "csebbs_predictor_maestro"):
        # MAESTROスコアを準備
        maestro_val_scores = _get_segment_scores_and_overlap_add(
            self.val_tune_sebbs_student,
            maestro_audio_durations,
            event_classes_maestro,
            segment_length=1.0,
        )
        
        # AUROCを最大化するようにチューニング
        self.csebbs_predictor_maestro, _ = SEBBsTuner.tune(
            maestro_val_scores,
            maestro_ground_truth,
            maestro_audio_durations,
            tune_function=select_best_auroc,
            max_fpr=0.1,
        )
```

### 6.4 get_sebbs メソッド

```python
def get_sebbs(self, scores_all_classes, model_type="student"):
    """DESEDとMAESTROの両クラスにcSEBBs後処理を適用.
    
    cSEBBs (change-point based Sound Event Bounding Boxes)は、
    フレームレベルの事後確率スコアから以下の処理によりイベント候補を生成:
    
    1. 変化点検出: ステップフィルタを用いてスコアの変化点を検出
    2. セグメント生成: 変化点を境界として候補セグメントを生成
    3. セグメントマージ: 類似スコアを持つ隣接セグメントを統合
    4. 信頼度計算: 各セグメントの平均スコアを信頼度として出力
    
    Args:
        scores_all_classes: 全クリップのスコアDataFrame辞書
                           各clipのDataFrameは全クラスのスコアを含む
                           形式: {clip_id: pd.DataFrame(onset, offset, class1, class2, ...)}
        model_type: 'student' or 'teacher'. モデルタイプを指定
    
    Returns:
        sed_scores_postprocessed: 後処理済みスコアDataFrame辞書
                                  cSEBBsにより生成されたイベント候補のスコア
    
    処理フロー:
        1. DESED/MAESTROクラスのスコアを個別に抽出
        2. 各データセットに最適化されたcSEBBs predictorで処理
        3. 生成されたイベント候補を統合
        4. sed_scores_eval形式のDataFrameに変換
    """
    # モデルタイプに応じたpredictorを選択
    if model_type == "teacher":
        csebbs_predictor_desed = self.csebbs_predictor_desed_teacher
        csebbs_predictor_maestro = self.csebbs_predictor_maestro_teacher
    else:
        csebbs_predictor_desed = self.csebbs_predictor_desed
        csebbs_predictor_maestro = self.csebbs_predictor_maestro
    
    # ステップ1: DESEDクラスに対するcSEBBs適用
    desed_classes = list(classes_labels_desed.keys())
    keys_desed = ["onset", "offset"] + sorted(desed_classes)
    scores_desed_classes = {
        clip_id: scores_all_classes[clip_id][keys_desed] 
        for clip_id in scores_all_classes.keys()
    }
    csebbs_desed_events = csebbs_predictor_desed.predict(scores_desed_classes)
    
    # ステップ2: MAESTROクラスに対するcSEBBs適用
    maestro_classes = sorted(classes_labels_maestro_real_eval)
    keys_maestro = ["onset", "offset"] + sorted(maestro_classes)
    scores_maestro_classes = {
        clip_id: scores_all_classes[clip_id][keys_maestro]
        for clip_id in scores_all_classes.keys()
    }
    csebbs_maestro_events = csebbs_predictor_maestro.predict(scores_maestro_classes)
    
    # ステップ3: イベント候補の統合
    all_sound_classes = sorted(list(set(desed_classes + maestro_classes)))
    all_clip_ids = set(csebbs_desed_events.keys()) | set(csebbs_maestro_events.keys())
    
    sebbs_all_events = {}
    for clip_id in all_clip_ids:
        desed_events = csebbs_desed_events.get(clip_id, [])
        maestro_events = csebbs_maestro_events.get(clip_id, [])
        sebbs_all_events[clip_id] = desed_events + maestro_events
    
    # ステップ4: sed_scores_eval形式への変換
    sed_scores_postprocessed = sed_scores_from_sebbs(
        sebbs_all_events,
        sound_classes=all_sound_classes,
        fill_value=0.0,
    )
    
    return sed_scores_postprocessed
```

### 6.5 test_step での使用

```python
# test_step内
if self.sebbs_enabled:
    # cSEBBs後処理を適用
    sed_scores_postprocessed_student = self.get_sebbs(
        self.test_buffer_sed_scores_eval_unprocessed_student,
        model_type="student",
    )
    sed_scores_postprocessed_teacher = self.get_sebbs(
        self.test_buffer_sed_scores_eval_unprocessed_teacher,
        model_type="teacher",
    )
else:
    # 従来のメディアンフィルタ
    sed_scores_postprocessed_student = median_filter_scores(...)
```

**目的:**
- イベント境界検出パラメータの自動チューニング
- データセット特性に応じた最適化 (DESED vs MAESTRO)
- セグメンテーション精度の向上
- 検証セットで一度チューニングし、テストセットで適用

**設定例:**
```yaml
sebbs:
  enabled: true
  # DESEDはPSDSで、MAESTROはAUROCでチューニング
```
