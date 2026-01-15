## 使用方法

### CMTの有効化

```yaml
# confs/pretrained.yaml
cmt:
  enabled: true
  warmup_epochs: 50
  phi_clip: 0.5
  phi_frame: 0.5
  phi_neg: 0.3
  phi_pos: 0.7
  use_neg_sample: false
```

### cSEBBsの有効化

```yaml
# confs/pretrained.yaml
sebbs:
  enabled: true
```

### WandB統合

```yaml
# confs/pretrained.yaml
wandb:
  use_wandb: true

experiment:
  mode: train  # or test
  category: baseline
  method: crnn_beats
  variant: cmt_sebbs
```

### SafeDataLoaderの使用

```python
# データセットがNoneを返す可能性がある場合、自動的に処理される
train_loader = self.train_dataloader()  # SafeDataLoaderを返す
```
