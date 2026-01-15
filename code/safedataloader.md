## 3. 新規クラス

### 3.1 SafeDataLoaderインフラストラクチャ

#### _NoneSafeIterator クラス
```python
class _NoneSafeIterator:
    """DataLoaderから返されるNoneバッチを内部でスキップする."""

    def __init__(self, dataloader_iter):
        self.dataloader_iter = dataloader_iter

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            batch = next(self.dataloader_iter)
            if batch is not None:
                return batch
            print("Skipping a batch that was None internally.")
```

#### SafeCollate クラス
```python
class SafeCollate:
    """データセットから返される None 値をフィルタリングする collate_fn.
    
    フィルタリング後にバッチが空になった場合は None を返す.
    """

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None  # このNoneは_NoneSafeIteratorで捕捉される
        return default_collate(batch)
```

#### SafeDataLoader クラス
```python
class SafeDataLoader(DataLoader):
    """Noneを返す可能性があるバッチを自動的にスキップするDataLoader.
    
    PyTorch Lightningなどのフレームワークで安全に使用できる。
    """

    def __init__(self, *args, **kwargs):
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = SafeCollate()
        super().__init__(*args, **kwargs)

    def __iter__(self):
        dataloader_iter = super().__iter__()
        return _NoneSafeIterator(dataloader_iter)
```

**目的:**
- 破損したオーディオファイルや欠損した埋め込みの処理
- トレーニング中のクラッシュ防止
- データセット品質が不完全な場合の堅牢性向上

**使用例:**
```python
def train_dataloader(self) -> SafeDataLoader:
    return SafeDataLoader(
        self.train_data,
        batch_sampler=self.train_sampler,
        num_workers=self.num_workers,
    )
```
