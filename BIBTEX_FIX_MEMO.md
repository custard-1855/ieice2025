# BibTeX修正箇所メモ

## 問題の原因

`sieicej.bst`（電子情報通信学会論文誌用BibTeXスタイル）には以下の特性があります：

1. `@inproceedings`の`organization`フィールドが`booktitle`の直後にスペースなしで連結される
2. `pages`フィールドがないエントリは`\textbullet`で出力される
3. arXiv論文を`@article`にすると不自然な出力になる

---

## 修正内容

### 1. `@inproceedings`の`organization`フィールド削除

`booktitle`を短縮形に変更し、`organization`を削除しました。

| エントリ | 修正前 | 修正後 |
|---------|--------|--------|
| `serizel2020sound` | `booktitle={ICASSP 2020-...}`, `organization={IEEE}` | `booktitle={Proc. ICASSP}` |
| `xiao2023multi` | `booktitle={ICASSP 2023-...}`, `organization={IEEE}` | `booktitle={Proc. ICASSP}` |
| `bilen2020framework` | `booktitle={ICASSP 2020-...}`, `organization={IEEE}` | `booktitle={Proc. ICASSP}` |

### 2. `pages`フィールド追加

| エントリ | 修正内容 |
|---------|---------|
| `turpault2019sound` | `pages={1--5}` 追加 |

### 3. `@article` → `@inproceedings` 変更

会議論文は`@inproceedings`が適切です。

| エントリ | 修正前 | 修正後 |
|---------|--------|--------|
| `tarvainen2017mean` | `@article`, `journal={Advances in neural...}` | `@inproceedings`, `booktitle={Advances in Neural Information Processing Systems}`, `pages={1195--1204}` |
| `zhou2021domain` | `@article`, `journal={arXiv preprint...}` | `@inproceedings`, `booktitle={Proc. ICLR}`, `pages={1--15}` |
| `zhang2017mixup` | `@article`, `journal={arXiv preprint...}`, `year={2017}` | `@inproceedings`, `booktitle={Proc. ICLR}`, `pages={1--13}`, `year={2018}` |
| `kingma2014adam` | `@article`, `author={Kingma, Diederik P}`, `year={2014}` | `@inproceedings`, `author={Kingma, Diederik P and Ba, Jimmy}`, `booktitle={Proc. ICLR}`, `pages={1--15}`, `year={2015}` |

### 4. arXiv論文を `@misc` に変更

| エントリ | 修正前 | 修正後 |
|---------|--------|--------|
| `schmid2024improving` | `@article`, `journal={arXiv preprint arXiv:2408.00791}` | `@misc`, `howpublished={arXiv:2408.00791}` |
| `kim2022domain` | `@article`, `journal={arXiv preprint arXiv:2206.12513}` | `@misc`, `howpublished={arXiv:2206.12513}` |
| `xiao2024fmsg` | `@article`, `journal={arXiv preprint arXiv:2407.00291}` | `@misc`, `howpublished={arXiv:2407.00291}` |
| `ebbers2024sound` | `@article`, `journal={arXiv preprint arXiv:2406.04212}` | `@misc`, `howpublished={arXiv:2406.04212}` |
| `chen2022beats` | `@article`, `journal={arXiv preprint arXiv:2212.09058}` | `@misc`, `howpublished={arXiv:2212.09058}` |

---

## ビルド手順

```bash
uplatex main.tex
upbibtex main
uplatex main.tex
uplatex main.tex
```

※ BibTeX実行後、参照解決のため2回のLaTeX実行が必要です。
