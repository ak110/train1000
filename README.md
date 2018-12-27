# train with 1000

訓練データを1000件だけ使って学習してみる試み、[train with 1000](http://www.ok.sc.e.titech.ac.jp/~mtanaka/proj/train1000/)をやってみたコード。

Kerasを使って1ファイルで全て実装したので、初心者にもわかりやすいコードになってます！ (嘘)

## 現在の成績

とりあえずCIFAR-10だけお試し中。

起動引数に`--model=heavy`を付けると頑張るバージョン。デフォルト(`--model=light`)は軽量バージョン。

```txt
[INFO ] Arguments: --data=cifar10 --model=light
[INFO ] Test Accuracy:      0.7668
[INFO ] Test Cross Entropy: 0.7959
```

```txt
[INFO ] Arguments: --data=cifar10 --model=heavy
[INFO ] Test Accuracy:      0.7751
[INFO ] Test Cross Entropy: 0.7699
```

(1回しかやってないのであまり確かな値ではないです。)

## 動かすために必要なもの

- TensorFlow (1.10.0で動作確認)
- Horovod
- OpenMPI
- albumentations
- opencv-python
- Pillow (or Pillow-SIMD)
- scikit-learn

## やってること(`--model=heavy`)

- ResidualでDenseな謎ネットワーク
- Parallel Grid Pooling <https://arxiv.org/abs/1803.11370>
- MixFeat <https://openreview.net/forum?id=HygT9oRqFX>
- Drop-Activation <https://arxiv.org/abs/1811.05850>
- linear learning rate (Horovod)
- learning rate warmup (Horovod)
- cosine annealing <https://arxiv.org/abs/1608.03983>
- SGD+Nesterov momentum
- The parameters of all BN layers were frozen for the last few training epochs <https://arxiv.org/abs/1709.01507>
- AutoAugment <https://arxiv.org/abs/1805.09501>

## TODO

- AutoAugmentの実装がひどすぎるので何とかしたい
- `--model=heavy`がただでさえ重いのにGPU利用率が低めなので何とかしたい
