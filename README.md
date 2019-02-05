# Train with 1000

訓練データを1000件だけ使って学習してみる試み、[Train with 1000](http://www.ok.sc.e.titech.ac.jp/~mtanaka/proj/train1000/)をやってみたコード。

Kerasを使って1ファイルで全て実装したので、初心者にもわかりやすいコードになってます！ (嘘)

## 現在の成績

とりあえずCIFAR-10だけお試し中。(各1回しかやってないのであまり確かな値ではないです。)

### `train.py`

```txt
[INFO ] Arguments: --data=cifar10
[INFO ] Test Accuracy:      0.7715
[INFO ] Test Cross Entropy: 0.7840
```

PGP <https://arxiv.org/abs/1803.11370> を使ったりEpoch数を増やしたりするともうちょっと伸びるけど重いので廃止してしまった。

### `train-light.py` (軽量版)

```txt
[INFO ] Arguments: --data=cifar10
[INFO ] Test Accuracy:      0.7410
[INFO ] Test Cross Entropy: 0.8714
```

実験など用。

## 動かすために必要なもの

- TensorFlow (1.12.0で動作確認)
- Horovod
- OpenMPI
- albumentations
- opencv-python
- Pillow (or Pillow-SIMD)
- scikit-learn

## やってること

- ResNet風
- MixFeat <https://openreview.net/forum?id=HygT9oRqFX>
- Drop-Activation <https://arxiv.org/abs/1811.05850>
- linear learning rate (Horovod)
- learning rate warmup (Horovod)
- cosine annealing <https://arxiv.org/abs/1608.03983>
- SGD+Nesterov momentum
- The parameters of all BN layers were frozen for the last few training epochs <https://arxiv.org/abs/1709.01507>
- AutoAugment <https://arxiv.org/abs/1805.09501>
- Cutout <https://arxiv.org/abs/1708.04552>
- Between-class Learning <https://arxiv.org/abs/1711.10284>
