# Train with 1000

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

訓練データを1,000件だけ使って学習してみる試み、[Train with 1000](http://www.ok.sc.e.titech.ac.jp/~mtanaka/proj/train1000/)をやってみたコード。

tf.kerasを使って1ファイルで全て実装したので、初心者にもわかりやすいコードになってます！ (嘘)

## スコア (CIFAR-10)

```txt
[INFO ] Val Accuracy:       0.8119 (5 runs)
[INFO ] Val Cross Entropy:  0.7364 (5 runs)
[INFO ] Test Accuracy:      0.8112 (5 runs)
[INFO ] Test Cross Entropy: 0.7406 (5 runs)
```

Valは訓練データの末尾10,000件。(独自)

## 動かすために必要なもの

- TensorFlow (2.1.0で動作確認)
- Horovod
- OpenMPI (複数GPU時)
- albumentations
- scikit-learn

## やってること

- ResNet風
- SGD+Nesterov momentum
- cosine annealing <https://arxiv.org/abs/1608.03983>
- Random Erasing <https://arxiv.org/abs/1708.04896>
- mixup <https://arxiv.org/abs/1710.09412>
- Label smoothing <https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/>
- Refined Data Augmentation <https://arxiv.org/abs/1909.09148>
- その他たくさんの怪しいDataAugmentation
