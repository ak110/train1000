# Train with 1000

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

訓練データを1000件だけ使って学習してみる試み、[Train with 1000](http://www.ok.sc.e.titech.ac.jp/~mtanaka/proj/train1000/)をやってみたコード。

tf.kerasを使って1ファイルで全て実装したので、初心者にもわかりやすいコードになってます！ (嘘)

## スコア

```txt
[INFO ] Arguments: --data=cifar10
[INFO ] Test Accuracy:      0.7970
[INFO ] Test Cross Entropy: 0.7814
```

## 動かすために必要なもの

- TensorFlow (2.0.0で動作確認)
- Horovod
- OpenMPI (複数GPU時)
- albumentations
- scikit-learn
- pydot
- scipy

## やってること

- ResNet風
- SGD+Nesterov momentum
- cosine annealing <https://arxiv.org/abs/1608.03983>
- linear learning rate warmup <https://arxiv.org/abs/1706.02677>
- Random Erasing <https://arxiv.org/abs/1708.04896>
- mixup <https://arxiv.org/abs/1710.09412>
- Label smoothing <https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/>
- Refined Data Augmentation <https://arxiv.org/abs/1909.09148>
- その他たくさんの怪しいDataAugmentation
