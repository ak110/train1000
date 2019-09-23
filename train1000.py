#!/usr/bin/env python3
"""Train with 1000"""
import argparse
import functools
import logging
import pathlib
import random

import albumentations as A
import cv2
import horovod.tensorflow.keras as hvd
import joblib
import numpy as np
import sklearn.metrics
import tensorflow as tf


def _main():
    try:
        import better_exceptions

        better_exceptions.hook()
    except BaseException:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="cifar10",
        choices=("mnist", "fashion_mnist", "cifar10", "cifar100"),
    )
    parser.add_argument("--check", action="store_true", help="3epochだけお試し実行(動作確認用)")
    parser.add_argument(
        "--results-dir", default=pathlib.Path("results"), type=pathlib.Path
    )
    args = parser.parse_args()

    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    handlers = [logging.StreamHandler()]
    if hvd.rank() == 0:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(args.results_dir / f"{args.data}.log", encoding="utf-8")
        )
    logging.basicConfig(
        format="[%(levelname)-5s] %(message)s", level="DEBUG", handlers=handlers
    )
    logger = logging.getLogger(__name__)

    (X_train, y_train), (X_test, y_test), num_classes = _load_data(args.data)

    epochs = 5 if args.check else 1800
    batch_size = 64
    base_lr = 1e-3 * batch_size * hvd.size()

    model = _create_network(X_train.shape[1:], num_classes)
    optimizer = tf.keras.optimizers.SGD(lr=base_lr, momentum=0.9, nesterov=True)
    optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.fp16)

    def loss(y_true, y_pred):
        del y_pred
        logits = model.get_layer("logits").output

        # categorical crossentropy
        log_p = logits - tf.math.reduce_logsumexp(logits, axis=-1, keepdims=True)
        loss = -tf.keras.backend.sum(y_true * log_p, axis=-1)

        # Label smoothing <https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/>
        label_smoothing = 0.2
        kl = -tf.keras.backend.mean(log_p, axis=-1)
        loss = (1 - label_smoothing) * loss + label_smoothing * kl

        return loss

    model.compile(
        optimizer, loss, metrics=["acc", "mae"], experimental_run_tf_function=False
    )
    model.summary(print_fn=logger.info if hvd.rank() == 0 else lambda x: x)
    try:
        tf.keras.utils.plot_model(
            model, args.results_dir / f"{args.data}.svg", show_shapes=True
        )
    except ValueError:
        pass

    model.fit(
        _generate(
            X_train,
            y_train,
            batch_size,
            num_classes,
            shuffle=True,
            data_augmentation="train",
        ),
        steps_per_epoch=-(-len(X_train) // (batch_size * hvd.size())),
        validation_data=_generate(
            X_test, y_test, batch_size, num_classes, shuffle=True
        ),
        validation_steps=-(-len(X_test) * 3 // (batch_size * hvd.size())),
        validation_freq=100,
        epochs=epochs,
        callbacks=[
            _cosine_annealing_callback(base_lr, epochs),
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
        ],
        verbose=1 if hvd.rank() == 0 else 0,
    )
    # Refined Data Augmentation <https://arxiv.org/abs/1909.09148>
    # + freeze BNs <https://arxiv.org/abs/1709.01507>
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    optimizer = tf.keras.optimizers.SGD(lr=base_lr / 100, momentum=0.9, nesterov=True)
    optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.fp16)
    model.compile(optimizer, loss, metrics=["acc"], experimental_run_tf_function=False)
    model.fit(
        _generate(
            X_train,
            y_train,
            batch_size,
            num_classes,
            shuffle=True,
            data_augmentation="refine",
        ),
        steps_per_epoch=-(-len(X_train) // (batch_size * hvd.size())),
        validation_data=_generate(
            X_test, y_test, batch_size, num_classes, shuffle=True
        ),
        validation_steps=-(-len(X_test) * 3 // (batch_size * hvd.size())),
        validation_freq=10,
        epochs=50,
        callbacks=[
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
        ],
        verbose=1 if hvd.rank() == 0 else 0,
    )

    if hvd.rank() == 0:
        # 検証
        pred_test = model.predict(
            _generate(
                X_test,
                np.zeros((len(X_test),), dtype=np.int32),
                batch_size,
                num_classes,
            ),
            steps=int(np.ceil(len(X_test) / batch_size)),
            verbose=1 if hvd.rank() == 0 else 0,
        )
        acc = sklearn.metrics.accuracy_score(y_test, pred_test.argmax(axis=-1))
        ce = sklearn.metrics.log_loss(y_test, pred_test)
        logger.info(f"Arguments: --data={args.data}")
        logger.info(f"Test Accuracy:      {acc:.4f}")
        logger.info(f"Test Cross Entropy: {ce:.4f}")
        # 後で何かしたくなった時のために一応保存
        model.save(args.results_dir / f"{args.data}.h5", include_optimizer=False)


def _load_data(data):
    """データの読み込み。"""
    (X_train, y_train), (X_test, y_test) = {
        "mnist": tf.keras.datasets.mnist.load_data,
        "fashion_mnist": tf.keras.datasets.fashion_mnist.load_data,
        "cifar10": tf.keras.datasets.cifar10.load_data,
        "cifar100": tf.keras.datasets.cifar100.load_data,
    }[data]()
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    num_classes = len(np.unique(y_train))
    X_train, y_train = _extract1000(X_train, y_train, num_classes=num_classes)
    return (X_train, y_train), (X_test, y_test), num_classes


def _extract1000(X, y, num_classes):
    """https://github.com/mastnk/train1000 を参考にクラスごとに均等に先頭から取得する処理。"""
    num_data = 1000
    num_per_class = num_data // num_classes

    index_list = []
    for c in range(num_classes):
        index_list.extend(np.where(y == c)[0][:num_per_class])
    assert len(index_list) == num_data

    return X[index_list], y[index_list]


def _create_network(input_shape, num_classes):
    """ネットワークを作成して返す。"""
    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    bn = functools.partial(
        tf.keras.layers.BatchNormalization,
        gamma_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    act = functools.partial(tf.keras.layers.Activation, "relu")

    def down(filters):
        def layers(x):
            x = conv2d(filters, kernel_size=4, strides=2)(x)
            x = bn()(x)
            return x

        return layers

    def blocks(filters, count):
        def layers(x):
            for _ in range(count):
                sc = x
                x = conv2d(filters)(x)
                x = bn()(x)
                x = act()(x)
                x = conv2d(filters)(x)
                # resblockのadd前だけgammaの初期値を0にする。 <https://arxiv.org/abs/1812.01187>
                x = bn(gamma_initializer="zeros")(x)
                x = tf.keras.layers.add([sc, x])
            x = bn()(x)
            x = act()(x)
            return x

        return layers

    inputs = x = tf.keras.layers.Input(input_shape)
    x = conv2d(128)(x)
    x = bn()(x)
    x = blocks(128, 8)(x)
    x = down(256)(x)
    x = blocks(256, 8)(x)
    x = down(512)(x)
    x = blocks(512, 8)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        num_classes, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="logits"
    )(x)
    x = tf.keras.layers.Activation(activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


def _cosine_annealing_callback(base_lr, epochs, warmup_epochs=5):
    """Cosine annealing <https://arxiv.org/abs/1608.03983>"""

    def _cosine_annealing(ep, lr):
        del lr
        if ep + 1 < warmup_epochs:
            return base_lr * (ep + 1) / warmup_epochs
        min_lr = base_lr * 0.01
        return min_lr + 0.5 * (base_lr - min_lr) * (
            1 + np.cos(np.pi * (ep + 1) / epochs)
        )

    return tf.keras.callbacks.LearningRateScheduler(_cosine_annealing)


def _generate(X, y, batch_size, num_classes, shuffle=False, data_augmentation="test"):
    """generator。"""
    if data_augmentation == "train":
        aug1 = A.Compose(
            [
                A.Rotate(
                    15,
                    interpolation=cv2.INTER_LANCZOS4,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=0.25,
                ),
                A.PadIfNeeded(40, 40, border_mode=cv2.BORDER_REPLICATE, p=1),
                A.OneOf(
                    [
                        A.RandomScale((0.875, 1), interpolation=cv2.INTER_AREA, p=0.25),
                        A.RandomScale(
                            (1, 1.125), interpolation=cv2.INTER_LANCZOS4, p=0.25
                        ),
                    ],
                    p=0.25,
                ),
                A.RandomCrop(32, 32, p=1),
                A.HorizontalFlip(p=0.5),
                RandomCompose(
                    [
                        A.Equalize(mode="pil", by_channels=True, p=0.0625),
                        A.Equalize(mode="pil", by_channels=False, p=0.0625),
                        A.CLAHE(p=0.0625),
                        A.RandomBrightnessContrast(brightness_by_max=True, p=0.5),
                        A.HueSaturationValue(val_shift_limit=0, p=0.5),
                        A.Posterize(num_bits=(4, 7), p=0.0625),
                        A.Solarize(threshold=(50, 255 - 50), p=0.0625),
                        A.Blur(blur_limit=1, p=0.125),
                        A.IAASharpen(alpha=(0, 0.5), p=0.125),
                        A.IAAEmboss(alpha=(0, 0.5), p=0.125),
                        A.GaussNoise(var_limit=(0, 10.0 ** 2), p=0.125),
                        A.ISONoise(color_shift=(0, 0.05), intensity=(0, 0.5), p=0.125),
                        A.ImageCompression(
                            quality_lower=50, quality_upper=100, p=0.125
                        ),
                        RandomErasing(alpha=0.125, p=0.25),
                    ],
                    p=1,
                ),
            ]
        )
        aug2 = RandomErasing(p=0.5)
    elif data_augmentation == "refine":
        aug1 = A.Compose(
            [
                A.PadIfNeeded(40, 40, border_mode=cv2.BORDER_REPLICATE, p=1),
                A.RandomCrop(32, 32, p=1),
                A.HorizontalFlip(p=0.5),
            ]
        )
        aug2 = A.Compose([])
    else:
        aug1 = A.Compose([])
        aug2 = A.Compose([])

    with joblib.Parallel(backend="threading", n_jobs=batch_size) as parallel:
        if shuffle:
            batch_indices = []
            for index in _generate_shuffled_indices(len(X)):
                batch_indices.append(index)
                if len(batch_indices) == batch_size:
                    yield _generate_batch(
                        X,
                        y,
                        aug1,
                        aug2,
                        num_classes,
                        data_augmentation,
                        batch_indices,
                        parallel,
                    )
                    batch_indices = []
        else:
            while True:
                for i in range(0, len(X), batch_size):
                    batch_indices = range(i, min(i + batch_size, len(X)))
                    yield _generate_batch(
                        X,
                        y,
                        aug1,
                        aug2,
                        num_classes,
                        data_augmentation,
                        batch_indices,
                        parallel,
                    )


def _generate_shuffled_indices(data_count):
    """シャッフルしたindexを無限に返すgenerator。"""
    all_indices = np.arange(data_count)
    while True:
        random.shuffle(all_indices)
        yield from all_indices


def _generate_batch(
    X, y, aug1, aug2, num_classes, data_augmentation, batch_indices, parallel
):
    """1バッチずつの処理。"""
    jobs = [
        _generate_instance(X, y, aug1, aug2, num_classes, data_augmentation, i)
        for i in batch_indices
    ]
    results = parallel(jobs)
    X_batch, y_batch = zip(*results)
    return np.array(X_batch), np.array(y_batch)


@joblib.delayed
def _generate_instance(X, y, aug1, aug2, num_classes, data_augmentation, index):
    """1サンプルずつの処理。"""
    X_i, y_i = X[index], y[index]
    X_i = aug1(image=X_i)["image"]
    c_i = tf.keras.utils.to_categorical(y_i, num_classes)

    if data_augmentation == "train":
        # mixup
        t = random.choice(range(len(y)))
        X_t, y_t = X[t], y[t]
        X_t = aug1(image=X_t)["image"]
        c_t = tf.keras.utils.to_categorical(y_t, num_classes)
        r = random.uniform(0.5, 1.0)
        X_i = X_i.astype(np.float32) * r + X_t.astype(np.float32) * (1 - r)
        c_i = c_i.astype(np.float32) * r + c_t.astype(np.float32) * (1 - r)

    X_i = aug2(image=X_i)["image"]
    X_i = X_i.astype(np.float32) / 127.5 - 1
    return X_i, c_i


class RandomCompose(A.Compose):
    """シャッフル付きCompose。"""

    def __call__(self, force_apply=False, **data):
        """変換の適用。"""
        backup = self.transforms.transforms.copy()
        try:
            random.shuffle(self.transforms.transforms)
            return super().__call__(force_apply=force_apply, **data)
        finally:
            self.transforms.transforms = backup


class RandomErasing(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    def __init__(
        self, scale=(0.02, 0.4), rate=(1 / 3, 3), alpha=None, always_apply=False, p=0.5
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale
        self.rate = rate
        self.alpha = alpha

    def apply(self, img, **data):
        del data
        for _ in range(30):
            s = img.shape[0] * img.shape[1] * random.uniform(*self.scale)
            r = np.exp(random.uniform(np.log(self.rate[0]), np.log(self.rate[1])))
            ew = int(np.sqrt(s / r))
            eh = int(np.sqrt(s * r))
            if ew <= 0 or eh <= 0 or ew >= img.shape[1] or eh >= img.shape[0]:
                continue
            ex = random.randint(0, img.shape[1] - ew - 1)
            ey = random.randint(0, img.shape[0] - eh - 1)

            img = np.copy(img)
            rc = np.array([random.randint(0, 255) for _ in range(img.shape[-1])])
            if self.alpha:
                img[ey : ey + eh, ex : ex + ew, :] = (
                    img[ey : ey + eh, ex : ex + ew, :] * (1 - self.alpha)
                    + rc * self.alpha
                ).astype(img.dtype)
            else:
                img[ey : ey + eh, ex : ex + ew, :] = rc[np.newaxis, np.newaxis, :]
            break
        return img


if __name__ == "__main__":
    _main()
