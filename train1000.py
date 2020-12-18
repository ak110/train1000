#!/usr/bin/env python3
"""Train with 1000"""
import argparse
import functools
import logging
import pathlib
import random
import typing

import albumentations as A
import cv2
import horovod.tensorflow.keras as hvd
import numpy as np
import scipy
import sklearn.metrics
import tensorflow as tf

logger = logging.getLogger(__name__)


def _main():
    try:
        import better_exceptions

        better_exceptions.hook()
    except Exception:
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
            logging.FileHandler(
                args.results_dir / f"{args.data}.log", mode="w", encoding="utf-8"
            )
        )
    logging.basicConfig(
        format="[%(levelname)-5s] %(message)s",
        level=logging.DEBUG if hvd.rank() == 0 else logging.WARNING,
        handlers=handlers,
    )

    runs = 1 if args.check else 5
    val_acc, val_ce, test_acc, test_ce = zip(*[_run(args) for _ in range(runs)])
    if runs > 1:
        val_acc = np.mean(val_acc, axis=0)
        val_ce = np.mean(val_ce, axis=0)
        test_acc = np.mean(test_acc, axis=0)
        test_ce = np.mean(test_ce, axis=0)
        logger.info(f"Val Accuracy:       {val_acc:.4f} ({runs} runs)")
        logger.info(f"Val Cross Entropy:  {val_ce:.4f} ({runs} runs)")
        logger.info(f"Test Accuracy:      {test_acc:.4f} ({runs} runs)")
        logger.info(f"Test Cross Entropy: {test_ce:.4f} ({runs} runs)")


def _run(args):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes = load_data(
        args.data
    )

    epochs = 2 if args.check else 1800
    refine_epoch = 2 if args.check else 100
    batch_size = 64
    global_batch_size = batch_size * hvd.size()
    base_lr = 1e-3 * global_batch_size

    input_shape = X_train.shape[1:]
    model = create_network(input_shape, num_classes)
    learning_rate = tf.keras.experimental.CosineDecay(
        base_lr, decay_steps=epochs * -(-len(X_train) // global_batch_size)
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)
    optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.fp16)

    def loss(y_true, logits):
        # categorical crossentropy
        log_p = logits - tf.math.reduce_logsumexp(logits, axis=-1, keepdims=True)
        loss = -tf.math.reduce_sum(y_true * log_p, axis=-1)
        # Label smoothing <https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/>
        label_smoothing = 0.2
        kl = -tf.math.reduce_mean(log_p, axis=-1)
        loss = (1 - label_smoothing) * loss + label_smoothing * kl
        return loss

    model.compile(optimizer, loss, metrics=["acc"], experimental_run_tf_function=False)
    model.summary(print_fn=logger.info)
    tf.keras.utils.plot_model(
        model, args.results_dir / f"{args.data}.png", show_shapes=True
    )

    model.fit(
        create_dataset(
            X_train, y_train, batch_size, num_classes, shuffle=True, mode="train"
        ),
        steps_per_epoch=-(-len(X_train) // global_batch_size),
        validation_data=create_dataset(
            X_val, y_val, batch_size, num_classes, shuffle=True
        ),
        validation_steps=-(-len(X_val) * 3 // global_batch_size),
        validation_freq=100,
        epochs=epochs,
        callbacks=[
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
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=base_lr / 100, momentum=0.9, nesterov=True
    )
    optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.fp16)
    model.compile(optimizer, loss, metrics=["acc"], experimental_run_tf_function=False)
    model.fit(
        create_dataset(
            X_train, y_train, batch_size, num_classes, shuffle=True, mode="refine"
        ),
        steps_per_epoch=-(-len(X_train) // global_batch_size),
        validation_data=create_dataset(
            X_val, y_val, batch_size, num_classes, shuffle=True
        ),
        validation_steps=-(-len(X_val) * 3 // global_batch_size),
        validation_freq=10,
        epochs=refine_epoch,
        callbacks=[
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
        ],
        verbose=1 if hvd.rank() == 0 else 0,
    )

    logger.info(f"Arguments: --data={args.data}")
    # 検証/評価
    # 両方出してたら分けた意味ない気はするけど面倒なので両方出しちゃう
    # 普段はValを見ておいて最終評価はTestというお気持ち
    val_acc, val_ce = evaluate("Val", X_val, y_val, model, batch_size, num_classes)
    test_acc, test_ce = evaluate("Test", X_test, y_test, model, batch_size, num_classes)

    if hvd.rank() == 0:
        # 後で何かしたくなった時のために一応保存
        model.save(args.results_dir / f"{args.data}.h5", include_optimizer=False)

    return val_acc, val_ce, test_acc, test_ce


def evaluate(name, X_test, y_test, model, batch_size, num_classes):
    pred_test_list = []
    for _ in range(64):
        # shardingして推論＆TTA (再現性は無いので注意)
        shard_size = len(X_test) // hvd.size()
        shard_offset = shard_size * hvd.rank()
        s = X_test[shard_offset : shard_offset + shard_size]
        pred_test = model.predict(
            create_dataset(
                s,
                np.zeros((len(s),), dtype=np.int32),
                batch_size,
                num_classes,
                mode="refine",
            ),
            steps=-(-len(s) // batch_size),
            verbose=1 if hvd.rank() == 0 else 0,
        )
        pred_test = hvd.allgather(pred_test).numpy()
        pred_test_list.append(pred_test)
    pred_test = np.mean(pred_test_list, axis=0)
    # 評価
    acc = sklearn.metrics.accuracy_score(y_test, pred_test.argmax(axis=-1))
    ce = sklearn.metrics.log_loss(y_test, scipy.special.softmax(pred_test, axis=-1))
    logger.info(f"{name} Accuracy:      {acc:.4f}")
    logger.info(f"{name} Cross Entropy: {ce:.4f}")
    return acc, ce


def load_data(data):
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
    # 末尾1万件を検証データとする (本実装独自)
    X_val, y_val = X_train[-10000:], y_train[-10000:]
    # 訓練データはクラスごとに先頭から切り出す (Train with 1000準拠)
    X_train, y_train = extract1000(X_train, y_train, num_classes=num_classes)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes


def extract1000(X, y, num_classes):
    """https://github.com/mastnk/train1000 を参考にクラスごとに均等に先頭から取得する処理。"""
    num_data = 1000
    num_per_class = num_data // num_classes

    index_list = []
    for c in range(num_classes):
        index_list.extend(np.where(y == c)[0][:num_per_class])
    assert len(index_list) == num_data

    return X[index_list], y[index_list]


def create_network(input_shape, num_classes):
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

    def blocks(filters, count, down=True):
        def layers(x):
            if down:
                x = conv2d(filters, kernel_size=4, strides=2)(x)
                x = bn()(x)
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
    x = blocks(128, 8, down=False)(x)
    x = blocks(256, 8)(x)
    x = blocks(512, 8)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        num_classes, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


def create_dataset(X, y, batch_size, num_classes, shuffle=False, mode="test"):
    """generator。"""
    if mode == "train":
        aug1 = A.Compose(
            [
                RandomTransform((32, 32), p=1),
                RandomCompose(
                    [
                        A.Equalize(mode="pil", by_channels=True, p=0.125),
                        A.Equalize(mode="pil", by_channels=False, p=0.125),
                        A.CLAHE(p=0.125),
                        A.RandomBrightnessContrast(brightness_by_max=True, p=0.5),
                        A.HueSaturationValue(val_shift_limit=0, p=0.5),
                        A.Posterize(num_bits=(4, 7), p=0.125),
                        A.Solarize(threshold=(50, 255 - 50), p=0.125),
                        A.Blur(blur_limit=1, p=0.125),
                        A.IAASharpen(alpha=(0, 0.5), p=0.125),
                        A.IAAEmboss(alpha=(0, 0.5), p=0.125),
                        A.GaussNoise(var_limit=(0, 10.0 ** 2), p=0.125),
                        RandomErasing(alpha=0.125, p=0.25),
                    ],
                    p=1,
                ),
            ]
        )
        aug2 = RandomErasing(p=0.5)
    elif mode == "refine":
        aug1 = RandomTransform.create_refine((32, 32))
        aug2 = A.Compose([])
    else:
        aug1 = A.Compose([])
        aug2 = A.Compose([])

    def do_aug1(img):
        return aug1(image=img)["image"]

    def do_aug2(img):
        return aug2(image=img)["image"]

    def process1(X, y):
        X = tf.numpy_function(do_aug1, inp=[X], Tout=tf.uint8)
        X = tf.cast(X, tf.float32)
        y = tf.one_hot(y, num_classes, dtype=tf.float32)
        return X, y

    def process2(X, y):
        X = tf.numpy_function(do_aug2, inp=[X], Tout=tf.float32)
        X = tf.ensure_shape(X, (None, None, None))
        X = X / 127.5 - 1
        return X, y

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(buffer_size=len(X)) if shuffle else ds
    ds = ds.map(process1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if mode == "train":
        assert shuffle
        ds = mixup(ds, process2)
    else:
        ds = ds.map(process2)
    ds = ds.repeat() if shuffle else ds  # シャッフル時はバッチサイズを固定するため先にrepeat
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def mixup(
    ds: tf.data.Dataset,
    postmix_fn: typing.Callable[..., typing.Any] = None,
    num_parallel_calls: int = None,
):
    """tf.dataでのmixup: <https://arxiv.org/abs/1710.09412>

    Args:
        ds: 元のデータセット
        postmix_fn: mixup後の処理
        num_parallel_calls: premix_fnの並列数

    """

    @tf.function
    def mixup_fn(*data):
        r = tf.random.uniform(())
        data = [
            tf.cast(d[0], tf.float32) * r + tf.cast(d[1], tf.float32) * (1 - r)
            for d in data
        ]
        return data if postmix_fn is None else postmix_fn(*data)

    ds = ds.repeat()
    ds = ds.batch(2)
    ds = ds.map(mixup_fn, num_parallel_calls=num_parallel_calls)
    return ds


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


class RandomTransform(A.DualTransform):
    """Flip, Scale, Resize, Rotateをまとめて処理。

    Args:
        size: 出力サイズ(h, w)
        flip: 反転の有無(vertical, horizontal)
        translate: 平行移動の量(vertical, horizontal)
        border_mode: edge, reflect, wrap

    """

    @classmethod
    def create_refine(
        cls,
        size: typing.Tuple[int, int],
        flip: typing.Tuple[bool, bool] = (False, True),
        translate: typing.Tuple[float, float] = (0.0625, 0.0625),
        border_mode: str = "edge",
        clip_bboxes=True,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        """Refined Data Augmentation <https://arxiv.org/abs/1909.09148> 用の控えめバージョンを作成する。"""
        return cls(
            size=size,
            flip=flip,
            translate=translate,
            border_mode=border_mode,
            scale_prob=0.0,
            aspect_prob=0.0,
            rotate_prob=0.0,
            clip_bboxes=clip_bboxes,
            always_apply=always_apply,
            p=p,
        )

    def __init__(
        self,
        size: typing.Tuple[int, int],
        flip: typing.Tuple[bool, bool] = (False, True),
        translate: typing.Tuple[float, float] = (0.125, 0.125),
        scale_prob: float = 0.5,
        scale_range: typing.Tuple[float, float] = (2 / 3, 3 / 2),
        base_scale: float = 1.0,
        aspect_prob: float = 0.5,
        aspect_range: typing.Tuple[float, float] = (3 / 4, 4 / 3),
        rotate_prob: float = 0.25,
        rotate_range: typing.Tuple[int, int] = (-15, +15),
        border_mode: str = "edge",
        clip_bboxes: bool = True,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.size = size
        self.flip = flip
        self.translate = translate
        self.scale_prob = scale_prob
        self.base_scale = base_scale
        self.scale_range = scale_range
        self.aspect_prob = aspect_prob
        self.aspect_range = aspect_range
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        self.border_mode = border_mode
        self.clip_bboxes = clip_bboxes

    def apply(self, image, m, interp=None, **params):
        # pylint: disable=arguments-differ
        cv2_border = {
            "edge": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT_101,
            "wrap": cv2.BORDER_WRAP,
        }[self.border_mode]

        if interp == "nearest":
            cv2_interp = cv2.INTER_NEAREST
        else:
            # 縮小ならINTER_AREA, 拡大ならINTER_LANCZOS4
            sh, sw = image.shape[:2]
            dr = cv2.perspectiveTransform(
                np.array([(0, 0), (sw, 0), (sw, sh), (0, sh)])
                .reshape((-1, 1, 2))
                .astype(np.float32),
                m,
            ).reshape((4, 2))
            dw = min(np.linalg.norm(dr[1] - dr[0]), np.linalg.norm(dr[2] - dr[3]))
            dh = min(np.linalg.norm(dr[3] - dr[0]), np.linalg.norm(dr[2] - dr[1]))
            cv2_interp = cv2.INTER_AREA if dw <= sw or dh <= sh else cv2.INTER_LANCZOS4

        if image.ndim == 2 or image.shape[-1] in (1, 3):
            image = cv2.warpPerspective(
                image, m, self.size[::-1], flags=cv2_interp, borderMode=cv2_border
            )
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
        else:
            resized_list = [
                cv2.warpPerspective(
                    image[:, :, ch],
                    m,
                    self.size[::-1],
                    flags=cv2_interp,
                    borderMode=cv2_border,
                )
                for ch in range(image.shape[-1])
            ]
            image = np.transpose(resized_list, (1, 2, 0))
        return image

    def apply_to_bbox(self, bbox, m, image_size, **params):
        # pylint: disable=arguments-differ
        del params
        bbox = np.asarray(bbox)
        assert bbox.shape == (4,)
        bbox *= np.array([image_size[1], image_size[0]] * 2)
        bbox = cv2.perspectiveTransform(
            bbox.reshape((-1, 1, 2)).astype(np.float32), m
        ).reshape(bbox.shape)
        if bbox[2] < bbox[0]:
            bbox = bbox[[2, 1, 0, 3]]
        if bbox[3] < bbox[1]:
            bbox = bbox[[0, 3, 2, 1]]
        bbox /= np.array([self.size[1], self.size[0]] * 2)
        assert bbox.shape == (4,)
        if self.clip_bboxes:
            bbox = np.clip(bbox, 0, 1)
        return tuple(bbox)

    def apply_to_keypoint(self, keypoint, m, **params):
        # pylint: disable=arguments-differ
        del params
        xy = np.asarray(keypoint[:2])
        xy = cv2.perspectiveTransform(
            xy.reshape((-1, 1, 2)).astype(np.float32), m
        ).reshape(xy.shape)
        return tuple(xy) + tuple(keypoint[2:])

    def apply_to_mask(self, img, interp=None, **params):
        # pylint: disable=arguments-differ
        del interp
        return self.apply(img, interp="nearest", **params)

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        scale = (
            self.base_scale
            * np.exp(
                random.uniform(np.log(self.scale_range[0]), np.log(self.scale_range[1]))
            )
            if random.random() <= self.scale_prob
            else self.base_scale
        )
        ar = (
            np.exp(
                random.uniform(
                    np.log(self.aspect_range[0]), np.log(self.aspect_range[1])
                )
            )
            if random.random() <= self.aspect_prob
            else 1.0
        )

        flip_v = self.flip[0] and random.random() <= 0.5
        flip_h = self.flip[1] and random.random() <= 0.5
        scale_v = scale / np.sqrt(ar)
        scale_h = scale * np.sqrt(ar)
        degrees = (
            random.uniform(self.rotate_range[0], self.rotate_range[1])
            if random.random() <= self.rotate_prob
            else 0
        )
        pos_v = random.uniform(0, 1)
        pos_h = random.uniform(0, 1)
        translate_v = random.uniform(-self.translate[0], self.translate[0])
        translate_h = random.uniform(-self.translate[1], self.translate[1])
        # 左上から時計回りに座標を用意
        src_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        dst_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        # 反転
        if flip_h:
            dst_points = dst_points[[1, 0, 3, 2]]
        if flip_v:
            dst_points = dst_points[[3, 2, 1, 0]]
        # 移動
        src_points[:, 0] -= translate_h
        src_points[:, 1] -= translate_v
        # 回転
        theta = degrees * np.pi * 2 / 360
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([[c, -s], [s, c]], dtype=np.float32)
        src_points = np.dot(r, (src_points - 0.5).T).T + 0.5
        # スケール変換
        src_points[:, 0] /= scale_h
        src_points[:, 1] /= scale_v
        src_points[:, 0] -= (1 / scale_h - 1) * pos_h
        src_points[:, 1] -= (1 / scale_v - 1) * pos_v
        # 変換行列の作成
        src_points[:, 0] *= image.shape[1]
        src_points[:, 1] *= image.shape[0]
        dst_points[:, 0] *= self.size[1]
        dst_points[:, 1] *= self.size[0]
        m = cv2.getPerspectiveTransform(src_points, dst_points)
        return {"m": m, "image_size": image.shape[:2]}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "size",
            "flip",
            "translate",
            "scale_prob",
            "scale_range",
            "base_scale",
            "aspect_prob",
            "aspect_range",
            "rotate_prob",
            "rotate_range",
            "border_mode",
        )


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
