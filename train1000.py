#!/usr/bin/env python3
"""Train with 1000"""
from __future__ import annotations

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
    steps_per_epoch = -(-len(X_train) // global_batch_size)

    input_shape = X_train.shape[1:]
    model = create_network(input_shape, num_classes)
    learning_rate = CosineAnnealing(base_lr, decay_steps=epochs * steps_per_epoch)
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)
    optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.fp16)

    def loss(y_true, logits):
        # categorical crossentropy
        log_p = logits - tf.math.reduce_logsumexp(logits, axis=-1, keepdims=True)
        loss = -tf.math.reduce_sum(y_true * log_p, axis=-1)
        # Label smoothing <https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/>
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
        steps_per_epoch=steps_per_epoch,
        validation_data=create_dataset(
            X_val, y_val, batch_size, num_classes, shuffle=True
        ),
        validation_steps=-(-len(X_val) * 3 // global_batch_size),
        validation_freq=[1] + list(range(epochs, 1, -int(epochs ** 0.5))),
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
        steps_per_epoch=steps_per_epoch,
        validation_data=create_dataset(
            X_val, y_val, batch_size, num_classes, shuffle=True
        ),
        validation_steps=-(-len(X_val) * 3 // global_batch_size),
        validation_freq=[1] + list(range(refine_epoch, 1, -int(refine_epoch ** 0.5))),
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
                x = conv2d(filters)(x)
                x = BlurPooling2D(taps=4)(x)
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
                        A.Sharpen(alpha=(0, 0.5), p=0.125),
                        A.Emboss(alpha=(0, 0.5), p=0.125),
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

    def process1(X_i, y_i):
        X_i = tf.numpy_function(
            lambda img: aug1(image=img)["image"], inp=[X_i], Tout=tf.uint8
        )
        X_i = tf.cast(X_i, tf.float32)
        y_i = tf.one_hot(y_i, num_classes, dtype=tf.float32)
        return X_i, y_i

    def process2(X_i, y_i):
        X_i = tf.numpy_function(
            lambda img: aug2(image=img)["image"], inp=[X_i], Tout=tf.float32
        )
        X_i = tf.ensure_shape(X_i, (None, None, None))
        X_i = X_i / 127.5 - 1
        return X_i, y_i

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(buffer_size=len(X)) if shuffle else ds
    ds = ds.map(
        process1, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle
    )
    if mode == "train":
        assert shuffle
        ds = mixup(ds, process2)
    else:
        ds = ds.map(process2)
    ds = ds.repeat() if shuffle else ds  # シャッフル時はバッチサイズを固定するため先にrepeat
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
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

    def __call__(self, *args, force_apply=False, **data):
        """変換の適用。"""
        backup = self.transforms.transforms.copy()
        try:
            random.shuffle(self.transforms.transforms)
            return super().__call__(*args, force_apply=force_apply, **data)
        finally:
            self.transforms.transforms = backup


class RandomTransform(A.DualTransform):
    """Flip, Scale, Resize, Rotateをまとめて処理。

    Args:
        size: 出力サイズ(h, w)
        flip: 反転の有無(vertical, horizontal)
        translate: 平行移動の量(vertical, horizontal)
        border_mode: edge, reflect, wrap, zero, half, one
        clip_bboxes: はみ出すbboxをclippingするか否か
        with_bboxes: Trueならできるだけbboxを1つ以上含むようにcropする (bboxesが必須になってしまうため既定値False)
        mode: "normal", "preserve_aspect", "crop"

    """

    @classmethod
    def create_refine(
        cls,
        size: tuple[int, int],
        flip: tuple[bool, bool] = (False, True),
        translate: tuple[float, float] = (0.0625, 0.0625),
        border_mode: str = "edge",
        clip_bboxes: bool = True,
        with_bboxes: bool = False,
        mode: str = "normal",
        always_apply: bool = False,
        p: float = 1.0,
    ) -> RandomTransform:
        """Refined Data Augmentation <https://arxiv.org/abs/1909.09148> 用の控えめバージョンを作成する。"""
        return cls(
            size=size,
            flip=flip,
            translate=translate,
            scale_prob=0.0,
            aspect_prob=0.0,
            rotate_prob=0.0,
            border_mode=border_mode,
            clip_bboxes=clip_bboxes,
            with_bboxes=with_bboxes,
            mode=mode,
            always_apply=always_apply,
            p=p,
        )

    @classmethod
    def create_test(
        cls,
        size: tuple[int, int],
        border_mode: str = "edge",
        clip_bboxes: bool = True,
        mode: str = "normal",
        always_apply: bool = False,
        p: float = 1.0,
    ) -> RandomTransform:
        """Data Augmentation無しバージョン(リサイズのみ)を作成する。"""
        return cls(
            size=size,
            flip=(False, False),
            translate=(0.0, 0.0),
            scale_prob=0.0,
            aspect_prob=0.0,
            rotate_prob=0.0,
            border_mode=border_mode,
            clip_bboxes=clip_bboxes,
            mode=mode,
            always_apply=always_apply,
            p=p,
        )

    def __init__(
        self,
        size: tuple[int, int],
        flip: tuple[bool, bool] = (False, True),
        translate: tuple[float, float] = (0.125, 0.125),
        scale_prob: float = 0.5,
        scale_range: tuple[float, float] = (2 / 3, 3 / 2),
        base_scale: float = 1.0,
        aspect_prob: float = 0.5,
        aspect_range: tuple[float, float] = (3 / 4, 4 / 3),
        rotate_prob: float = 0.25,
        rotate_range: tuple[int, int] = (-15, +15),
        border_mode: str = "edge",
        clip_bboxes: bool = True,
        with_bboxes: bool = False,
        mode: str = "normal",
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
        self.with_bboxes = with_bboxes
        self.mode = mode

    def apply(self, img, m, interp=None, **params):
        # pylint: disable=arguments-differ
        cv2_border, borderValue = {
            "edge": (cv2.BORDER_REPLICATE, None),
            "reflect": (cv2.BORDER_REFLECT_101, None),
            "wrap": (cv2.BORDER_WRAP, None),
            "zero": (cv2.BORDER_CONSTANT, [0, 0, 0]),
            "half": (
                cv2.BORDER_CONSTANT,
                [0.5, 0.5, 0.5]
                if img.dtype in (np.float32, np.float64)
                else [127, 127, 127],
            ),
            "one": (
                cv2.BORDER_CONSTANT,
                [1, 1, 1] if img.dtype in (np.float32, np.float64) else [255, 255, 255],
            ),
        }[self.border_mode]

        if interp == "nearest":
            cv2_interp = cv2.INTER_NEAREST
        else:
            # 縮小ならINTER_AREA, 拡大ならINTER_LANCZOS4
            sh, sw = img.shape[:2]
            dr = cv2.perspectiveTransform(
                np.array([(0, 0), (sw, 0), (sw, sh), (0, sh)])
                .reshape((-1, 1, 2))
                .astype(np.float32),
                m,
            ).reshape((4, 2))
            dw = min(np.linalg.norm(dr[1] - dr[0]), np.linalg.norm(dr[2] - dr[3]))
            dh = min(np.linalg.norm(dr[3] - dr[0]), np.linalg.norm(dr[2] - dr[1]))
            cv2_interp = cv2.INTER_AREA if dw <= sw or dh <= sh else cv2.INTER_LANCZOS4

        if img.ndim == 2 or img.shape[-1] in (1, 3):
            img = cv2.warpPerspective(
                img,
                m,
                self.size[::-1],
                flags=cv2_interp,
                borderMode=cv2_border,
                borderValue=borderValue,
            )
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        else:
            resized_list = [
                cv2.warpPerspective(
                    img[:, :, ch],
                    m,
                    self.size[::-1],
                    flags=cv2_interp,
                    borderMode=cv2_border,
                    borderValue=borderValue,
                )
                for ch in range(img.shape[-1])
            ]
            img = np.transpose(resized_list, (1, 2, 0))
        return img

    def apply_to_bbox(self, bbox, **params):
        return self.apply_to_bboxes([tuple(bbox) + (1, 1, "A")], **params)[0][:4]

    def apply_to_bboxes(self, bboxes, m, image_size, **params):
        # pylint: disable=arguments-differ
        del params
        if len(bboxes) <= 0:
            return bboxes
        etc = [tuple(bbox[4:]) for bbox in bboxes]
        bboxes = np.array([bbox[:4] for bbox in bboxes], dtype=np.float32)

        # 座標変換
        bboxes *= [[image_size[1], image_size[0]] * 2]
        bboxes = cv2.perspectiveTransform(bboxes.reshape((-1, 1, 2)), m).reshape(
            bboxes.shape
        )
        bboxes /= [[self.size[1], self.size[0]] * 2]

        # x1>x2やy1>y2になっていたら直す
        bboxes = bboxes.reshape((-1, 2, 2))
        bboxes = np.concatenate([bboxes.min(axis=1), bboxes.max(axis=1)], axis=-1)
        assert (bboxes[:, 0] <= bboxes[:, 2]).all()
        assert (bboxes[:, 1] <= bboxes[:, 3]).all()

        if self.clip_bboxes:
            bboxes = np.clip(bboxes, 0, 1)

        return [tuple(bbox) + etc for bbox, etc in zip(bboxes, etc)]

    def apply_to_keypoint(self, keypoint, **params):
        return self.apply_to_keypoints([keypoint], **params)[0]

    def apply_to_keypoints(self, keypoints, m, **params):
        # pylint: disable=arguments-differ
        del params
        etc = [tuple(keypoint[2:]) for keypoint in keypoints]
        xys = np.array([keypoint[:2] for keypoint in keypoints], dtype=np.float32)
        xys = cv2.perspectiveTransform(xys.reshape((-1, 1, 2)), m).reshape(xys.shape)
        return [tuple(xy) + etc for xy, etc in zip(xys, etc)]

    def apply_to_mask(self, img, interp=None, **params):
        # pylint: disable=arguments-differ
        del interp
        return self.apply(img, interp="nearest", **params)

    def get_params_dependent_on_targets(self, params):
        if self.with_bboxes and len(params["bboxes"]) >= 1:
            # self.with_bboxes == Trueかつ元々bboxが1個以上あるなら、
            # 出来るだけ有効なbboxが1個以上あるようになるまでretryする。
            # (ただし極端な条件の場合ランダム任せだと厳しいのでリトライの上限は適当)
            for _ in range(100):  # retry
                d = self._get_params(params)
                bboxes = np.asarray(
                    [
                        np.clip(bbox[:4], 0, 1)
                        for bbox in self.apply_to_bboxes(params["bboxes"], **d)
                    ]
                )
                valid_bboxes = sum(
                    np.prod(np.maximum(bb[2:] - bb[:2], 0.0)) > 0 for bb in bboxes
                )
                if valid_bboxes >= 1:
                    break
        else:
            d = self._get_params(params)
        return d

    def _get_params(self, params):
        img = params["image"]
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
        scale = np.array([scale / np.sqrt(ar), scale * np.sqrt(ar)])
        degrees = (
            random.uniform(self.rotate_range[0], self.rotate_range[1])
            if random.random() <= self.rotate_prob
            else 0
        )
        pos = np.array([random.uniform(-0.5, +0.5), random.uniform(-0.5, +0.5)])
        translate = np.array(
            [
                random.uniform(-self.translate[0], self.translate[0]),
                random.uniform(-self.translate[1], self.translate[1]),
            ]
        )
        # 左上から時計回りに座標を用意
        src_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        if self.mode == "normal":
            # アスペクト比を無視して出力サイズに合わせる
            dst_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        elif self.mode == "preserve_aspect":
            # アスペクト比を維持するように縮小する
            if img.shape[0] < img.shape[1]:
                # 横長
                hr = img.shape[0] / img.shape[1]
                yr = (1 - hr) / 2
                dst_points = np.array(
                    [[0, yr], [1, yr], [1, yr + hr], [0, yr + hr]], dtype=np.float32
                )
            else:
                # 縦長
                wr = img.shape[1] / img.shape[0]
                xr = (1 - wr) / 2
                dst_points = np.array(
                    [[xr, 0], [xr + wr, 0], [xr + wr, 1], [xr, 1]], dtype=np.float32
                )
        elif self.mode == "crop":
            # 入力サイズによらず固定サイズでcrop
            hr = self.size[0] / img.shape[0]
            wr = self.size[1] / img.shape[1]
            yr = random.uniform(0, 1 - hr)
            xr = random.uniform(0, 1 - wr)
            dst_points = np.array(
                [[xr, yr], [xr + wr, yr], [xr + wr, yr + hr], [xr, yr + hr]],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        # 反転
        if flip_h:
            dst_points = dst_points[[1, 0, 3, 2]]
        if flip_v:
            dst_points = dst_points[[3, 2, 1, 0]]
        # 原点が中心になるように移動
        src_points -= 0.5
        # 回転
        theta = degrees * np.pi * 2 / 360
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([[c, -s], [s, c]], dtype=np.float32)
        src_points = np.dot(r, src_points.T).T
        # スケール変換
        src_points /= scale
        # 移動
        # スケール変換で余った分 + 最初に0.5動かした分 + translate分
        src_points += (1 - 1 / scale) * pos + 0.5 + translate / scale
        # 変換行列の作成
        src_points *= [img.shape[1], img.shape[0]]
        dst_points *= [self.size[1], self.size[0]]
        m = cv2.getPerspectiveTransform(src_points, dst_points)
        return {"m": m, "image_size": img.shape[:2]}

    @property
    def targets_as_params(self):
        if self.with_bboxes:
            return ["image", "bboxes"]
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
            "clip_bboxes",
            "with_bboxes",
            "mode",
        )


class RandomErasing(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """Random Erasing <https://arxiv.org/abs/1708.04896>"""

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


@tf.keras.utils.register_keras_serializable()
class BlurPooling2D(tf.keras.layers.Layer):
    """Blur Pooling Layer <https://arxiv.org/abs/1904.11486>"""

    def __init__(self, taps=5, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.taps = taps
        self.strides = normalize_tuple(strides, 2)
        self.kernel = None

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        input_shape = list(input_shape)
        input_shape[1] = (
            input_shape[1] + int(input_shape[1]) % self.strides[0]
        ) // self.strides[0]
        input_shape[2] = (
            input_shape[2] + int(input_shape[2]) % self.strides[1]
        ) // self.strides[1]
        return tuple(input_shape)

    def build(self, input_shape):
        in_filters = int(input_shape[-1])
        pascals_tr = np.zeros((self.taps, self.taps), dtype=np.float32)
        pascals_tr[0, 0] = 1
        for i in range(1, self.taps):
            pascals_tr[i, :] = pascals_tr[i - 1, :]
            pascals_tr[i, 1:] += pascals_tr[i - 1, :-1]
        filter1d = pascals_tr[self.taps - 1, :]
        filter2d = filter1d[np.newaxis, :] * filter1d[:, np.newaxis]
        filter2d = filter2d * (self.taps ** 2 / filter2d.sum())
        kernel = np.tile(filter2d[:, :, np.newaxis, np.newaxis], (1, 1, in_filters, 1))
        self.kernel = tf.constant(kernel, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        kernel = tf.cast(self.kernel, inputs.dtype)
        s = tf.shape(inputs)
        outputs = tf.nn.depthwise_conv2d(
            inputs, kernel, strides=(1,) + self.strides + (1,), padding="SAME"
        )
        norm = tf.ones((s[0], s[1], s[2], 1), dtype=inputs.dtype)
        norm = tf.nn.depthwise_conv2d(
            norm,
            kernel[:, :, :1, :],
            strides=(1,) + self.strides + (1,),
            padding="SAME",
        )
        return outputs / norm

    def get_config(self):
        config = {"taps": self.taps, "strides": self.strides}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def normalize_tuple(value, n: int) -> tuple[int, ...]:
    """n個の要素を持つtupleにして返す。"""
    assert value is not None
    if isinstance(value, int):
        return (value,) * n
    else:
        value = tuple(value)
        assert len(value) == n
        return value


@tf.keras.utils.register_keras_serializable()
class CosineAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine Annealing without restart。

    Args:
        initial_learning_rate: 初期学習率
        decay_steps: 全体のステップ数 (len(train_set) // (batch_size * app.num_replicas_in_sync * tk.hvd.size()) * epochs)
        warmup_steps: 最初にlinear warmupするステップ数。既定値は1000。ただし最大でdecay_steps // 8。
        name: 名前

    References:
        - SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>

    """

    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        warmup_steps: int = 1000,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = min(warmup_steps, decay_steps // 8)
        self.name = name
        assert initial_learning_rate > 0
        assert 0 <= self.warmup_steps < self.decay_steps

    def __call__(self, step):
        with tf.name_scope(self.name or "CosineAnnealing"):
            initial_learning_rate = tf.cast(
                self.initial_learning_rate, dtype=tf.float32
            )
            decay_steps = tf.cast(self.decay_steps, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            step = tf.cast(step, tf.float32)

            # linear warmup
            fraction1 = (step + 1) / tf.math.maximum(warmup_steps, 1)

            # cosine annealing
            wdecay_steps = decay_steps - warmup_steps
            warmed_steps = tf.math.minimum(step - warmup_steps + 1, wdecay_steps)
            r = warmed_steps / wdecay_steps
            fraction2 = 0.5 * (1.0 + tf.math.cos(np.pi * r))

            fraction = tf.where(step < warmup_steps, fraction1, fraction2)
            return initial_learning_rate * fraction

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }


if __name__ == "__main__":
    _main()
