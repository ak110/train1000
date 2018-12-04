#!/usr/bin/env python3
import argparse
import logging
import pathlib

import albumentations as A
import cv2
import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import sklearn.externals.joblib as joblib
import sklearn.metrics
import tensorflow as tf

USE_TF_KERAS = True
if USE_TF_KERAS:
    keras = tf.keras
    import horovod.tensorflow.keras as hvd
else:
    import keras
    import horovod.keras as hvd


def _main():
    try:
        import better_exceptions
        better_exceptions.hook()
    except BaseException:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='cifar10', choices=('mnist', 'fashion_mnist', 'cifar10', 'cifar100'))
    parser.add_argument('--model', default='lite', choices=('lite', 'full'))
    parser.add_argument('--check', action='store_true', help='3epochだけお試し実行(動作確認用)')
    parser.add_argument('--results-dir', default=pathlib.Path('results'), type=pathlib.Path)
    args = parser.parse_args()

    hvd.init()

    handlers = [logging.StreamHandler()]
    if hvd.rank() == 0:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.results_dir / f'{args.data}.{args.model}.log', encoding='utf-8'))
    logging.basicConfig(format='[%(levelname)-5s] %(message)s', level='DEBUG', handlers=handlers)
    logger = logging.getLogger(__name__)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    keras.backend.set_session(tf.Session(config=config))

    (X_train, y_train), (X_test, y_test), num_classes = _load_data(args.data)

    if args.check:
        epochs = 3
    else:
        epochs = 3600 if args.model == 'full' else 1800
    batch_size = 16
    base_lr = 1e-3 * batch_size * hvd.size()

    model = _create_network(X_train.shape[1:], num_classes, args.model)
    optimizer = keras.optimizers.SGD(lr=base_lr, momentum=0.9, nesterov=True)
    optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.fp16)
    model.compile(optimizer, 'categorical_crossentropy')
    model.summary(print_fn=logger.info if hvd.rank() == 0 else lambda x: x)

    callbacks = [
        _cosine_annealing_callback(base_lr, epochs),
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]
    model.fit_generator(_generate(X_train, y_train, batch_size, num_classes, shuffle=True, data_augmentation=True),
                        steps_per_epoch=int(np.ceil(len(X_train) / batch_size / hvd.size())),
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=1 if hvd.rank() == 0 else 0)

    if hvd.rank() == 0:
        # 検証
        pred_test = model.predict_generator(
            _generate(X_test, np.zeros((len(X_test),), dtype=np.int32), batch_size, num_classes),
            int(np.ceil(len(X_test) / batch_size)),
            verbose=1 if hvd.rank() == 0 else 0)
        logger.info(f'Arguments:          --data={args.data} --model={args.model}')
        logger.info(f'Test Accuracy:      {sklearn.metrics.accuracy_score(y_test, pred_test.argmax(axis=-1)):.4f}')
        logger.info(f'Test Cross Entropy: {sklearn.metrics.log_loss(y_test, pred_test):.4f}')
        # 後で何かしたくなった時のために一応保存
        model.save(args.results_dir / f'{args.data}.{args.model}.h5', include_optimizer=False)


def _load_data(data):
    """データの読み込み。"""
    (X_train, y_train), (X_test, y_test) = {
        'mnist': keras.datasets.mnist.load_data,
        'fashion_mnist': keras.datasets.fashion_mnist.load_data,
        'cifar10': keras.datasets.cifar10.load_data,
        'cifar100': keras.datasets.cifar100.load_data,
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


def _create_network(input_shape, num_classes, model):
    """ネットワークを作成して返す。"""
    reg = keras.regularizers.l2(1e-5)

    def _conv2d(filters, kernel_size=3, strides=1, use_act=True):
        def _layers(x):
            x = keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                    padding='same', use_bias=False,
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=reg,
                                    bias_regularizer=reg)(x)
            if model == 'full':
                x = MixFeat()(x)
            x = _bn_act(use_act=use_act)(x)
            return x
        return _layers

    def _bn_act(use_act=True):
        def _layers(x):
            x = keras.layers.BatchNormalization(gamma_regularizer=reg,
                                                beta_regularizer=reg)(x)
            x = keras.layers.Activation('elu')(x) if use_act else x
            return x
        return _layers

    x = inputs = keras.layers.Input(input_shape)
    for stage, filters in enumerate([128, 256, 384]):
        if stage == 0:
            x = _conv2d(128, use_act=False)(x)
        else:
            if model == 'full':
                x = _conv2d(filters, 1, use_act=False)(x)
                x = ParallelGridPooling2D()(x)
            else:
                x = _conv2d(filters, 2, strides=2, use_act=False)(x)
        for block in range(8):
            sc = x
            x = _conv2d(filters // 4)(x)
            for d in range(7):
                t = _conv2d(filters // 4)(x)
                x = keras.layers.concatenate([x, t])
            x = _conv2d(filters, 1, use_act=False)(x)
            x = keras.layers.add([sc, x])
        x = _bn_act()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_uniform',
                           kernel_regularizer=reg,
                           bias_regularizer=reg)(x)
    if model == 'full':
        x = ParallelGridGather(4 * 4)(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


class ParallelGridPooling2D(keras.layers.Layer):
    """Parallel Grid Poolingレイヤー。"""

    def __init__(self, pool_size=(2, 2), **kargs):
        super().__init__(**kargs)
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
        assert len(self.pool_size) == 2

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] % self.pool_size[0] == 0  # パディングはとりあえず未対応
        assert input_shape[2] % self.pool_size[1] == 0  # パディングはとりあえず未対応
        b, h, w, c = input_shape
        return b, h // self.pool_size[0], w // self.pool_size[1], c

    def call(self, inputs, **kwargs):
        shape = keras.backend.shape(inputs)
        int_shape = keras.backend.int_shape(inputs)
        rh, rw = self.pool_size
        b, h, w, c = shape[0], shape[1], shape[2], int_shape[3]
        inputs = keras.backend.reshape(inputs, (b, h // rh, rh, w // rw, rw, c))
        inputs = tf.transpose(inputs, perm=(2, 4, 0, 1, 3, 5))
        inputs = keras.backend.reshape(inputs, (rh * rw * b, h // rh, w // rw, c))
        return inputs

    def get_config(self):
        config = {'pool_size': self.pool_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ParallelGridGather(keras.layers.Layer):
    """ParallelGridPoolingでparallelにしたのを戻すレイヤー。"""

    def __init__(self, r, **kargs):
        super().__init__(**kargs)
        self.r = r

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        shape = keras.backend.shape(inputs)
        b = shape[0]
        gather_shape = keras.backend.concatenate([[self.r, b // self.r], shape[1:]], axis=0)
        inputs = keras.backend.reshape(inputs, gather_shape)
        inputs = keras.backend.mean(inputs, axis=0)
        return inputs

    def get_config(self):
        config = {'r': self.r}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MixFeat(keras.layers.Layer):
    """MixFeat <https://openreview.net/forum?id=HygT9oRqFX>"""

    def __init__(self, sigma=0.2, **kargs):
        self.sigma = sigma
        super().__init__(**kargs)

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        def _passthru():
            return inputs

        def _mixfeat():
            @tf.custom_gradient
            def _forward(x):
                shape = keras.backend.shape(x)
                indices = keras.backend.arange(start=0, stop=shape[0])
                indices = tf.random_shuffle(indices)
                rs = keras.backend.concatenate([keras.backend.constant([1], dtype='int32'), shape[1:]])
                r = keras.backend.random_normal(rs, 0, self.sigma, dtype='float16')
                theta = keras.backend.random_uniform(rs, -np.pi, +np.pi, dtype='float16')
                a = 1 + r * keras.backend.cos(theta)
                b = r * keras.backend.sin(theta)
                y = x * keras.backend.cast(a, keras.backend.floatx()) + keras.backend.gather(x, indices) * keras.backend.cast(b, keras.backend.floatx())

                def _backword(dx):
                    inv = tf.invert_permutation(indices)
                    return dx * keras.backend.cast(a, keras.backend.floatx()) + keras.backend.gather(dx, inv) * keras.backend.cast(b, keras.backend.floatx())

                return y, _backword

            return _forward(inputs)

        return keras.backend.in_train_phase(_mixfeat, _passthru, training=training)

    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _cosine_annealing_callback(base_lr, epochs):
    """Cosine annealing。

    # [1608.03983] SGDR: Stochastic Gradient Descent with Warm Restarts
    https://arxiv.org/abs/1608.03983
    """
    def _cosine_annealing(ep, lr):
        min_lr = base_lr * 0.01
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * (ep + 1) / epochs))
    return keras.callbacks.LearningRateScheduler(_cosine_annealing)


def _generate(X, y, batch_size, num_classes, shuffle=False, data_augmentation=False):
    """generator。"""
    if data_augmentation:
        aug1 = A.Compose([
            A.PadIfNeeded(40, 40, border_mode=cv2.BORDER_CONSTANT, value=[127, 127, 127], p=1),
            _create_autoaugment(),
            A.RandomSizedCrop((20, 32), 32, 32),
            A.HorizontalFlip(),
        ])
        aug2 = A.Compose([A.Normalize(mean=0.5, std=0.5), A.Cutout(num_holes=1, max_h_size=16, max_w_size=16)])
    else:
        aug1 = A.Compose([])
        aug2 = A.Compose([A.Normalize(mean=0.5, std=0.5)])

    with joblib.Parallel(backend='threading', n_jobs=batch_size) as parallel:
        if shuffle:
            batch_indices = []
            for index in _generate_shuffled_indices(len(X)):
                batch_indices.append(index)
                if len(batch_indices) == batch_size:
                    yield _generate_batch(X, y, aug1, aug2, num_classes, data_augmentation, batch_indices, parallel)
                    batch_indices = []
        else:
            while True:
                for i in range(0, len(X), batch_size):
                    yield _generate_batch(X, y, aug1, aug2, num_classes, data_augmentation, range(i, i + batch_size), parallel)


def _generate_shuffled_indices(data_count):
    """シャッフルしたindexを無限に返すgenerator。"""
    all_indices = np.arange(data_count)
    while True:
        np.random.shuffle(all_indices)
        yield from all_indices


def _generate_batch(X, y, aug1, aug2, num_classes, data_augmentation, batch_indices, parallel):
    """1バッチずつの処理。"""
    jobs = [_generate_instance(X, y, aug1, aug2, num_classes, data_augmentation, i) for i in batch_indices]
    results = parallel(jobs)
    X_batch, y_batch = zip(*results)
    return np.array(X_batch), np.array(y_batch)


@joblib.delayed
def _generate_instance(X, y, aug1, aug2, num_classes, data_augmentation, index):
    """1サンプルずつの処理。"""
    X_i, y_i = X[index], y[index]
    X_i = aug1(image=X_i)['image']
    c_i = _to_categorical(y_i, num_classes)

    if data_augmentation:
        # Between-class Learning
        while True:
            t = np.random.choice(len(y))
            if y[t] != y_i:
                break
        X_t, y_t = X[t], y[t]
        X_t = aug1(image=X_t)['image']
        c_t = _to_categorical(y_t, num_classes)
        r = np.random.uniform(0.5, 1.0)
        X_i = (X_i * r + X_t * (1 - r)).astype(np.float32)
        c_i = (c_i * r + c_t * (1 - r)).astype(np.float32)

    X_i = aug2(image=X_i)['image']
    return X_i, c_i


def _to_categorical(index, num_classes):
    """indexからone-hot vectorを作成。(スカラー用)"""
    onehot = np.zeros((num_classes,), dtype=np.float32)
    onehot[index] = 1
    return onehot


def _create_autoaugment():
    """albumentationsでAutoAugment(CIFAR-10)な変換を作って返す。

    ■[1805.09501] AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501
    """
    sp = {
        'ShearX': lambda p, mag: Affine(shear_x_mag=mag, p=p),
        'ShearY': lambda p, mag: Affine(shear_y_mag=mag, p=p),
        'TranslateX': lambda p, mag: Affine(translate_x_mag=mag, p=p),
        'TranslateY': lambda p, mag: Affine(translate_y_mag=mag, p=p),
        'Rotate': lambda p, mag: A.Rotate(limit=mag / 9 * 30, p=p),
        'Color': lambda p, mag: Color(mag=mag, p=p),
        'Posterize': lambda p, mag: Posterize(mag=mag, p=p),
        'Solarize': lambda p, mag: Solarize(mag=mag, p=p),
        'Contrast': lambda p, mag: Contrast(mag=mag, p=p),
        'Sharpness': lambda p, mag: Sharpness(mag=mag, p=p),
        'Brightness': lambda p, mag: Brightness(mag=mag, p=p),
        'AutoContrast': lambda p, mag: AutoContrast(p=p),
        'Equalize': lambda p, mag: Equalize(p=p),
        'Invert': lambda p, mag: A.InvertImg(p=p),
    }
    return A.OneOf([
        A.Compose([sp['Invert'](0.1, 7), sp['Contrast'](0.2, 6)]),
        A.Compose([sp['Rotate'](0.7, 2), sp['TranslateX'](0.3, 9)]),
        A.Compose([sp['Sharpness'](0.8, 1), sp['Sharpness'](0.9, 3)]),
        A.Compose([sp['ShearY'](0.5, 8), sp['TranslateY'](0.7, 9)]),
        A.Compose([sp['AutoContrast'](0.5, 8), sp['Equalize'](0.9, 2)]),
        A.Compose([sp['ShearY'](0.2, 7), sp['Posterize'](0.3, 7)]),
        A.Compose([sp['Color'](0.4, 3), sp['Brightness'](0.6, 7)]),
        A.Compose([sp['Sharpness'](0.3, 9), sp['Brightness'](0.7, 9)]),
        A.Compose([sp['Equalize'](0.6, 5), sp['Equalize'](0.5, 1)]),
        A.Compose([sp['Contrast'](0.6, 7), sp['Sharpness'](0.6, 5)]),
        A.Compose([sp['Color'](0.7, 7), sp['TranslateX'](0.5, 8)]),
        A.Compose([sp['Equalize'](0.3, 7), sp['AutoContrast'](0.4, 8)]),
        A.Compose([sp['TranslateY'](0.4, 3), sp['Sharpness'](0.2, 6)]),
        A.Compose([sp['Brightness'](0.9, 6), sp['Color'](0.2, 8)]),
        A.Compose([sp['Solarize'](0.5, 2), sp['Invert'](0.0, 3)]),
        A.Compose([sp['Equalize'](0.2, 0), sp['AutoContrast'](0.6, 0)]),
        A.Compose([sp['Equalize'](0.2, 8), sp['Equalize'](0.8, 4)]),
        A.Compose([sp['Color'](0.9, 9), sp['Equalize'](0.6, 6)]),
        A.Compose([sp['AutoContrast'](0.8, 4), sp['Solarize'](0.2, 8)]),
        A.Compose([sp['Brightness'](0.1, 3), sp['Color'](0.7, 0)]),
        A.Compose([sp['Solarize'](0.4, 5), sp['AutoContrast'](0.9, 3)]),
        A.Compose([sp['TranslateY'](0.9, 9), sp['TranslateY'](0.7, 9)]),
        A.Compose([sp['AutoContrast'](0.9, 2), sp['Solarize'](0.8, 3)]),
        A.Compose([sp['Equalize'](0.8, 8), sp['Invert'](0.1, 3)]),
        A.Compose([sp['TranslateY'](0.7, 9), sp['AutoContrast'](0.9, 1)]),
    ], p=1)


class Affine(A.ImageOnlyTransform):

    def __init__(self, shear_x_mag=0, shear_y_mag=0, translate_x_mag=0, translate_y_mag=0, always_apply=False, p=.5):
        super().__init__(always_apply, p)
        self.shear_x_mag = shear_x_mag
        self.shear_y_mag = shear_y_mag
        self.translate_x_mag = translate_x_mag
        self.translate_y_mag = translate_y_mag

    def apply(self, image, shear_x, shear_y, translate_x, translate_y, **params):
        img = PIL.Image.fromarray(image, mode='RGB')
        data = (1, shear_x, translate_x, shear_y, 1, translate_y)
        return np.asarray(img.transform(img.size, PIL.Image.AFFINE, data, PIL.Image.BICUBIC, fillcolor=(128, 128, 128)), dtype=np.uint8)

    def get_params(self):
        return {
            'shear_x': self.shear_x_mag / 9 * 0.3 * np.random.choice([-1, 1]),
            'shear_y': self.shear_y_mag / 9 * 0.3 * np.random.choice([-1, 1]),
            'translate_x': self.translate_x_mag / 9 * (150 / 331) * np.random.choice([-1, 1]),
            'translate_y': self.translate_y_mag / 9 * (150 / 331) * np.random.choice([-1, 1]),
        }


class Color(A.ImageOnlyTransform):

    def __init__(self, mag=10, always_apply=False, p=.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, factor=1, **params):
        img = PIL.Image.fromarray(image, mode='RGB')
        return np.asarray(PIL.ImageEnhance.Color(img).enhance(factor), dtype=np.uint8)

    def get_params(self):
        return {'factor': 1 + self.mag / 9 * np.random.choice([-1, 1])}


class Posterize(A.ImageOnlyTransform):

    def __init__(self, mag=10, always_apply=False, p=.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, bit=8, **params):
        img = PIL.Image.fromarray(image, mode='RGB')
        return np.asarray(PIL.ImageOps.posterize(img, bit), dtype=np.uint8)

    def get_params(self):
        return {'bit': np.round(8 - self.mag * 4 / 9).astype(np.int)}


class Solarize(A.ImageOnlyTransform):

    def __init__(self, mag=10, always_apply=False, p=.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, threshold=128, **params):
        img = PIL.Image.fromarray(image, mode='RGB')
        return np.asarray(PIL.ImageOps.solarize(img, threshold), dtype=np.uint8)

    def get_params(self):
        return {'threshold': 256 - self.mag * 256 / 9}


class Contrast(A.ImageOnlyTransform):

    def __init__(self, mag=10, always_apply=False, p=.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, factor=1, **params):
        img = PIL.Image.fromarray(image, mode='RGB')
        return np.asarray(PIL.ImageEnhance.Contrast(img).enhance(factor), dtype=np.uint8)

    def get_params(self):
        return {'factor': 1 + self.mag / 9 * np.random.choice([-1, 1])}


class Sharpness(A.ImageOnlyTransform):

    def __init__(self, mag=10, always_apply=False, p=.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, factor=1, **params):
        img = PIL.Image.fromarray(image, mode='RGB')
        return np.asarray(PIL.ImageEnhance.Sharpness(img).enhance(factor), dtype=np.uint8)

    def get_params(self):
        return {'factor': 1 + self.mag / 9 * np.random.choice([-1, 1])}


class Brightness(A.ImageOnlyTransform):

    def __init__(self, mag=10, always_apply=False, p=.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, factor=1, **params):
        img = PIL.Image.fromarray(image, mode='RGB')
        return np.asarray(PIL.ImageEnhance.Brightness(img).enhance(factor), dtype=np.uint8)

    def get_params(self):
        return {'factor': 1 + self.mag / 9 * np.random.choice([-1, 1])}


class AutoContrast(A.ImageOnlyTransform):

    def __init__(self, always_apply=False, p=.5):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        img = PIL.Image.fromarray(image, mode='RGB')
        return np.asarray(PIL.ImageOps.autocontrast(img), dtype=np.uint8)

    def get_params(self):
        return {}


class Equalize(A.ImageOnlyTransform):

    def __init__(self, always_apply=False, p=.5):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        img = PIL.Image.fromarray(image, mode='RGB')
        return np.asarray(PIL.ImageOps.equalize(img), dtype=np.uint8)

    def get_params(self):
        return {}


if __name__ == '__main__':
    _main()
