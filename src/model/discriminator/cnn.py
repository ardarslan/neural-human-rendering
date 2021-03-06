import tensorflow as tf
from model.model_utils import downsample, pad


def CNNDiscriminator(cfg):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = tf.keras.layers.Input(
        shape=[
            cfg["cropped_image_height"],
            cfg["cropped_image_width"],
            cfg["num_in_channels"],
        ],
        name="input_image",
    )
    tar = tf.keras.layers.Input(
        shape=[
            cfg["cropped_image_height"],
            cfg["cropped_image_width"],
            cfg["num_out_channels"],
        ],
        name="target_image",
    )

    if cfg["dataset_type"] == "face":
        x = tf.keras.layers.concatenate([inp, tar])

        down1 = downsample(32, 4, False)(x)
        down2 = downsample(32, 4)(down1)
        down3 = downsample(64, 4)(down2)

        # layer 1
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
        conv1 = tf.keras.layers.Conv2D(
            64, 4, strides=1, kernel_initializer=initializer, use_bias=False
        )(zero_pad1)
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv1)
        leaky_relu1 = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu1)
        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
            zero_pad2
        )

    elif cfg["dataset_type"] == "face_reconstruction":
        x = tf.keras.layers.concatenate(
            [pad(cfg, inp), pad(cfg, tar)]
        )  # (batch_size, 256, 256, channels*2)

        down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(
            512, 4, strides=1, kernel_initializer=initializer, use_bias=False
        )(
            zero_pad1
        )  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(
            leaky_relu
        )  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
            zero_pad2
        )  # (batch_size, 30, 30, 1)
    else:
        raise Exception(f"Not a valid dataset_type {cfg['dataset_type']}.")

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
