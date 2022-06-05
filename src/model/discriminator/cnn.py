import tensorflow as tf
from model.model_utils import downsample


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

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(128, 4)(down2)

    # layer 1
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv1 = tf.keras.layers.Conv2D(
        128, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv1)
    leaky_relu1 = tf.keras.layers.LeakyReLU()(batchnorm1)

    # layer 2
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu1)
    conv2 = tf.keras.layers.Conv2D(
        128, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad2)
    batchnorm2 = tf.keras.layers.BatchNormalization()(conv2)
    leaky_relu2 = tf.keras.layers.LeakyReLU()(batchnorm2)

    # layer 3
    zero_pad3 = tf.keras.layers.ZeroPadding2D()(leaky_relu2)
    conv3 = tf.keras.layers.Conv2D(
        128, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad3)
    batchnorm3 = tf.keras.layers.BatchNormalization()(conv3)
    leaky_relu3 = tf.keras.layers.LeakyReLU()(batchnorm3)

    zero_pad4 = tf.keras.layers.ZeroPadding2D()(leaky_relu3)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad4
    )
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
