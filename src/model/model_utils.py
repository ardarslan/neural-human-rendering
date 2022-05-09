import tensorflow as tf


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def center_crop(cfg, images):
    """
    Input:
        tf.Tensor
        Shape: (cfg["full_image_height"], cfg["full_image_width"], num_channels)
                or
               (cfg["batch_size"], cfg["full_image_height"], cfg["full_image_width"], num_channels)
    Output:
        tf.Tensor
        Shape: (cfg["cropped_image_height"], cfg["cropped_image_width"], num_channels)
                or
               (cfg["batch_size"], cfg["cropped_image_height"], cfg["cropped_image_width"], num_channels)
    """
    return tf.image.crop_to_bounding_box(
        images,
        offset_height=int((cfg["full_image_height"] - cfg["cropped_image_height"]) / 2),
        offset_width=int((cfg["full_image_width"] - cfg["cropped_image_width"]) / 2),
        target_height=cfg["cropped_image_height"],
        target_width=cfg["cropped_image_width"],
    )


def pad(cfg, images):
    """
    Input:
        tf.Tensor
        Shape: (cfg["cropped_image_height"], cfg["cropped_image_width"], num_channels)
                or
               (cfg["batch_size"], cfg["cropped_image_height"], cfg["cropped_image_width"], num_channels)
    Output:
        tf.Tensor
        Shape: (cfg["full_image_height"], cfg["full_image_width"], num_channels)
                or
               (cfg["batch_size"], cfg["full_image_height"], cfg["full_image_width"], num_channels)
    """
    return tf.image.pad_to_bounding_box(
        images,
        offset_height=int((cfg["full_image_height"] - cfg["cropped_image_height"]) / 2),
        offset_width=int((cfg["full_image_width"] - cfg["cropped_image_width"]) / 2),
        target_height=cfg["full_image_height"],
        target_width=cfg["full_image_width"],
    )
