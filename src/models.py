import tensorflow as tf
from transformers import CLIPProcessor, TFCLIPVisionModel


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


def Generator(cfg):
    inputs = tf.keras.layers.Input(
        shape=[cfg["image_height"], cfg["image_width"], cfg["num_in_channels"]]
    )

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        cfg["num_out_channels"],
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def CNNDiscriminator(cfg):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = tf.keras.layers.Input(
        shape=[cfg["image_height"], cfg["image_width"], cfg["num_in_channels"]],
        name="input_image",
    )
    tar = tf.keras.layers.Input(
        shape=[cfg["image_height"], cfg["image_width"], cfg["num_out_channels"]],
        name="target_image",
    )

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

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

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def CLIPDiscriminator(cfg):
    model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    model.trainable = cfg[
        "clip_fine_tune"
    ]  # TODO: Make only the last layer trainable when cfg["clip_fine_tine"] is True.
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = tf.keras.layers.Input(shape=(cfg["image_height"], cfg["image_width"], 3))
    
    outputs = processor(images=inputs, return_tensors="tf")
    outputs = model(**outputs)
    # last_hidden_state = outputs.last_hidden_state  # shape: (cfg["batch_size"], N_patches, 768)
    if cfg["clip_output_type"] == "cls":
        outputs = outputs.pooler_output  # CLS states, shape: (cfg["batch_size"], 768)
    elif cfg["clip_output_type"] == "mean":
        outputs = tf.reduce_mean(
            outputs.last_hidden_state, axis=1
        )  # averaged last hidden states, shape: (cfg["batch_size"], 768)
    outputs = tf.keras.layers.BatchNormalization(outputs)
    outputs = tf.keras.layers.Dense(256, activation="leaky_relu")(outputs)
    outputs = tf.keras.layers.BatchNormalization(outputs)
    outputs = tf.keras.layers.Dense(256, activation="leaky_relu")(outputs) + outputs
    outputs = tf.keras.layers.BatchNormalization(outputs)
    outputs = tf.keras.layers.Dense(1, activation="linear")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
