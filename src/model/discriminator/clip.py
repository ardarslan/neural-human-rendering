import tensorflow as tf
from transformers import TFCLIPVisionModel


def CLIPDiscriminator(cfg):
    model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    model.trainable = cfg[
        "clip_fine_tune"
    ]  # TODO: Make only the last layer trainable when cfg["clip_fine_tine"] is True.
    image_mean = tf.constant([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    image_std = tf.constant([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]

    def center_crop(images):
        """
            Input:
                tf.Tensor
                Shape: (cfg["batch_size"], cfg["image_height"], cfg["image_width"], num_channels)
            Output:
                tf.Tensor
                Shape: (cfg["batch_size"], 224, 224, num_channels)
        """
        return images[
            :,
            int(cfg["image_height"] / 2 - 112) : int(cfg["image_height"] / 2 + 112),
            int(cfg["image_width"] / 2 - 112) : int(cfg["image_width"] / 2 + 112),
            :,
        ]

    def repeat(images, n_repeats):
        """
        Repeats images in channel dimension.
        """
        return tf.repeat(images, repeats=[n_repeats], axis=3)

    def process(images, image_mean, image_std):
        """
        Inputs:
            images: tf.Tensor with shape (batch_size, image_height, image_width, num_channels), values are in [-1, 1].
            image_mean: tf.Tensor with shape (1, num_channels, 1, 1)
            image_std: tf.Tensor with shape (1, num_channels, 1, 1)
        """
        result = (images + 1.0) / 2.0
        result = tf.transpose(result, perm=[0, 3, 1, 2])
        result = (result - image_mean) / image_std
        return {"pixel_values": result}

    def embed(cfg, processed_images):
        outputs = model(**processed_images)
        if cfg["clip_output_type"] == "cls":
            outputs = (
                outputs.pooler_output
            )  # CLS states, shape: (cfg["batch_size"], 768)
        elif cfg["clip_output_type"] == "mean":
            outputs = tf.reduce_mean(
                outputs.last_hidden_state, axis=1
            )  # averaged last hidden states, shape: (cfg["batch_size"], 768)
        return outputs

    inp = tf.keras.layers.Input(
        shape=[cfg["image_height"], cfg["image_width"], cfg["num_in_channels"]],
        name="input_image",
    )
    tar = tf.keras.layers.Input(
        shape=[cfg["image_height"], cfg["image_width"], cfg["num_out_channels"]],
        name="target_image",
    )

    outputs = tf.keras.layers.concatenate(
        [
            embed(
                cfg=cfg,
                processed_images=process(
                    images=repeat(center_crop(inp), n_repeats=cfg["num_out_channels"]),
                    image_mean=image_mean,
                    image_std=image_std,
                ),
            ),
            embed(
                cfg=cfg,
                processed_images=process(
                    images=center_crop(tar), image_mean=image_mean, image_std=image_std
                ),
            ),
        ]
    )
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(64, activation="leaky_relu")(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(64, activation="leaky_relu")(outputs) + outputs
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(1, activation="linear")(outputs)

    return tf.keras.Model(inputs=[inp, tar], outputs=outputs)
