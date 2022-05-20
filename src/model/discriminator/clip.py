import tensorflow as tf
from transformers import TFCLIPVisionModel


def CLIPDiscriminator(cfg):
    model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    if cfg["clip_fine_tune"]:
        model.clip.vision_model.embeddings.trainable = False
        model.clip.vision_model.pre_layernorm.trainable = False
        model.clip.vision_model.post_layernorm.trainable = True
        for layer in model.clip.vision_model.encoder.layers:
            if layer.name == "layers_._11":
                layer.trainable = True
            else:
                layer.trainable = False
    else:
        model.trainable = False

    image_mean = tf.constant([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    image_std = tf.constant([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]

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

    outputs = tf.keras.layers.concatenate(
        [
            embed(
                cfg=cfg,
                processed_images=process(
                    images=repeat(inp, n_repeats=cfg["num_out_channels"]),
                    image_mean=image_mean,
                    image_std=image_std,
                ),
            ),
            embed(
                cfg=cfg,
                processed_images=process(
                    images=tar, image_mean=image_mean, image_std=image_std
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
