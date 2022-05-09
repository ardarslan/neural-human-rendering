import math
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt

"""
code adapted from https://keras.io/examples/vision/vit_small_ds/
"""


class ShiftedPatchTokenization(layers.Layer):
    def __init__(
        self,
        cfg,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vanilla = cfg["vanilla"]  # Flag to swtich to vanilla patch extractor
        self.image_size = cfg["image_height"]
        self.patch_size = cfg["patch_size"]
        self.half_patch = cfg["patch_size"] // 2
        self.flatten_patches = layers.Reshape(
            ((cfg["image_height"] // cfg["patch_size"]) ** 2, -1)
        )
        self.projection = layers.Dense(units=cfg["projection_dim"])
        self.layer_norm = layers.LayerNormalization(epsilon=cfg["norm_eps"])

    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

    def __call__(self, images):
        if not self.vanilla:
            # Concat the shifted images with the original image
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)
        return tokens, patches


def show_sample(image, cfg):
    # Get a random image from the training dataset
    # and resize the image

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(cfg["image_height"], cfg["image_height"])
    )

    # Vanilla patch maker: This takes an image and divides into
    # patches as in the original ViT paper
    (token, patch) = ShiftedPatchTokenization(cfg, vanilla=True)(resized_image / 255.0)
    (token, patch) = (token[0], patch[0])
    n = patch.shape[0]
    count = 1
    plt.figure(figsize=(4, 4))
    for row in range(n):
        for col in range(n):
            plt.subplot(n, n, count)
            count = count + 1
            image = tf.reshape(
                patch[row][col], (cfg["patch_size"], cfg["patch_size"], 3)
            )
            plt.imshow(image)
            plt.axis("off")
    plt.show()

    # Shifted Patch Tokenization: This layer takes the image, shifts it
    # diagonally and then extracts patches from the concatinated images
    (token, patch) = ShiftedPatchTokenization(cfg, vanilla=False)(resized_image / 255.0)
    (token, patch) = (token[0], patch[0])
    n = patch.shape[0]
    shifted_images = ["ORIGINAL", "LEFT-UP", "LEFT-DOWN", "RIGHT-UP", "RIGHT-DOWN"]
    for index, name in enumerate(shifted_images):
        print(name)
        count = 1
        plt.figure(figsize=(4, 4))
        for row in range(n):
            for col in range(n):
                plt.subplot(n, n, count)
                count = count + 1
                image = tf.reshape(
                    patch[row][col], (cfg["patch_size"], cfg["patch_size"], 5 * 3)
                )
                plt.imshow(image[..., 3 * index : 3 * index + 3])
                plt.axis("off")
        plt.show()


class PatchEncoder(layers.Layer):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = (cfg["image_height"] // cfg["patch_size"]) ** 2
        self.position_embedding = layers.Embedding(
            input_dim=(cfg["image_height"] // cfg["patch_size"]) ** 2,
            output_dim=cfg["projection_dim"],
        )
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def __call__(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches


class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The trainable temperature term. The initial value is
        # the square root of the key dimension.
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def VITDiscriminator(cfg):
    inputs = layers.Input(
        shape=(cfg["image_height"], cfg["image_width"], cfg["num_in_channels"]),
        name="input_image",
    )
    targets = layers.Input(
        shape=(cfg["image_height"], cfg["image_width"], cfg["num_out_channels"]),
        name="target_image",
    )

    data = layers.concatenate([inputs, targets])
    # Augment data.
    # augmented = data_augmentation(inputs)
    augmented = data
    # Create patches.
    (tokens, _) = ShiftedPatchTokenization(cfg)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(cfg)(tokens)

    diag_attn_mask = 1 - tf.eye((cfg["image_height"] // cfg["patch_size"]) ** 2)
    diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

    # Create multiple layers of the Transformer block.
    for _ in range(cfg["num_transformer_layers"]):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        if not cfg["vanilla"]:
            attention_output = MultiHeadAttentionLSA(
                num_heads=cfg["num_heads"], key_dim=cfg["projection_dim"], dropout=0.1
            )(x1, x1, attention_mask=diag_attn_mask)
        else:
            attention_output = layers.MultiHeadAttention(
                num_heads=cfg["num_heads"], key_dim=cfg["projection_dim"], dropout=0.1
            )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=cfg["transformer_units"], dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=cfg["mlp_head_units"], dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(cfg["num_classes"])(features)
    # Create the Keras model.
    model = keras.Model(inputs=[inputs, targets], outputs=logits)
    return model
