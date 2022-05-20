import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from model.model_utils import pad

"""
code adapted from https://keras.io/examples/vision/mlp_image_classification/
"""


class Patches(layers.Layer):
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches


def MLPMixerDiscrimator(cfg, positional_encoding=False):
    num_patches = (cfg['full_image_height'] // cfg['mlp_patch_size']) ** 2
    mlpmixer_blocks = keras.Sequential(
        [MLPMixerLayer(num_patches, cfg['mlp_embedding_dim'], cfg['mlp_dropout_rate'], cfg)
            for _ in range(cfg['mlp_num_blocks'])])

    inputs = layers.Input(
        shape=(
            cfg["cropped_image_height"],
            cfg["cropped_image_width"],
            cfg["num_in_channels"],
        ),
        name="input_image",
    )
    targets = layers.Input(
        shape=(
            cfg["cropped_image_height"],
            cfg["cropped_image_width"],
            cfg["num_out_channels"],
        ),
        name="target_image",
    )

    data = layers.concatenate([pad(cfg, inputs), pad(cfg, targets)])

    #inputs = layers.Input(shape=(cfg['full_image_height'], cfg['full_image_width']))
    # Augment data.
    #augmented = data_augmentation(inputs)
    augmented = data

    # Create patches.
    patches = Patches(cfg['mlp_patch_size'], num_patches)(augmented)
    # Encode patches to generate a [batch_size, num_patches, embedding_dim] tensor.
    x = layers.Dense(units=cfg['mlp_embedding_dim'])(patches)
    if positional_encoding:
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=cfg['mlp_embedding_dim']
        )(positions)
        x = x + position_embedding
    # Process x using the module blocks.
    x = mlpmixer_blocks(x)
    # Apply global average pooling to generate a [batch_size, embedding_dim] representation tensor.
    representation = layers.GlobalAveragePooling1D()(x)
    # Apply dropout.
    representation = layers.Dropout(rate=cfg['mlp_dropout_rate'])(representation)
    # Compute logits outputs.
    logits = layers.Dense(cfg['num_classes'])(representation)
    # Create the Keras model.
    return keras.Model(inputs=[inputs, targets], outputs=logits)


class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate, cfg, *args, **kwargs):
        super(MLPMixerLayer, self).__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=cfg['mlp_embedding_dim']),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x