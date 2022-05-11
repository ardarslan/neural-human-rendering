import os
import cv2
import time
import random
import pprint
import argparse
import numpy as np
import tensorflow as tf
from model.generator.cnn import CNNGenerator
from model.discriminator.cnn import CNNDiscriminator
from model.discriminator.vit import VITDiscriminator
from model.discriminator.clip import CLIPDiscriminator


def get_argument_parser():
    parser = argparse.ArgumentParser(description="Arguments for running the script")
    parser.add_argument(
        "--datasets_dir",
        type=str,
        required=True,  # fix
        # default="/cluster/scratch/aarslan/virtual_humans_data",  # fix
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,  # fix
        # default="/cluster/scratch/aarslan/virtual_humans_checkpoints",  # fix
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,  # fix
        # default="face",  # fix
        help='Dataset type should be "face" or "body_smplpix".',
        choices=["face", "body_smplpix"],
    )
    parser.add_argument("--generator_type", type=str, choices=["cnn"], default="cnn")
    parser.add_argument(
        "--discriminator_type",
        type=str,
        choices=["cnn", "vit", "mlp-mixer", "clip"],
        # default="vit",  # fix
        required=True,  # fix
    )
    parser.add_argument(
        "--experiment_time",
        type=str,
        default="",
        help="To load a previous checkpoint. Used both in train.py and test.py",
    )
    parser.add_argument(
        "--l1_weight",
        type=int,
        # required=True,
        default=100,
        help="Weight of l1 loss in generator loss.",
    )
    parser.add_argument("--generator_lr", type=float, default=2e-4)
    parser.add_argument("--discriminator_lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--save_checkpoint_every_iter", type=int, default=5000  # fix
    )  # should be consistent if model will be loaded from a previous checkpoint
    parser.add_argument("--num_iterations", type=int, default=200000)
    # parser.add_argument("--max_gradient_norm", type=float, default=1.0)
    parser.add_argument("--full_image_height", type=int, default=256)
    parser.add_argument("--full_image_width", type=int, default=256)
    parser.add_argument("--cropped_image_height", type=int, default=224)
    parser.add_argument("--cropped_image_width", type=int, default=224)

    # VIT
    parser.add_argument("--patch_size", type=int, default=6, help="")
    parser.add_argument("--projection_dim", type=int, default=64, help="")
    parser.add_argument("--norm_eps", type=float, default=1e-6, help="")
    parser.add_argument("--vanilla", dest="vanilla", action="store_true")
    parser.add_argument("--num_heads", type=int, default=4, help="")
    parser.add_argument("--num_transformer_layers", type=int, default=8, help="")
    parser.add_argument("--num_classes", type=int, default=2, help="")

    # CLIP
    parser.add_argument("--clip_fine_tune", action="store_true")
    parser.add_argument("--clip_output_type", default="cls", help="")

    # FID
    parser.add_argument("--fid_dims", type=int, default=2048, help="")
    parser.add_argument("--fid_num_workers", type=int, default=None, help="")
    parser.add_argument("--fid_batch_size", type=int, default=64, help="")
    parser.add_argument("--fid_device", type=str, default=None, help="")

    return parser


def set_seeds(cfg):
    seed = cfg["seed"]
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def get_dataset(cfg, split, shuffle):
    input_images_dir = os.path.join(
        cfg["datasets_dir"], cfg["dataset_type"], split, "input"
    )
    input_image_paths = sorted(
        [
            os.path.join(input_images_dir, input_image_name)
            for input_image_name in os.listdir(input_images_dir)
            if input_image_name[-4:] == ".png"
        ]
    )
    if shuffle:
        random.shuffle(input_image_paths)
    real_image_paths = [
        input_image_path.replace("input", "output")
        for input_image_path in input_image_paths
    ]

    ds = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(input_image_paths),
            tf.data.Dataset.from_tensor_slices(real_image_paths),
        )
    )

    ds = ds.map(
        lambda input_image_path, real_image_path: load_images_helper(
            input_image_path=input_image_path,
            real_image_path=real_image_path,
            image_height=cfg["full_image_height"],
            image_width=cfg["full_image_width"],
            split=split,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(cfg["batch_size"])
    return ds


def get_model(cfg, model_type):
    if model_type == "generator":
        if cfg["generator_type"] == "cnn":
            ModelClass = CNNGenerator
        else:
            raise NotImplementedError()
    elif model_type == "discriminator":
        if cfg["discriminator_type"] == "cnn":
            ModelClass = CNNDiscriminator
        elif cfg["discriminator_type"] == "vit":
            ModelClass = VITDiscriminator
        elif cfg["discriminator_type"] == "clip":
            ModelClass = CLIPDiscriminator
        elif cfg["discriminator_type"] == "mlp-mixer":
            raise NotImplementedError()
        else:
            raise Exception(f"Not a valid discriminator_type {discriminator_type}.")
    else:
        raise Exception(f"Not a valid model_type {model_type}.")

    return ModelClass(cfg)


def get_optimizer(cfg, optimizer_type):
    if optimizer_type == "generator":
        return tf.keras.optimizers.Adam(cfg["generator_lr"], beta_1=0.5)
    elif optimizer_type == "discriminator":
        return tf.keras.optimizers.Adam(cfg["discriminator_lr"], beta_1=0.5)
    else:
        raise Exception(f"Not a valid optimizer_type {optimizer_type}.")


def get_time():
    return str(int(time.time()))


def resize(image, height, width):
    return tf.image.resize(
        image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )


def random_crop(input_image, real_image, image_height, image_width):
    random_start_y = np.random.randint(low=0, high=input_image.shape[0] - image_height)
    random_start_x = np.random.randint(low=0, high=input_image.shape[1] - image_width)

    cropped_input_image = input_image[
        random_start_y : random_start_y + image_height,
        random_start_x : random_start_x + image_width,
        :,
    ]

    cropped_real_image = real_image[
        random_start_y : random_start_y + image_height,
        random_start_x : random_start_x + image_width,
        :,
    ]

    return cropped_input_image, cropped_real_image


# Normalizing the images to [-1, 1]
def normalize(image):
    return (tf.cast(image, tf.float32) / 127.5) - 1.0


@tf.function()
def random_jitter(input_image, real_image, image_height, image_width):
    input_image = resize(
        input_image, int(image_height * 1.117), int(image_width * 1.117)
    )
    real_image = resize(real_image, int(image_height * 1.117), int(image_width * 1.117))

    input_image, real_image = random_crop(
        input_image, real_image, image_height, image_width
    )

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_images_helper(
    input_image_path, real_image_path, image_height, image_width, split
):
    input_image = tf.io.decode_png(tf.io.read_file(input_image_path))
    real_image = tf.io.decode_png(tf.io.read_file(real_image_path))

    if split == "train":
        input_image, real_image = random_jitter(
            input_image, real_image, image_height, image_width
        )
    elif split in ["validation", "test"]:
        input_image = resize(input_image, image_height, image_width)
        real_image = resize(real_image, image_height, image_width)
    else:
        raise Exception(f"Not a valid split {split}.")

    input_image = normalize(input_image)
    real_image = normalize(real_image)

    return input_image, real_image


def generator_loss(cfg, disc_generated_output, gen_output, target):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Center crop. Final shape: (batch_size, 224, 224, num_channels)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))  # Mean absolute error
    total_gen_loss = gan_loss + (cfg["l1_weight"] * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def get_new_directory(folder_names):
    joined_directory = os.path.join(*folder_names)
    os.makedirs(joined_directory, exist_ok=True)
    return joined_directory


def get_checkpoints_dir(cfg):
    return get_new_directory([cfg["checkpoints_dir"], cfg["experiment_time"]])


def get_checkpoint_saver(
    cfg, generator, discriminator, generator_optimizer, discriminator_optimizer
):
    checkpoints_dir = get_checkpoints_dir(cfg)
    checkpoint_prefix = os.path.join(checkpoints_dir, "ckpt")
    checkpoint_saver = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    return checkpoint_saver


def save_cfg(cfg):
    checkpoints_dir = get_checkpoints_dir(cfg)
    cfg_path = os.path.join(checkpoints_dir, "cfg.txt")
    with open(cfg_path, "w") as cfg_writer:
        cfg_writer.write(pprint.pformat(cfg, indent=4))


def save_new_checkpoint(cfg, checkpoint_saver):
    checkpoints_dir = get_checkpoints_dir(cfg)
    checkpoint_prefix = os.path.join(checkpoints_dir, "ckpt")
    checkpoint_saver.save(file_prefix=checkpoint_prefix)


def restore_last_checkpoint(cfg, checkpoint_saver):
    checkpoints_dir = get_checkpoints_dir(cfg)
    last_checkpoint_path = tf.train.latest_checkpoint(checkpoints_dir)
    checkpoint_saver.restore(last_checkpoint_path)
    last_checkpoint_number = int(last_checkpoint_path.split("/")[-1].split("-")[-1])
    start_iteration = last_checkpoint_number * cfg["save_checkpoint_every_iter"]
    return start_iteration


def get_summary_writer(cfg):
    checkpoints_dir = get_checkpoints_dir(cfg)
    log_dir = get_new_directory([checkpoints_dir, "logs"])
    summary_writer = tf.summary.create_file_writer(log_dir)
    return summary_writer


def generate_intermediate_images(
    cfg, model, example_inputs, example_targets, iteration
):
    predictions = model(example_inputs, training=True)
    # Getting the pixel values in the [0, 255] range to plot.
    file_names = ["input", "ground_truth", "predicted"]
    for i in range(10):
        current_images = [example_inputs[i], example_targets[i], predictions[i]]
        current_file_names = [f"{file_name}_{i}.png" for file_name in file_names]
        for current_file_name, current_image in zip(current_file_names, current_images):
            current_image = np.array(current_image)
            cv2.imwrite(
                os.path.join(
                    get_new_directory(
                        [
                            get_checkpoints_dir(cfg),
                            "intermediate_images",
                            f"iteration_{str(iteration.numpy()).zfill(7)}",
                        ]
                    ),
                    current_file_name,
                ),
                ((current_image[:, :, ::-1] * 0.5 + 0.5) * 255).astype(np.int32),
            )


def generate_final_images(cfg, model, test_ds):
    save_idx_counter = 0
    for test_input, _ in test_ds:
        prediction = model(test_input, training=True)
        # Getting the pixel values in the [0, 255] range to plot.
        for i in range(prediction.shape[0]):
            current_prediction = np.array(prediction[i])
            cv2.imwrite(
                os.path.join(
                    get_new_directory([get_checkpoints_dir(cfg), "final_images"]),
                    f"{str(save_idx_counter).zfill(7)}.png",
                ),
                ((current_prediction[:, :, ::-1] * 0.5 + 0.5) * 255).astype(np.int32),
            )
            save_idx_counter += 1
