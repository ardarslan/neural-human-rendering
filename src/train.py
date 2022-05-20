import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
import os
import time
import numpy as np
from itertools import islice

from utils import (
    save_cfg,
    generate_intermediate_images,
    generator_loss,
    discriminator_loss,
    get_argument_parser,
    set_seeds,
    get_dataset,
    get_experiment_name,
    get_sample_images,
    get_model,
    get_optimizer,
    get_summary_writer,
    get_checkpoint_saver,
    generate_final_images,
    save_new_checkpoint,
    restore_last_checkpoint,
)
from model.model_utils import center_crop


@tf.function
def train_step(
    cfg,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    input_image,
    target,
    summary_writer,
    iteration,
):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        input_image = center_crop(cfg, input_image)
        target = center_crop(cfg, target)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            cfg, disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", gen_total_loss, step=iteration // 1000)
        tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=iteration // 1000)
        tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=iteration // 1000)
        tf.summary.scalar("disc_loss", disc_loss, step=iteration // 1000)
        tf.summary.scalar(
            "disc_mean_real_output",
            tf.math.reduce_mean(disc_real_output),
            step=iteration // 1000,
        )
        tf.summary.scalar(
            "disc_mean_generated_output",
            tf.math.reduce_mean(disc_generated_output),
            step=iteration // 1000,
        )


def train(
    cfg,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    train_ds,
    val_ds,
    summary_writer,
    checkpoint_saver,
    start_iteration,
):
    # takes cfg["num_iterations"] + start_iteration elements from train dataset
    current_train_ds = iter(
        train_ds.repeat().take(cfg["num_iterations"] + start_iteration).enumerate()
    )

    train_inputs, train_targets = get_sample_images(cfg, train_ds)
    val_inputs, val_targets = get_sample_images(cfg, val_ds)

    # islice is used to skip the first "start_iteration" elements.
    for iteration, (input_image, target) in islice(
        current_train_ds,
        start_iteration,
        None,
        1,
    ):
        if (iteration) % 1000 == 0:
            generate_intermediate_images(
                cfg,
                generator,
                train_inputs,
                train_targets,
                val_inputs,
                val_targets,
                iteration,
            )
            print(f"Iteration: {iteration//1000}k")

        train_step(
            cfg,
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            input_image,
            target,
            summary_writer,
            iteration,
        )

        if (iteration + 1) % cfg["save_checkpoint_every_iter"] == 0:
            save_new_checkpoint(cfg, checkpoint_saver)

        iteration += 1


if __name__ == "__main__":
    cfg = get_argument_parser().parse_args().__dict__
    set_seeds(cfg)

    cfg["mlp_head_units"] = [2048, 1024]
    cfg["transformer_units"] = [cfg["projection_dim"] * 2, cfg["projection_dim"]]

    cfg["num_in_channels"] = 1
    cfg["num_out_channels"] = 3

    if cfg["experiment_name"] == "" or cfg["experiment_name"] is None:
        cfg["experiment_name"] = get_experiment_name()
        retraining = False
    elif (
        isinstance(cfg["experiment_name"], str) and len(cfg["experiment_name"]) == 43
    ):  # retraining
        retraining = True
    else:
        raise Exception(f"Not a valid experiment_name {experiment_name}.")

    save_cfg(cfg)

    generator = get_model(cfg, model_type="generator")
    discriminator = get_model(cfg, model_type="discriminator")

    generator_optimizer = get_optimizer(cfg, optimizer_type="generator")
    discriminator_optimizer = get_optimizer(cfg, optimizer_type="discriminator")

    checkpoint_saver = get_checkpoint_saver(
        cfg, generator, discriminator, generator_optimizer, discriminator_optimizer
    )

    summary_writer = get_summary_writer(cfg)

    if retraining:
        start_iteration = restore_last_checkpoint(cfg, checkpoint_saver)
    else:
        start_iteration = 0

    train_ds = get_dataset(cfg, split="train", shuffle=True)
    val_ds = get_dataset(cfg, split="validation", shuffle=True)

    train(
        cfg,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        train_ds,
        val_ds,
        summary_writer,
        checkpoint_saver,
        start_iteration,
    )
