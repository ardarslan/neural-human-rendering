from utils import (
    get_argument_parser,
    set_seeds,
    get_model,
    get_optimizer,
    get_checkpoint_saver,
    restore_last_checkpoint,
    generate_final_images,
    get_dataset,
)

from evaluation_metrics import save_evaluation_scores_of_final_images

if __name__ == "__main__":
    cfg = get_argument_parser().parse_args().__dict__
    set_seeds(cfg)
    assert (
        cfg["experiment_time"].isdigit()
        and isinstance(cfg["experiment_time"], str)
        and len(cfg["experiment_time"]) == 10
    ), "experiment_time should be a string of length 10."

    cfg["mlp_head_units"] = [2048, 1024]
    cfg["transformer_units"] = [cfg["projection_dim"] * 2, cfg["projection_dim"]]

    if cfg["dataset_type"] == "face":
        cfg["num_in_channels"] = 1
        cfg["num_out_channels"] = 3
        cfg["image_height"] = 256
        cfg["image_width"] = 256
    elif cfg["dataset_type"] == "body_smplpix":
        cfg["num_in_channels"] = 3
        cfg["num_out_channels"] = 3
        cfg["image_height"] = 256
        cfg["image_width"] = 256
    else:
        raise Exception(f"Not a valid dataset_type {dataset_type}.")

    generator = get_model(cfg, model_type="generator")
    discriminator = get_model(cfg, model_type="discriminator")

    generator_optimizer = get_optimizer(cfg, optimizer_type="generator")
    discriminator_optimizer = get_optimizer(cfg, optimizer_type="discriminator")

    test_ds = get_dataset(cfg, split="test")

    checkpoint_saver = get_checkpoint_saver(
        cfg, generator, discriminator, generator_optimizer, discriminator_optimizer
    )

    restore_last_checkpoint(cfg, checkpoint_saver)

    generate_final_images(cfg, generator, test_ds)

    save_evaluation_scores_of_final_images(cfg)
