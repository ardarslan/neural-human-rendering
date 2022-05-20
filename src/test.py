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

    cfg["num_in_channels"] = 1
    cfg["num_out_channels"] = 3

    generator = get_model(cfg, model_type="generator")
    discriminator = get_model(cfg, model_type="discriminator")

    generator_optimizer = get_optimizer(cfg, optimizer_type="generator")
    discriminator_optimizer = get_optimizer(cfg, optimizer_type="discriminator")

    test_ds = get_dataset(cfg, split="test", shuffle=False)

    checkpoint_saver = get_checkpoint_saver(
        cfg, generator, discriminator, generator_optimizer, discriminator_optimizer
    )

    restore_last_checkpoint(cfg, checkpoint_saver)

    generate_final_images(cfg, generator, test_ds)
