import os

from omegaconf import DictConfig, OmegaConf


def save_model_config(config: DictConfig) -> None:
    config_model = {}
    for k, v in OmegaConf.to_container(
        config.model.model_architectures.model_config
    ).items():  # type: ignore
        if isinstance(v, str):
            if v.startswith("${"):
                v = config[k]
        config_model[k] = v
    config_model["target_columns"] = config.model["target_columns"]
    model_path = os.path.join(config.dir.output_dir, "model_config.yaml")
    with open(model_path, "w") as f:
        OmegaConf.save(config_model, f)

    return
