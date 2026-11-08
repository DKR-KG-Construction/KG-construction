import pathlib
import yaml


def load_config(path="config.yaml"):
    path = pathlib.Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(path.parent)
    cfg["_config_file"] = str(path)
    return cfg


def rpath(cfg, rel):
    """Resolve a path from config relative to the config file's directory."""
    p = pathlib.Path(rel)
    return p if p.is_absolute() else pathlib.Path(cfg["_root"]) / p
