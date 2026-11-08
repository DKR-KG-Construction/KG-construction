"""Guard the held-out test split against configuration changes after freeze."""
import hashlib
import json
import os
import pathlib

from .config import rpath
from .extract import fingerprint
from .schema import load_schema


def _relative_to_root(root, path):
    root = pathlib.Path(root).resolve()
    path = pathlib.Path(path).resolve()
    try:
        return path.relative_to(root)
    except ValueError:
        return pathlib.Path(os.path.relpath(path, root))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_test_configuration(cfg):
    paths = {
        "config": pathlib.Path(cfg.get("_config_file", pathlib.Path(cfg["_root"]) / "config.yaml")),
        "prompt": rpath(cfg, cfg["extractor"]["prompt_file"]),
        "schema": rpath(cfg, cfg["paths"]["schema"]),
        "test_data": rpath(cfg, cfg["paths"]["test"]),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze test setup; missing files: {', '.join(missing)}")
    manifest = {
        "files": {name: {"path": str(_relative_to_root(cfg["_root"], path)),
                         "sha256": _sha256(path)}
                  for name, path in paths.items()},
        "test_extractor_fingerprint": fingerprint(cfg, load_schema(cfg), "test"),
        "eval": cfg["eval"],
        "judge": cfg["judge"],
        "matching": cfg["matching"],
        "c": cfg["c"],
        "s": cfg["s"],
        "p": cfg["p"],
    }
    target = rpath(cfg, cfg["paths"].get("test_freeze", "docs/test_freeze.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return target


def assert_test_frozen(cfg):
    target = rpath(cfg, cfg["paths"].get("test_freeze", "docs/test_freeze.json"))
    if not target.is_file():
        raise ValueError(f"Test split is locked. Freeze the final setup first: "
                         f"python -m kgrepair freeze-test (writes {target})")
    with open(target, encoding="utf-8") as f:
        manifest = json.load(f)
    expected = manifest.get("files", {})
    for name in ("config", "prompt", "schema", "test_data"):
        if name not in expected:
            raise ValueError(f"{target} is missing the frozen {name} checksum")
        frozen_path = expected[name]["path"]
        path = pathlib.Path(cfg["_root"]) / frozen_path if not pathlib.Path(frozen_path).is_absolute() else pathlib.Path(frozen_path)
        if not path.is_file() or _sha256(path) != expected[name]["sha256"]:
            raise ValueError(f"Test setup changed after freeze ({name}); "
                             "review the development results and explicitly freeze again.")
    if manifest.get("test_extractor_fingerprint") != fingerprint(cfg, load_schema(cfg), "test"):
        raise ValueError("Test extractor fingerprint changed after freeze; explicitly freeze again.")
    for key in ("eval", "judge", "matching", "c", "s", "p"):
        if manifest.get(key) != cfg[key]:
            raise ValueError(f"Test setting {key!r} changed after freeze; explicitly freeze again.")
