"""
Shared YAML config loading for training scripts.

Loads the `defaults:` section of one or more configs/*.yaml files and
applies them as argparse defaults -- so running a script with NO flags
picks up whatever's currently in the yaml (single source of truth for
"what will this run actually use"), while any individual value can
still be overridden with an explicit --flag on the command line for a
one-off experiment or a sweep launcher.

Usage:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    ...
    apply_yaml_defaults(parser, "configs/model.yaml", "configs/pretrain.yaml")
    args = parser.parse_args()
    # args.lr is now whatever configs/pretrain.yaml says, unless --lr
    # was actually passed on the command line, which still wins.
"""

import yaml  # PyYAML -- not declared in pyproject.toml (this repo has no
             # [project.dependencies] section at all), but almost
             # certainly already installed as a transitive dependency of
             # transformers/huggingface_hub. `pip install pyyaml` if not.


def load_yaml_defaults(*paths: str) -> dict:
    """Merges the `defaults:` dict of each yaml file, in order -- later
    files override earlier ones on key collisions. Keys with no
    matching argparse argument are simply ignored by apply_yaml_defaults
    below (not every documented yaml entry has a CLI flag; each yaml
    file's own comments say which do)."""
    merged: dict = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        merged.update(data.get("defaults", {}))
    return merged


def apply_yaml_defaults(parser, *paths: str) -> None:
    """Sets each yaml default as the parser's default for any
    already-registered argument with a matching dest. Call this AFTER
    all add_argument() calls, BEFORE parse_args() -- argparse applies
    set_defaults() values only to arguments not explicitly passed on
    the command line, which is exactly the override behavior wanted
    here."""
    defaults = load_yaml_defaults(*paths)
    known_dests = {action.dest for action in parser._actions}
    applicable = {k: v for k, v in defaults.items() if k in known_dests}
    parser.set_defaults(**applicable)
