import argparse
import yaml
import os
from pathlib import Path


def get_config():
    """Load configuration from YAML and normalize important paths.

    Returns an argparse.Namespace populated with YAML keys and a few
    convenience-normalized absolute paths (data paths, feature path).
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--yaml", type=str, default=None)
    argparser.add_argument("--split", type=str, default=None, help="Optional split name to override YAML")

    # Parse CLI args
    args = argparser.parse_args()

    # -------------------------------
    # Locate YAML
    # -------------------------------
    project_root = Path(__file__).resolve().parents[0]

    if args.yaml is not None:
        yaml_file_path = Path(args.yaml)
    else:
        yaml_file_path = project_root / "yaml" / "clear10" / "train.yaml"

    # -------------------------------
    # Load YAML file
    # -------------------------------
    with open(yaml_file_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Add YAML items to args (flatten simple two-level dict)
    for key in cfg.keys():
        sub_dict = cfg[key]
        for keyy in sub_dict.keys():
            setattr(args, keyy, sub_dict[keyy])

    # Respect an explicit CLI --split override
    if args.split is not None:
        setattr(args, "split", args.split)

    # -------------------------------
    # Dataset paths (LOCAL, ABSOLUTE)
    # -------------------------------
    # Build absolute paths relative to repository root
    def _abs(p):
        if p is None:
            return None
        p = Path(p)
        if not p.is_absolute():
            p = project_root / p
        return str(p.resolve())

    # Data paths with defaults
    args.data_folder_path = _abs(getattr(args, "data_folder_path", None)) or str((project_root / "dataset" / "release_in_the_wild").resolve())
    args.data_train_path = _abs(getattr(args, "data_train_path", None)) or str((project_root / "dataset" / "release_in_the_wild" / "dataset1").resolve())
    args.data_test_path = _abs(getattr(args, "data_test_path", None)) or str((project_root / "dataset" / "release_in_the_wild" / "dataset2").resolve())
    args.use_all_datasets = getattr(args, "use_all_datasets", True)

    # feature path
    args.feature_path = _abs(getattr(args, "feature_path", None)) or args.data_folder_path
    os.makedirs(args.feature_path, exist_ok=True)

    # -------------------------------
    # Defaults
    # -------------------------------
    args.class_list = getattr(args, "class_list", "real fake")
    if not getattr(args, "pretrain_feature", None):
        args.pretrain_feature = "moco_resnet50_clear_10_feature"

    return args
