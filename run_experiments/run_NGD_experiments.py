import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import random_split, TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import pickle
from typing import Any, Dict, Optional
import json
import random

# climb up to the repo root and add <repo>/src to Python's path
repo_root = Path().resolve().parents[0]   # parent of "notebooks"
sys.path.insert(0, str(repo_root / "src"))

from fisher_information.fim import FisherInformationMatrix
from models.image_classification_models import ConvModelMNIST
from models.train_test import *
#from prunning_methods.LTH import *
from fisher_information.NGD import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_json_arg(json_str: Optional[str], json_path: Optional[str]) -> Dict[str, Any]:
    if json_str and json_path:
        raise ValueError("Use either *_json or *_json_path (not both).")
    if json_path:
        return json.loads(Path(json_path).read_text(encoding="utf-8"))
    if json_str:
        return json.loads(json_str)
    return {}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def to_json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): to_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json_safe(v) for v in x]

    if np is not None and isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 50_000 else {"type": "ndarray", "shape": list(x.shape), "dtype": str(x.dtype)}

    if torch is not None and isinstance(x, torch.Tensor):
        n = x.numel()
        return x.detach().cpu().tolist() if n <= 50_000 else {
            "type": "tensor", "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)
        }

    return str(x)



def main():
    p = argparse.ArgumentParser()

    p.add_argument("--entry_module", default="experiment_entry", help="Module that defines run_once()")
    p.add_argument("--entry_fn", default="run_once", help="Function name inside entry_module")

    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)

    p.add_argument("--fim_args_json", default=None)
    p.add_argument("--fim_args_json_path", default=None)
    p.add_argument("--lth_args_json", default=None)
    p.add_argument("--lth_args_json_path", default=None)

    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--base_seed", type=int, default=12345)

    p.add_argument("--out_dir", required=True)
    p.add_argument("--save_json", action="store_true")
    p.add_argument("--save_all_in_one", action="store_true", help="Also saves a single all_runs.pkl at the end")

    args = p.parse_args()

    fim_args = read_json_arg(args.fim_args_json, args.fim_args_json_path)
    lth_args = read_json_arg(args.lth_args_json, args.lth_args_json_path)

    run_once = import_run_once(args.entry_module, args.entry_fn)

    root = Path(args.out_dir) / f"{args.model}__{args.dataset}"
    root.mkdir(parents=True, exist_ok=True)

    (root / "sweep_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "dataset": args.dataset,
                "fim_args": fim_args,
                "lth_args": lth_args,
                "n_runs": args.n_runs,
                "base_seed": args.base_seed,
                "entry_module": args.entry_module,
                "entry_fn": args.entry_fn,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_outputs = []

    for i in range(args.n_runs):
        seed = args.base_seed + i
        set_seeds(seed)

        run_dir = root / f"run_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        output_dict = run_once(
            model_name=args.model,
            dataset_name=args.dataset,
            fim_args=fim_args,
            lth_args=lth_args,
            seed=seed,
            run_id=i,
        )
        runtime_s = time.time() - t0

        save_pickle(output_dict, run_dir / "output_dict.pkl")
        if args.save_json:
            (run_dir / "output_dict.json").write_text(
                json.dumps(to_json_safe(output_dict), indent=2),
                encoding="utf-8",
            )

        (run_dir / "meta.json").write_text(
            json.dumps({"run_id": i, "seed": seed, "runtime_s": runtime_s}, indent=2),
            encoding="utf-8",
        )

        all_outputs.append(output_dict)
        print(f"[OK] run {i} seed={seed} time={runtime_s:.2f}s -> {run_dir}")

    if args.save_all_in_one:
        save_pickle(all_outputs, root / "all_runs.pkl")

    print(f"\nDone. Saved results to: {root}")


if __name__ == "__main__":
    main()