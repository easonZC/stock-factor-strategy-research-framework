"""Train and persist a model-based factor on synthetic data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

from factorlab.models import train_model_factor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train model factor using synthetic panel",
        epilog=(
            "Examples:\n"
            "  python apps/train_model_factor.py --model ridge --out artifacts/models/ridge_model_factor.joblib\n"
            "  python apps/train_model_factor.py --model rf --out artifacts/models/rf_model_factor.joblib\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", default="ridge", choices=["ridge", "rf"])
    parser.add_argument("--out", default="artifacts/models/model_factor.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = train_model_factor(model_name=args.model, model_out=args.out)
    print(path)


if __name__ == "__main__":
    main()
