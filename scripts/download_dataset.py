from __future__ import annotations

import sys
from pathlib import Path

try:
    from src.data_pipeline import prepare_dataset
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.data_pipeline import prepare_dataset


if __name__ == "__main__":
    df = prepare_dataset()
    print(f"Dataset prepared successfully. Shape: {df.shape}")
    print(df.head().to_string(index=False))
