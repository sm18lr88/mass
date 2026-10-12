"""Download the pinned public base checkpoints into models/; does not load a model."""

import json
from pathlib import Path


def main() -> None:
    from huggingface_hub import snapshot_download

    root = Path(__file__).resolve().parents[1]
    models = json.loads((root / "configs/models.json").read_text(encoding="utf-8"))
    for name, spec in models.items():
        snapshot_download(**spec, local_dir=root / "models" / name)


if __name__ == "__main__":
    main()
