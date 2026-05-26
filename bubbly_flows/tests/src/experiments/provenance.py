from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    family: str
    variant: str
    parameters: Dict[str, Any]


def write_run_manifest(
    run_dir: Path,
    spec: ExperimentSpec,
    dataset_name: str,
    dataset_items: List[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": spec.experiment_id,
        "family": spec.family,
        "variant": spec.variant,
        "parameters": spec.parameters,
        "dataset_name": dataset_name,
        "dataset_items": dataset_items,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if extra:
        payload["extra"] = extra

    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
