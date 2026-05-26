#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


PIPELINE_FAMILIES = {
    "bubble_frst_sam3_mask.py": "hybrid",
    "bubble_sam3_mask.py": "sam3",
    "blackhat_mask.py": "deterministic",
    "classical_test.py": "deterministic",
    "detect_bubbles.py": "deterministic",
    "big_bubble_prompt_fb.py": "prompting",
    "big_bubble_prompt_hf.py": "prompting",
    "frst_point_backend_fb.py": "backend",
    "frst_point_backend_hf.py": "backend",
}


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_sample_input(path: Path) -> bool:
    return (
        path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        and path.parent.name not in {"output", "logs", "__pycache__"}
        and "output" not in path.parts
        and "unit" not in path.parts
    )


def build_registry(root: Path) -> Dict[str, List[Dict[str, str]]]:
    root = root.resolve()
    registry: Dict[str, List[Dict[str, str]]] = {
        "pipeline_entrypoints": [],
        "helper_modules": [],
        "sample_inputs": [],
        "outputs": [],
        "logs": [],
    }

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = _rel(root, path)
        if "__pycache__" in path.parts:
            continue

        if path.suffix == ".py":
            if path.parent == root and path.name in PIPELINE_FAMILIES:
                registry["pipeline_entrypoints"].append(
                    {"name": path.name, "path": rel, "family": PIPELINE_FAMILIES[path.name]}
                )
            elif len(path.parts) >= 3 and path.parts[-3] == "src" and path.name in PIPELINE_FAMILIES:
                registry["pipeline_entrypoints"].append(
                    {"name": path.name, "path": rel, "family": PIPELINE_FAMILIES[path.name]}
                )
            elif path.parent != root and "unit" not in path.parts:
                registry["helper_modules"].append({"path": rel})
            continue

        if _is_sample_input(path):
            registry["sample_inputs"].append({"path": rel})
            continue

        if "output" in path.parts:
            registry["outputs"].append({"path": rel})
            continue

        if path.parent.name == "logs" and path.suffix == ".log":
            registry["logs"].append({"path": rel})

    return registry


def write_registry(root: Path, output_path: Path) -> Dict[str, List[Dict[str, str]]]:
    registry = build_registry(root)
    output_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return registry


def load_experiment_metadata(metadata_path: Path) -> Dict[str, Dict[str, object]]:
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="ascii"))
    entries = payload.get("experiments", [])
    return {entry["path"]: entry for entry in entries if "path" in entry}


def render_markdown(
    registry: Dict[str, List[Dict[str, str]]], metadata: Dict[str, Dict[str, object]] | None = None
) -> str:
    metadata = metadata or {}
    lines = [
        "# Hybrid Experiment Index",
        "",
        "Current inventory of hybrid bubble-identification research artifacts under `bubbly_flows/tests/`.",
        "",
    ]
    for section, title in [
        ("pipeline_entrypoints", "Pipeline Entrypoints"),
        ("helper_modules", "Helper Modules"),
        ("sample_inputs", "Sample Inputs"),
        ("outputs", "Known Outputs"),
        ("logs", "Known Logs"),
    ]:
        lines.append(f"## {title}")
        items = registry.get(section, [])
        if not items:
            lines.append("- none")
        else:
            for item in items:
                label = item.get("name", item.get("path", ""))
                family = f" ({item['family']})" if "family" in item else ""
                path = item.get("path", label)
                lines.append(f"- `{label}`{family}: `{path}`")
                if section == "pipeline_entrypoints" and path in metadata:
                    entry = metadata[path]
                    curated_label = entry.get("label")
                    if curated_label:
                        lines.append(f"  Label: {curated_label}")
                    status = entry.get("status")
                    if status:
                        lines.append(f"  Status: {status}")
                    components = entry.get("components", [])
                    if components:
                        lines.append(f"  Components: {', '.join(str(component) for component in components)}")
                    question = entry.get("question")
                    if question:
                        lines.append(f"  Question: {question}")
                    outputs = entry.get("outputs", [])
                    if outputs:
                        lines.append(f"  Outputs: {', '.join(str(output) for output in outputs)}")
                    notes = entry.get("notes")
                    if notes:
                        lines.append(f"  Notes: {notes}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(
    registry: Dict[str, List[Dict[str, str]]], output_path: Path, metadata: Dict[str, Dict[str, object]] | None = None
) -> None:
    output_path.write_text(render_markdown(registry, metadata), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid experiment inventory for bubbly_flows/tests")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--markdown", dest="markdown_path", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    registry = build_registry(root)
    json_path = args.json_path or (root / "experiment_registry.json")
    markdown_path = args.markdown_path or (root / "EXPERIMENT_INDEX.md")
    metadata = load_experiment_metadata(root / "experiment_metadata.json")
    json_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="ascii")
    write_markdown(registry, markdown_path, metadata)


if __name__ == "__main__":
    main()
