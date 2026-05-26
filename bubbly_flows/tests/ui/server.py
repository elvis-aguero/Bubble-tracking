from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from bubbly_flows.tests.src.experiments.one_image_tuning import HYBRID_BASELINE_PARAMETERS
from bubbly_flows.tests.ui.render import render_gold_overlay, summarize_labelme
from bubbly_flows.tests.ui.state import HybridUIState


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_IMAGE = REPO_ROOT / "bubbly_flows" / "data" / "frames" / "images_16bit_png" / "ZeroG_FlightDay_Test_C1S0014_img006001.png"
GOLD_MASK = REPO_ROOT / "bubbly_flows" / "tests" / "output" / "eval_sets" / "gold_seed_v04" / "labels" / "ZeroG_FlightDay_Test_C1S0014_img006001.tif"
UI_ROOT = REPO_ROOT / "bubbly_flows" / "tests" / "output" / "ui"
STATIC_ROOT = Path(__file__).resolve().parent / "static"


class HybridUIController:
    def __init__(self, output_root: Path | None = None, debounce_seconds: float = 5.0) -> None:
        self.output_root = output_root or UI_ROOT
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.state = HybridUIState(baseline_parameters=HYBRID_BASELINE_PARAMETERS, debounce_seconds=debounce_seconds)
        self.current_dir = self.output_root / "current"
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.gold_overlay_path = self.output_root / "gold_overlay.png"
        render_gold_overlay(SOURCE_IMAGE, GOLD_MASK, self.gold_overlay_path)
        self.current_overlay_path = self.output_root / "placeholder.png"
        if not self.current_overlay_path.exists():
            self.current_overlay_path.write_bytes(self.gold_overlay_path.read_bytes())
        self.revision = int(time.time())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self) -> None:
        self.state.reset_to_baseline(now=time.time())
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._worker.join(timeout=1.0)

    def update_params(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            self.state.update_params(updates, now=time.time())
            self.revision = int(time.time() * 1000)

    def reset(self) -> None:
        with self._lock:
            self.state.reset_to_baseline(now=time.time())
            self.revision = int(time.time() * 1000)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            payload = self.state.snapshot()
            payload["revision"] = self.revision
            return payload

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                params = self.state.maybe_begin_run(now=time.time())
            if params is None:
                time.sleep(0.25)
                continue
            summary: Dict[str, Any] = {}
            error: str | None = None
            try:
                summary = self._run_hybrid(params)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            with self._lock:
                self.state.finish_run(summary=summary, now=time.time(), error=error)
                self.revision = int(time.time() * 1000)

    def _run_hybrid(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.current_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = self.current_dir / f"{SOURCE_IMAGE.stem}_overlay.png"
        labelme_path = self.current_dir / f"{SOURCE_IMAGE.stem}.json"
        command = [
            sys.executable,
            "-m",
            "bubbly_flows.tests.src.hybrid.bubble_frst_sam3_mask",
            "--input",
            str(SOURCE_IMAGE),
            "--output",
            str(overlay_path),
            "--output_json",
            str(labelme_path),
            "--no_analysis_json",
            "--big_backend",
            "hf",
        ]
        for key, value in params.items():
            flag = f"--{key}"
            if isinstance(value, bool):
                if value:
                    command.append(flag)
                continue
            command.extend([flag, str(value)])
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        self.current_overlay_path = overlay_path
        return summarize_labelme(labelme_path)


class HybridUIRequestHandler(BaseHTTPRequestHandler):
    server_version = "HybridUI/0.1"

    @property
    def controller(self) -> HybridUIController:
        return self.server.controller  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._serve_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._send_json(self.controller.snapshot())
            return
        if path == "/api/image/gold":
            self._serve_file(self.controller.gold_overlay_path)
            return
        if path == "/api/image/current":
            self._serve_file(self.controller.current_overlay_path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/params":
            payload = self._read_json()
            self.controller.update_params(payload)
            self._send_json({"ok": True})
            return
        if path == "/api/reset":
            self.controller.reset()
            self._send_json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _serve_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        mime = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def make_server(host: str = "127.0.0.1", port: int = 7860) -> ThreadingHTTPServer:
    controller = HybridUIController()
    controller.start()
    server = ThreadingHTTPServer((host, port), HybridUIRequestHandler)
    server.controller = controller  # type: ignore[attr-defined]
    return server


def serve(host: str = "127.0.0.1", port: int = 7860) -> None:
    server = make_server(host=host, port=port)
    try:
        server.serve_forever()
    finally:
        server.controller.stop()  # type: ignore[attr-defined]
        server.server_close()
