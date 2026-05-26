from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class HybridUIState:
    baseline_parameters: Dict[str, Any]
    debounce_seconds: float = 5.0
    current_params: Dict[str, Any] = field(init=False)
    status: str = field(default="idle", init=False)
    dirty_since: float | None = field(default=None, init=False)
    debounce_deadline: float | None = field(default=None, init=False)
    running_params: Dict[str, Any] | None = field(default=None, init=False)
    pending_rerun: bool = field(default=False, init=False)
    last_summary: Dict[str, Any] = field(default_factory=dict, init=False)
    last_error: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.current_params = copy.deepcopy(self.baseline_parameters)

    def update_params(self, updates: Dict[str, Any], now: float) -> None:
        changed = False
        for key, value in updates.items():
            if self.current_params.get(key) != value:
                self.current_params[key] = value
                changed = True
        if not changed:
            return
        self.last_error = None
        self.dirty_since = now
        self.debounce_deadline = now + self.debounce_seconds
        if self.status == "running":
            self.pending_rerun = True
        else:
            self.status = "waiting"

    def reset_to_baseline(self, now: float) -> None:
        self.current_params = copy.deepcopy(self.baseline_parameters)
        self.dirty_since = now
        self.debounce_deadline = now + self.debounce_seconds
        self.last_error = None
        if self.status == "running":
            self.pending_rerun = True
        else:
            self.status = "waiting"

    def maybe_begin_run(self, now: float) -> Dict[str, Any] | None:
        if self.status == "running":
            return None
        if self.debounce_deadline is None or now < self.debounce_deadline:
            return None
        self.status = "running"
        self.pending_rerun = False
        self.running_params = copy.deepcopy(self.current_params)
        return copy.deepcopy(self.running_params)

    def finish_run(self, summary: Dict[str, Any] | None, now: float, error: str | None = None) -> None:
        self.running_params = None
        self.last_summary = summary or {}
        self.last_error = error
        if self.pending_rerun:
            self.status = "waiting"
            self.debounce_deadline = now
        else:
            self.status = "error" if error else "idle"
            self.debounce_deadline = None
            self.dirty_since = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "params": copy.deepcopy(self.current_params),
            "summary": copy.deepcopy(self.last_summary),
            "last_error": self.last_error,
            "pending_rerun": self.pending_rerun,
        }
