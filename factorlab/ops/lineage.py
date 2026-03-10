"""数据血缘与实验注册工具。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def stable_hash(payload: Any, length: int = 16) -> str:
    """生成稳定短哈希。"""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[: max(8, int(length))]


def describe_panel_lineage(
    *,
    panel: pd.DataFrame,
    data_cfg: dict[str, Any],
    load_report: dict[str, Any],
    mode_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """描述输入数据血缘与简要指纹。"""
    dates = pd.to_datetime(panel["date"], errors="coerce") if "date" in panel.columns else pd.Series(dtype="datetime64[ns]")
    source = {
        "adapter": data_cfg.get("adapter"),
        "mode": data_cfg.get("mode"),
        "path": str(Path(str(data_cfg["path"])).expanduser().resolve()) if data_cfg.get("path") else None,
        "raw_pattern": data_cfg.get("raw_pattern"),
        "synthetic": data_cfg.get("synthetic"),
    }
    profile = {
        "rows": int(len(panel)),
        "columns": sorted(str(col) for col in panel.columns),
        "assets": int(panel["asset"].nunique()) if "asset" in panel.columns else 0,
        "dates": int(dates.nunique()) if not dates.empty else 0,
        "date_min": str(dates.min()) if not dates.empty else None,
        "date_max": str(dates.max()) if not dates.empty else None,
    }
    payload = {
        "source": source,
        "panel_profile": profile,
        "load_profile": load_report.get("panel_profile", {}),
        "mode_report": mode_report or {},
    }
    payload["fingerprint"] = stable_hash(payload)
    return payload


def write_json_artifact(path: str | Path, payload: Any) -> Path:
    """写出 JSON 产物。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def build_experiment_registry(
    *,
    out_dir: str | Path,
    config_hash: str,
    data_lineage: dict[str, Any],
    runtime_manifest: dict[str, Any],
    scope: dict[str, Any],
    governance: dict[str, Any],
    factors: dict[str, Any],
    strategy: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    """构建运行级实验注册信息。"""
    root = Path(out_dir)
    experiment_id = f"{root.name}_{config_hash[:8]}"
    payload = {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "out_dir": str(root.resolve()),
        "config_hash": config_hash,
        "data_fingerprint": data_lineage.get("fingerprint"),
        "runtime": runtime_manifest.get("git", {}),
        "scope": scope,
        "governance": {
            "stop_after": governance.get("stop_after"),
            "research_profile": governance.get("research_profile"),
            "config_governance_mode": governance.get("config_governance_mode"),
            "leakage_guard_mode": governance.get("leakage_guard_mode"),
        },
        "factors": factors,
        "strategy": strategy,
        "outputs": outputs,
    }
    payload["fingerprint"] = stable_hash(payload)
    return payload
