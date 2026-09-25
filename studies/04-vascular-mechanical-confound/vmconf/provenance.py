"""Reproducibility metadata for model runs.

Each result records source hashes, dependency identity, network fingerprints,
network parameters, command-line arguments, and run settings. Absolute local
paths are deliberately excluded from public result files.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone

import numpy as np


def _hash_tree(root, suffix=".py"):
    h = hashlib.sha256()
    for dp, _, names in sorted(os.walk(root)):
        for nm in sorted(names):
            if nm.endswith(suffix):
                path = os.path.join(dp, nm)
                rel = os.path.relpath(path, root).replace(os.sep, "/")
                h.update(rel.encode("utf-8"))
                h.update(b"\0")
                with open(path, "rb") as f:
                    h.update(f.read())
                h.update(b"\0")
    return h.hexdigest()


def code_hashes():
    here = os.path.dirname(os.path.abspath(__file__))
    import porovasc

    porovasc_root = os.path.dirname(os.path.abspath(porovasc.__file__))
    return {
        "vmconf_sha256": _hash_tree(here),
        "porovasc_package": "porovasc",
        "porovasc_version": getattr(porovasc, "__version__", None),
        "porovasc_sha256": _hash_tree(porovasc_root),
    }


def network_fingerprints(net):
    from porovasc.study import fingerprints

    topo, path = fingerprints(net)
    return dict(topology=topo, path=path)


def record(net=None, **settings):
    d = dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
             python=platform.python_version(), numpy=np.__version__,
             argv=_jsonable(list(sys.argv)), **code_hashes())
    if net is not None:
        d["network"] = dict(n_segments=int(net.n),
                            params=_jsonable(getattr(net, "params", None)),
                            **network_fingerprints(net))
    d["settings"] = _jsonable(settings)
    return d


def _jsonable(o):
    if is_dataclass(o) and not isinstance(o, type):
        return _jsonable(asdict(o))
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, str) and os.path.isabs(o):
        return os.path.basename(o)
    return o


def save(path, provenance, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            dict(provenance=provenance, **_jsonable(payload)),
            f,
            indent=1,
            allow_nan=False,
        )
    return path
