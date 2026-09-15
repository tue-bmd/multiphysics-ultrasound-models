"""Source and configuration fingerprints, and the checkpoint guard.

A checkpoint that survives a change to the model, the data or the configuration
is worse than no checkpoint: it silently mixes two studies.  Every checkpoint
carries the sha256 of the package sources and of the configuration that produced
it, and refuses to resume when either has moved.
"""
from __future__ import annotations
import hashlib
import json
import os
import pickle


def code_hash():
    """sha256 over the package sources, in path order."""
    root = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for f in sorted(filenames):
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            h.update(os.path.relpath(p, root).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def config_hash(obj):
    """sha256 of a configuration, canonically serialized."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def fingerprint(config):
    return dict(code=code_hash(), config=config_hash(config))


class IncompatibleCheckpoint(RuntimeError):
    pass


def save_checkpoint(path, payload, config):
    """Write a checkpoint with its fingerprint."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blob = dict(payload=payload, fingerprint=fingerprint(config))
    with open(path, "wb") as fh:
        pickle.dump(blob, fh)
    return blob["fingerprint"]


def load_checkpoint(path, config, allow_code_change=False):
    """Read a checkpoint, refusing one written by different code or a different
    configuration.

    ``allow_code_change`` exists for the case where a change is known not to
    touch the computation, for instance a docstring.  It must be passed
    explicitly, and what it permits is recorded in the returned fingerprint.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    want, got = fingerprint(config), blob.get("fingerprint", {})
    if got.get("config") != want["config"]:
        raise IncompatibleCheckpoint(
            "%s was written under a different configuration (%s, now %s); "
            "delete it or run with the configuration it belongs to"
            % (path, str(got.get("config"))[:12], want["config"][:12]))
    if got.get("code") != want["code"] and not allow_code_change:
        raise IncompatibleCheckpoint(
            "%s was written by a different version of the package (%s, now %s); "
            "delete it, or pass allow_code_change if the change cannot reach "
            "the computation" % (path, str(got.get("code"))[:12],
                                 want["code"][:12]))
    return blob["payload"]
