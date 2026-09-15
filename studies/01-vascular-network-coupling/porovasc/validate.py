"""Does an archived run still hold what its manifest says it holds?

    python -m porovasc.validate runs/baseline
    python -m porovasc.validate runs/*                 # every arm of a study
    python -m porovasc.validate runs/baseline --json

Every analysis in this package now calls `check` before it reads anything, and
refuses to proceed on a failure unless `--partial` is given.  The reason is
concrete: a summary can outlive the records it was computed from.  A truncated
final line, a directory copied while a run was still writing, a manifest from
one attempt beside the records of another, an arm silently short of its seeds -
none of these announce themselves, and all of them produce a summary that looks
ordinary.

What is checked:

  markers      COMPLETE or FAILED, and whether they agree with the record count
  schema       the manifest schema against the one this code reads
  parse        every line of records.jsonl, counting valid, unparsable and the
               truncated final line that an interrupted write leaves behind
  count        records found against manifest expected_records and records
  duplicates   the same (arm, seed, rve, condition) written twice
  seeds        the seeds present against the seeds the manifest declares
  conditions   the arms or conditions present against those declared
  checksum     sha256 of records.jsonl against the value the manifest records,
               when the run was written by a version that stores it; when it is
               absent the computed digest is reported so that it can be pinned
  code         the manifest's code_sha256 against the code that is running now

A code-hash mismatch is reported as a warning, not a failure: it does not make
the numbers wrong, it makes their provenance unverifiable, and that distinction
belongs to the reader.  Everything else is a failure.
"""
from __future__ import annotations
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys

from .study import SCHEMA, code_hash

RECORD_KEY = ("arm", "seed", "rve", "condition")


def _records_path(folder):
    plain = os.path.join(folder, "records.jsonl")
    gz = plain + ".gz"
    if os.path.exists(plain) and os.path.exists(gz):
        return None, "both records.jsonl and records.jsonl.gz are present"
    if os.path.exists(plain):
        return plain, None
    if os.path.exists(gz):
        return gz, None
    return None, "no records.jsonl"


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def scan_records(path):
    """Parse line by line, so that a bad line is located instead of aborting.

    A file that does not end in a newline has an incomplete final line: that is
    what an interrupted write leaves, and it is reported separately from a line
    that is merely malformed."""
    op = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    recs, bad, blank = [], [], 0
    last_line = ""
    with op as fh:
        for i, line in enumerate(fh, 1):
            last_line = line
            s = line.strip()
            if not s:
                blank += 1
                continue
            try:
                recs.append(json.loads(s))
            except Exception as e:
                bad.append(dict(line=i, chars=len(s), reason=str(e)[:80]))
    truncated = bool(last_line) and not last_line.endswith("\n")
    return recs, bad, blank, truncated


def check(folder, expect_seeds=None, expect_conditions=None):
    """Full report for one run directory.  Never raises on a bad archive; the
    caller decides what to do with `ok`."""
    rep = dict(folder=folder, ok=True, failures=[], warnings=[], info={})

    def fail(msg):
        rep["ok"] = False
        rep["failures"].append(msg)

    def warn(msg):
        rep["warnings"].append(msg)

    man_path = os.path.join(folder, "manifest.json")
    if not os.path.exists(man_path):
        fail("no manifest.json")
        return rep
    try:
        man = json.load(open(man_path))
    except Exception as e:
        fail("manifest.json is not readable: %s" % str(e)[:80])
        return rep
    rep["manifest"] = man

    if man.get("schema") != SCHEMA:
        fail("schema %s, this code reads %d" % (man.get("schema"), SCHEMA))

    complete = os.path.exists(os.path.join(folder, "COMPLETE"))
    failed = os.path.exists(os.path.join(folder, "FAILED"))
    rep["info"]["complete"] = complete
    rep["info"]["failed_marker"] = failed
    if failed:
        fail("the run is marked FAILED: %s"
             % open(os.path.join(folder, "FAILED")).read().strip())
    elif not complete:
        fail("no COMPLETE marker: the run did not finish, or was copied while running")

    path, err = _records_path(folder)
    if err:
        fail(err)
        return rep
    rep["info"]["records_file"] = os.path.basename(path)
    rep["info"]["records_bytes"] = os.path.getsize(path)

    recs, bad, blank, truncated = scan_records(path)
    good = [r for r in recs if "error" not in r]
    errored = [r for r in recs if "error" in r]
    rep["info"].update(parsed=len(recs), valid=len(good), errored=len(errored),
                       unparsable=len(bad), blank_lines=blank,
                       truncated_final_line=truncated)
    if bad:
        fail("%d unparsable line(s): %s"
             % (len(bad), ", ".join("line %d (%d chars)" % (b["line"], b["chars"])
                                    for b in bad[:5])))
    if truncated:
        fail("the last line has no terminating newline: the file is truncated")
    if errored:
        warn("%d record(s) carry an error field and are excluded from analysis"
             % len(errored))

    # ---- counts ----------------------------------------------------------
    n_man = man.get("records")
    n_exp = man.get("expected_records")
    if n_man is not None and len(recs) != n_man:
        fail("manifest reports %d records, the file holds %d" % (n_man, len(recs)))
    if n_exp is not None and len(good) != n_exp:
        (fail if complete else warn)(
            "manifest expected %d records, %d valid record(s) present" % (n_exp, len(good)))

    # ---- duplicates ------------------------------------------------------
    seen, dupes = set(), []
    for r in good:
        key = tuple(r.get(k) for k in RECORD_KEY)
        if key in seen:
            dupes.append(key)
        seen.add(key)
    if dupes:
        fail("%d duplicate record key(s): %s"
             % (len(dupes), ", ".join(str(d) for d in dupes[:4])))

    # ---- seeds and conditions -------------------------------------------
    seeds = sorted({r["seed"] for r in good if "seed" in r})
    conds = sorted({r.get("arm") or r.get("condition") for r in good
                    if r.get("arm") or r.get("condition")})
    rep["info"]["seeds"] = seeds
    rep["info"]["conditions"] = conds
    want_seeds = expect_seeds if expect_seeds is not None else man.get("seeds")
    if want_seeds is not None:
        missing = sorted(set(want_seeds) - set(seeds))
        extra = sorted(set(seeds) - set(want_seeds))
        if missing:
            fail("seeds declared but absent: %s" % missing)
        if extra:
            fail("seeds present but not declared: %s" % extra)
    if expect_conditions is not None:
        missing = sorted(set(expect_conditions) - set(conds))
        if missing:
            fail("conditions declared but absent: %s" % missing)

    # per (condition, seed) completeness, which is what a paired design needs
    if conds:
        grid = {}
        for r in good:
            c = r.get("arm") or r.get("condition")
            grid.setdefault(c, set()).add(r.get("seed"))
        short = {c: sorted(set(seeds) - s) for c, s in grid.items()
                 if set(seeds) - s}
        if short:
            fail("condition(s) missing seeds: %s"
                 % "; ".join("%s missing %s" % (c, s) for c, s in short.items()))
        rep["info"]["records_per_condition"] = {c: len(s) for c, s in grid.items()}

    # ---- checksums -------------------------------------------------------
    digest = file_sha256(path)
    rep["info"]["records_sha256"] = digest
    stored = man.get("records_sha256")
    if stored is None:
        warn("the manifest carries no records_sha256; computed %s (pin it with "
             "--write-checksum)" % digest[:16])
    elif stored != digest:
        fail("records_sha256 mismatch: manifest %s, file %s"
             % (stored[:16], digest[:16]))

    # ---- provenance of the code -----------------------------------------
    now = code_hash()
    rep["info"]["code_sha256_manifest"] = man.get("code_sha256")
    rep["info"]["code_sha256_now"] = now
    if man.get("code_sha256") and man["code_sha256"] != now:
        warn("written by a different source version (manifest %s, current %s): "
             "the numbers are not invalidated, but their provenance cannot be "
             "verified against this checkout"
             % (man["code_sha256"][:12], now[:12]))
    return rep


def write_checksum(folder):
    """Pin the digest of an existing archive into its manifest, once, so that
    later reads can detect corruption.  Refuses to overwrite a stored value."""
    man_path = os.path.join(folder, "manifest.json")
    man = json.load(open(man_path))
    if man.get("records_sha256"):
        raise SystemExit("%s already carries records_sha256" % folder)
    path, err = _records_path(folder)
    if err:
        raise SystemExit(err)
    man["records_sha256"] = file_sha256(path)
    man["records_bytes"] = os.path.getsize(path)
    man["checksum_added"] = "after the fact, by porovasc.validate"
    tmp = man_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(man, fh, indent=1)
    os.replace(tmp, man_path)
    print("pinned %s in %s" % (man["records_sha256"][:16], folder))


def require(folder, partial=False, **kw):
    """Validate, print, and stop unless the archive is sound or `partial`."""
    rep = check(folder, **kw)
    report(rep)
    if not rep["ok"] and not partial:
        raise SystemExit(
            "%s failed validation; fix the archive, or pass --partial to read it "
            "anyway and mark every number derived from it as provisional" % folder)
    return rep


def report(rep, stream=sys.stdout):
    i = rep.get("info", {})
    head = "OK  " if rep["ok"] else "FAIL"
    print("%s %s" % (head, rep["folder"]), file=stream)
    if "valid" in i:
        print("     %d valid records, %d errored, %d unparsable, seeds %s"
              % (i["valid"], i["errored"], i["unparsable"], i.get("seeds")),
              file=stream)
        if i.get("records_per_condition"):
            print("     per condition: %s" % i["records_per_condition"], file=stream)
    for f in rep["failures"]:
        print("     FAILURE  %s" % f, file=stream)
    for w in rep["warnings"]:
        print("     warning  %s" % w, file=stream)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    ap.add_argument("--write-checksum", action="store_true",
                    help="pin the current digest into the manifest of a sound archive")
    a = ap.parse_args(argv)
    folders = []
    for f in a.folders:
        folders += sorted(glob.glob(f)) if any(c in f for c in "*?[") else [f]
    folders = [f for f in folders if os.path.isdir(f)]
    reps = [check(f) for f in folders]
    if a.write_checksum:
        for f, r in zip(folders, reps):
            if r["ok"]:
                write_checksum(f)
            else:
                print("skipping %s: it does not validate" % f)
        return 0
    if a.json:
        json.dump([{k: v for k, v in r.items() if k != "manifest"} for r in reps],
                  sys.stdout, indent=2)
        print()
    else:
        for r in reps:
            report(r)
        n_bad = sum(1 for r in reps if not r["ok"])
        print("\n%d of %d archive(s) validate" % (len(reps) - n_bad, len(reps)))
    return 1 if any(not r["ok"] for r in reps) else 0


if __name__ == "__main__":
    sys.exit(main())
