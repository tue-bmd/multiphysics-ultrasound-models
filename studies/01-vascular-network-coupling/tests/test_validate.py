"""Archive validation: does it catch the ways an archive goes wrong?"""
import json
import os
import shutil

import pytest

from porovasc import validate as VAL
from porovasc.study import SCHEMA, code_hash


def _archive(tmp_path, n_seeds=3, arms=("baseline", "tort_1.45"), complete=True):
    d = tmp_path / "run"
    d.mkdir()
    seeds = [100 + i for i in range(n_seeds)]
    recs = [dict(arm=a, condition=a, seed=s, rve=0, payload=1.0)
            for a in arms for s in seeds]
    with open(d / "records.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    man = dict(schema=SCHEMA, kind="coupling", seeds=seeds,
               expected_records=len(recs), records=len(recs),
               code_sha256=code_hash(), finished="2026-09-13T00:00:00",
               records_sha256=VAL.file_sha256(str(d / "records.jsonl")))
    json.dump(man, open(d / "manifest.json", "w"))
    if complete:
        open(d / "COMPLETE", "w").write("done\n")
    return d


def test_a_sound_archive_validates(tmp_path):
    rep = VAL.check(str(_archive(tmp_path)))
    assert rep["ok"], rep["failures"]
    assert rep["info"]["valid"] == 6
    assert rep["info"]["seeds"] == [100, 101, 102]
    assert not rep["warnings"]


def test_a_missing_completion_marker_fails(tmp_path):
    rep = VAL.check(str(_archive(tmp_path, complete=False)))
    assert not rep["ok"]
    assert any("COMPLETE" in f for f in rep["failures"])


def test_a_truncated_final_line_is_caught(tmp_path):
    d = _archive(tmp_path)
    with open(d / "records.jsonl") as fh:
        text = fh.read()
    with open(d / "records.jsonl", "w") as fh:          # drop the final newline
        fh.write(text.rstrip("\n"))
    rep = VAL.check(str(d))
    assert not rep["ok"]
    assert any("truncated" in f for f in rep["failures"])


def test_a_half_written_final_record_is_caught(tmp_path):
    d = _archive(tmp_path)
    with open(d / "records.jsonl", "a") as fh:
        fh.write('{"arm": "tort_1.45", "seed": 103, "rve": 0, "pay')
    rep = VAL.check(str(d))
    assert not rep["ok"]
    assert any("unparsable" in f for f in rep["failures"])
    assert rep["info"]["unparsable"] == 1


def test_a_duplicate_record_is_caught(tmp_path):
    d = _archive(tmp_path)
    with open(d / "records.jsonl") as fh:
        first = fh.readline()
    with open(d / "records.jsonl", "a") as fh:
        fh.write(first)
    rep = VAL.check(str(d))
    assert not rep["ok"]
    assert any("duplicate" in f for f in rep["failures"])


def test_a_missing_seed_in_one_arm_is_caught(tmp_path):
    d = _archive(tmp_path)
    lines = open(d / "records.jsonl").read().splitlines()
    keep = [l for l in lines if not (json.loads(l)["arm"] == "tort_1.45"
                                     and json.loads(l)["seed"] == 102)]
    open(d / "records.jsonl", "w").write("\n".join(keep) + "\n")
    rep = VAL.check(str(d))
    assert not rep["ok"]
    assert any("missing seeds" in f for f in rep["failures"])


def test_a_record_count_that_disagrees_with_the_manifest_is_caught(tmp_path):
    d = _archive(tmp_path)
    man = json.load(open(d / "manifest.json"))
    man["records"] = 30
    json.dump(man, open(d / "manifest.json", "w"))
    rep = VAL.check(str(d))
    assert not rep["ok"]
    assert any("manifest reports 30" in f for f in rep["failures"])


def test_a_changed_record_file_breaks_the_checksum(tmp_path):
    d = _archive(tmp_path)
    with open(d / "records.jsonl", "a") as fh:
        fh.write(json.dumps(dict(arm="baseline", condition="baseline",
                                 seed=100, rve=1, payload=2.0)) + "\n")
    rep = VAL.check(str(d))
    assert not rep["ok"]
    assert any("records_sha256 mismatch" in f for f in rep["failures"])


def test_a_different_source_version_is_a_warning_not_a_failure(tmp_path):
    d = _archive(tmp_path)
    man = json.load(open(d / "manifest.json"))
    man["code_sha256"] = "0" * 64
    json.dump(man, open(d / "manifest.json", "w"))
    rep = VAL.check(str(d))
    assert rep["ok"], rep["failures"]
    assert any("different source version" in w for w in rep["warnings"])


def test_an_absent_checksum_is_a_warning_and_can_be_pinned(tmp_path):
    d = _archive(tmp_path)
    man = json.load(open(d / "manifest.json"))
    digest = man.pop("records_sha256")
    json.dump(man, open(d / "manifest.json", "w"))
    rep = VAL.check(str(d))
    assert rep["ok"]
    assert any("no records_sha256" in w for w in rep["warnings"])
    VAL.write_checksum(str(d))
    assert json.load(open(d / "manifest.json"))["records_sha256"] == digest
    assert VAL.check(str(d))["ok"]
    with pytest.raises(SystemExit):
        VAL.write_checksum(str(d))


def test_require_stops_on_a_bad_archive_and_partial_lets_it_through(tmp_path, capsys):
    d = _archive(tmp_path, complete=False)
    with pytest.raises(SystemExit):
        VAL.require(str(d))
    rep = VAL.require(str(d), partial=True)
    assert not rep["ok"]


def test_the_paired_reader_refuses_a_bad_archive(tmp_path):
    from porovasc import analyse_paired as AP
    d = _archive(tmp_path, complete=False)
    with pytest.raises(SystemExit):
        AP.load(str(d))
    man, recs, errors = AP.load(str(d), partial=True)
    assert len(recs) == 6
