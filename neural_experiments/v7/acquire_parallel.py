"""Reproducible V7B public acquisition with synchronized transport and raw limits.

Only provider redirects observed in the normal public download flow may be
supplied through --url-file. That private file and all downloaded samples stay in
the ignored cache. This module performs no signal conversion or outcome scoring.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path

from neural_experiments.v7 import fresh_data as data


class SharedBudget:
    """Reserve concurrent reads before I/O; record actual returned bytes afterward."""

    def __init__(self, cache, *, compressed_limit, raw_limit):
        self.cache = Path(cache)
        self.lock = threading.RLock()
        self.last_save = 0.0
        self.compressed_limit = compressed_limit
        self.raw_limit = raw_limit
        attempts = {}
        for p in [
            *(self.cache / "acquisition").glob("*.json"),
            *(self.cache / "failures").glob("*.json"),
        ]:
            r = json.loads(p.read_text())
            if "compressed_prefix_bytes" in r:
                key = (r["participant_id"], r["started_at"])
                attempts[key] = max(attempts.get(key, 0), r["compressed_prefix_bytes"])
        spent = sum(attempts.values())
        # A crash can leave the last transport ledger ahead of its member receipt.
        previous = self.cache / "parallel-acquisition.json"
        if previous.exists():
            old = json.loads(previous.read_text())
            if (
                old.get("status") not in {"complete", "failed"}
                or old.get("compressed_bytes_claimed_pending", 0) != 0
            ):
                # The 10s checkpoint can lag bytes consumed before a hard kill.
                # Never guess the missing transfer or silently reset its budget.
                raise ValueError(
                    "Unclean acquisition stop: exact consumed transfer is unknown; "
                    "automatic resume refused. Preserve receipts for explicit budget review."
                )
            spent = max(
                spent,
                old["compressed_bytes_actual_total"]
                + old.get("compressed_bytes_claimed_pending", 0),
            )
            data._write(
                self.cache / "failures" / f"parallel-{old['started_at'].replace(':', '')}.json", old
            )
        raw = sum(p.stat().st_size for p in (self.cache / "raw").rglob("*") if p.is_file())
        if spent > compressed_limit or raw > raw_limit:
            raise ValueError("Existing cache already exceeds a locked budget")
        self.state = {
            "status": "running",
            "started_at": data.v2._now(),
            "plan_sha256": data.PLAN_SHA256,
            "orchestrator_sha256": data.v2.sha256_file(__file__),
            "importer_sha256": data.v2.sha256_file(data.__file__),
            "maximum_compressed_bytes_total": compressed_limit,
            "maximum_retained_raw_bytes_total": raw_limit,
            "compressed_bytes_before_orchestration": spent,
            "compressed_bytes_actual_total": spent,
            "compressed_bytes_claimed_pending": 0,
            "raw_bytes_before_orchestration": raw,
            "raw_bytes_reserved_total": raw,
            "participants": {},
        }

    def save(self, force=False):
        with self.lock:
            if force or time.monotonic() - self.last_save >= 10:
                self.state["updated_at"] = data.v2._now()
                data._write(self.cache / "parallel-acquisition.json", self.state)
                self.last_save = time.monotonic()

    def start(self, participant):
        with self.lock:
            self.state["participants"][participant] = {
                "status": "running",
                "started_at": data.v2._now(),
                "compressed_bytes_this_attempt": 0,
                "selected_member_bytes_reserved": 0,
            }
            self.save(True)

    def reserve_read(self, size):
        if size < 0:
            raise ValueError("Unbounded transport read prohibited")
        with self.lock:
            remaining = (
                self.compressed_limit
                - self.state["compressed_bytes_actual_total"]
                - self.state["compressed_bytes_claimed_pending"]
            )
            allowed = min(size, remaining)
            if allowed <= 0:
                raise ValueError("Synchronized global compressed budget exhausted")
            self.state["compressed_bytes_claimed_pending"] += allowed
            return allowed

    def finish_read(self, participant, reserved, returned):
        with self.lock:
            self.state["compressed_bytes_claimed_pending"] -= reserved
            self.state["compressed_bytes_actual_total"] += returned
            self.state["participants"][participant]["compressed_bytes_this_attempt"] += returned
            self.save()

    def reserve_member(self, participant, target, size):
        with self.lock:
            temporary = target.with_suffix(".mat.part")
            partial = temporary.stat().st_size if temporary.exists() else 0
            # The existing final file remains on disk while its replacement is
            # written. Only already-counted .part bytes offset temporary growth.
            # Reservations are retained conservatively after atomic replacement.
            added = max(0, size - partial)
            if self.state["raw_bytes_reserved_total"] + added > self.raw_limit:
                raise ValueError("Synchronized retained raw-storage budget exhausted")
            self.state["raw_bytes_reserved_total"] += added
            self.state["participants"][participant]["selected_member_bytes_reserved"] += added
            self.save(True)


class BudgetResponse:
    def __init__(self, response, budget, participant):
        self.response = response
        self.headers = response.headers
        self.budget = budget
        self.participant = participant

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.response.__exit__(*args)

    def read(self, size=-1):
        reserved = self.budget.reserve_read(size)
        try:
            chunk = self.response.read(reserved)
        except BaseException:
            # A transport failure may have consumed unreturned bytes. Charge the
            # entire reservation conservatively and preserve this failed attempt.
            self.budget.finish_read(self.participant, reserved, reserved)
            raise
        self.budget.finish_read(self.participant, reserved, len(chunk))
        return chunk


def acquire_parallel(cache, v6_development_index, v6_new_index, *, url_file=None, workers=3):
    if type(workers) is not int or not 1 <= workers <= 3:
        raise ValueError("V7B permits one to three acquisition workers")
    cache = Path(cache)
    data._study_lock(Path(data.__file__).with_name("acquisition-plan.json"), data.PLAN_SHA256)
    # Authenticate source parents and pin/write provider metadata exactly once.
    old = data._read_parents(v6_development_index, v6_new_index)
    manifest = data.v2.pin_metadata(cache)
    complete = set()
    for participant in data.PARTICIPANTS:
        path = cache / "acquisition" / f"{participant}.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt["status"] == "complete":
                data.validate_receipt(receipt, participant, data._prior_members(old, participant))
                for s in receipt["selected"]:
                    if (
                        data.v2.sha256_file(data.v2._safe_path(cache, s["raw_path"]))
                        != s["member_sha256"]
                    ):
                        raise ValueError("Cached raw identity changed before resume")
                complete.add(participant)
    budget = SharedBudget(
        cache, compressed_limit=data.MAX_COMPRESSED_BYTES, raw_limit=data.MAX_CACHE_BYTES
    )
    budget.state["previously_complete_participants"] = sorted(complete)
    budget.state["maximum_parallel_workers"] = workers
    budget.save(True)

    def run(participant):
        budget.start(participant)

        def opener(*args, **kwargs):
            return BudgetResponse(data.urlopen(*args, **kwargs), budget, participant)

        try:
            data.acquire(
                cache,
                v6_development_index,
                v6_new_index,
                participants=(participant,),
                url_file=url_file,
                manifest=manifest,
                opener=opener,
                reserve_member=lambda target, size: budget.reserve_member(
                    participant, target, size
                ),
            )
            with budget.lock:
                budget.state["participants"][participant]["status"] = "complete"
        except Exception as error:
            with budget.lock:
                budget.state["participants"][participant].update(
                    status="failed", error_type=type(error).__name__
                )
            # Detailed failure receipts remain private; do not print transport URLs.
            print(
                json.dumps({"participant": participant, "failed": type(error).__name__}), flush=True
            )
            raise
        finally:
            with budget.lock:
                budget.state["participants"][participant]["finished_at"] = data.v2._now()
                budget.save(True)

    failures = []
    pending = [p for p in data.PARTICIPANTS if p not in complete]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(run, p): p for p in pending}
        for job in concurrent.futures.as_completed(jobs):
            try:
                job.result()
            except Exception:
                failures.append(jobs[job])
    with budget.lock:
        budget.state.update(
            status="failed" if failures else "complete",
            failed_participants=sorted(failures),
            finished_at=data.v2._now(),
            raw_bytes_actual_final=sum(
                p.stat().st_size for p in (cache / "raw").rglob("*") if p.is_file()
            ),
        )
        if budget.state["raw_bytes_actual_final"] > data.MAX_CACHE_BYTES:
            raise AssertionError("Retained raw budget exceeded")
        budget.save(True)
    print(
        json.dumps(
            {
                k: budget.state[k]
                for k in (
                    "status",
                    "compressed_bytes_actual_total",
                    "raw_bytes_actual_final",
                    "failed_participants",
                )
            }
        ),
        flush=True,
    )
    if failures:
        raise SystemExit(1)
    return budget.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--v6-development-index", type=Path, default=Path("data/neural_v6/development/index.json")
    )
    parser.add_argument(
        "--v6-new-index", type=Path, default=Path("data/neural_v6/new_sessions/index.json")
    )
    parser.add_argument("--url-file", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    acquire_parallel(
        args.cache,
        args.v6_development_index,
        args.v6_new_index,
        url_file=args.url_file,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
