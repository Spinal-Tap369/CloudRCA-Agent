from __future__ import annotations

import argparse
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "ibm-research/ITBench-Lite"
REPO_TYPE = "dataset"
DEFAULT_SRE_VERSION = "v0.2-B96DF826-4BB2-4B62-97AB-6D84254C53D7"


def parse_scenario_number(name: str) -> int | None:
    match = re.search(r"Scenario-(\d+)", name)
    return int(match.group(1)) if match else None


def parse_scenario_filter(value: str | None) -> set[int] | None:
    if not value or value.lower() in {"all", "*"}:
        return None

    result: set[int] = set()

    for part in value.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            result.update(range(start, end + 1))
        else:
            result.add(int(part))

    return result


def scenario_local_dir(local_dir: Path, sre_version: str, scenario_id: int) -> Path:
    return (
        local_dir
        / "snapshots"
        / "sre"
        / sre_version
        / f"Scenario-{scenario_id}"
    )


def count_local_files(scenario_dir: Path) -> int:
    if not scenario_dir.exists():
        return 0

    return sum(1 for p in scenario_dir.rglob("*") if p.is_file())


def has_local_ground_truth(scenario_dir: Path) -> bool:
    if not scenario_dir.exists():
        return False

    return any(
        p.name.lower() in {"ground_truth.yaml", "ground_truth.yml", "ground_truth.json"}
        for p in scenario_dir.rglob("*")
        if p.is_file()
    )


def scenario_is_complete(
    local_dir: Path,
    sre_version: str,
    scenario_id: int,
    expected_files: list[str],
    min_file_ratio: float,
) -> tuple[bool, str]:
    sdir = scenario_local_dir(local_dir, sre_version, scenario_id)
    local_count = count_local_files(sdir)
    expected_count = len(expected_files)
    has_gt = has_local_ground_truth(sdir)

    if expected_count == 0:
        return False, "not found in remote listing"

    required_count = max(1, int(expected_count * min_file_ratio))

    if not sdir.exists():
        return False, "missing local folder"

    if not has_gt:
        return False, "missing ground_truth file"

    if local_count < required_count:
        return False, f"incomplete file count: local={local_count}, expected~={expected_count}"

    return True, f"complete: local={local_count}, expected={expected_count}"


def list_remote_sre_scenarios(
    sre_version: str,
    scenario_filter: set[int] | None,
) -> dict[int, list[str]]:
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

    prefix = f"snapshots/sre/{sre_version}/Scenario-"
    by_scenario: dict[int, list[str]] = defaultdict(list)

    for path in files:
        if not path.startswith(prefix):
            continue

        scenario_id = parse_scenario_number(path)

        if scenario_id is None:
            continue

        if scenario_filter is not None and scenario_id not in scenario_filter:
            continue

        by_scenario[scenario_id].append(path)

    return dict(sorted(by_scenario.items()))


def download_one_scenario(
    local_dir: Path,
    sre_version: str,
    scenario_id: int,
    retries: int,
    min_delay: float,
    max_delay: float,
    max_workers: int,
) -> bool:
    allow_pattern = f"snapshots/sre/{sre_version}/Scenario-{scenario_id}/**"

    for attempt in range(1, retries + 1):
        try:
            print()
            print(f"Scenario-{scenario_id}: download attempt {attempt}/{retries}")
            print(f"Pattern: {allow_pattern}")

            snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                allow_patterns=[allow_pattern],
                ignore_patterns=[
                    "**/.DS_Store",
                    "**/__MACOSX/**",
                    "**/.git/**",
                ],
                local_dir=str(local_dir),
                max_workers=max_workers,
            )

            print(f"Scenario-{scenario_id}: download finished")
            return True

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            message = str(exc)
            print(f"Scenario-{scenario_id}: download failed: {message[:500]}")

            wait_seconds = min((2 ** attempt) * 15, 300)

            if "429" in message or "rate" in message.lower() or "too many requests" in message.lower():
                wait_seconds = max(wait_seconds, 180)

            wait_seconds += random.uniform(min_delay, max_delay)

            if attempt < retries:
                print(f"Scenario-{scenario_id}: waiting {wait_seconds:.1f}s before retry")
                time.sleep(wait_seconds)

    print(f"Scenario-{scenario_id}: FAILED after {retries} attempts")
    return False


def polite_sleep(min_delay: float, max_delay: float) -> None:
    delay = random.uniform(min_delay, max_delay)

    if delay > 0:
        print(f"Sleeping {delay:.1f}s before next scenario")
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-dir",
        default="data/ITBench-Lite",
        help="Local ITBench-Lite folder",
    )
    parser.add_argument(
        "--sre-version",
        default=DEFAULT_SRE_VERSION,
        help="SRE snapshot version folder",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help='Scenario filter, e.g. "all", "1-35", "20,23,105", "1-22,102,105"',
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Retries per scenario",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=8.0,
        help="Minimum sleep between successful scenario downloads",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=20.0,
        help="Maximum sleep between successful scenario downloads",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel download workers inside Hugging Face downloader. Keep at 1 to be gentle.",
    )
    parser.add_argument(
        "--min-file-ratio",
        type=float,
        default=0.95,
        help="Local scenario is considered complete if it has at least this ratio of expected remote files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be downloaded",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download scenarios even if they look complete",
    )
    parser.add_argument(
        "--disable-xet",
        action="store_true",
        help="Set HF_HUB_DISABLE_XET=1 before downloading",
    )
    args = parser.parse_args()

    # Safer defaults for slow/unstable connections.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    scenario_filter = parse_scenario_filter(args.scenarios)

    print("Listing remote files from Hugging Face...")
    remote_by_scenario = list_remote_sre_scenarios(
        sre_version=args.sre_version,
        scenario_filter=scenario_filter,
    )

    if not remote_by_scenario:
        raise RuntimeError(
            "No remote scenarios found. Check --sre-version or repo access."
        )

    print(f"Remote scenarios found: {list(remote_by_scenario.keys())}")

    to_download: list[tuple[int, str]] = []
    already_complete: list[int] = []

    for scenario_id, expected_files in remote_by_scenario.items():
        complete, reason = scenario_is_complete(
            local_dir=local_dir,
            sre_version=args.sre_version,
            scenario_id=scenario_id,
            expected_files=expected_files,
            min_file_ratio=args.min_file_ratio,
        )

        if complete and not args.force:
            already_complete.append(scenario_id)
            print(f"Scenario-{scenario_id}: skip ({reason})")
        else:
            to_download.append((scenario_id, reason))
            print(f"Scenario-{scenario_id}: needs download ({reason})")

    print()
    print("Summary before download")
    print(f"Already complete: {already_complete}")
    print(f"To download: {[sid for sid, _ in to_download]}")

    if args.dry_run:
        print("Dry run only. No files downloaded.")
        return

    ok: list[int] = []
    failed: list[int] = []

    for index, (scenario_id, reason) in enumerate(to_download, start=1):
        print()
        print("=" * 80)
        print(f"{index}/{len(to_download)} Scenario-{scenario_id}")
        print(f"Reason: {reason}")

        success = download_one_scenario(
            local_dir=local_dir,
            sre_version=args.sre_version,
            scenario_id=scenario_id,
            retries=args.retries,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            max_workers=args.max_workers,
        )

        expected_files = remote_by_scenario[scenario_id]
        complete, final_reason = scenario_is_complete(
            local_dir=local_dir,
            sre_version=args.sre_version,
            scenario_id=scenario_id,
            expected_files=expected_files,
            min_file_ratio=args.min_file_ratio,
        )

        if success and complete:
            ok.append(scenario_id)
            print(f"Scenario-{scenario_id}: OK after download ({final_reason})")
        else:
            failed.append(scenario_id)
            print(f"Scenario-{scenario_id}: still incomplete ({final_reason})")

        if index < len(to_download):
            polite_sleep(args.min_delay, args.max_delay)

    print()
    print("=" * 80)
    print("Final download summary")
    print(f"Already complete before run: {already_complete}")
    print(f"Downloaded OK: {ok}")
    print(f"Failed/incomplete: {failed}")

    if failed:
        print()
        print("Retry failed scenarios later with:")
        failed_arg = ",".join(str(x) for x in failed)
        print(
            f'python .\\CloudRCA\\scripts\\download_missing_sre_scenarios.py '
            f'--scenarios "{failed_arg}" --retries {args.retries} --disable-xet'
        )


if __name__ == "__main__":
    main()
