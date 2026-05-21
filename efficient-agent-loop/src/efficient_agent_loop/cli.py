from __future__ import annotations

import argparse
import json
from typing import Sequence

from .scheduler import Scheduler, run_demo
from .schemas import LabConfig, read_experiment, read_lab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="efficient-agent-loop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one experiment loop")
    run_parser.add_argument("--experiment", required=True, help="path to experiment.json")
    run_parser.add_argument("--lab", required=True, help="path to lab.yaml or lab.json")
    run_parser.add_argument("--workdir", help="git worktree to run the experiment in")
    run_parser.add_argument(
        "--resume-run-dir",
        help="continue an existing .eal run directory from its latest completed reviewer decision",
    )
    run_parser.add_argument("--max-iterations", type=int, help="override lab max_iterations")
    run_parser.add_argument(
        "--commit",
        action="store_true",
        help="commit experiment changes when reviewer returns ACCEPT",
    )
    run_parser.add_argument(
        "--no-rollback",
        action="store_true",
        help="do not rollback when reviewer returns REJECT",
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help="write a no-op three-role event trace without running agents",
    )
    demo_parser.add_argument("--experiment", required=True, help="path to experiment.json")
    demo_parser.add_argument(
        "--lab",
        help="optional lab.yaml or lab.json, used only to display configured commands",
    )
    demo_parser.add_argument(
        "--workdir",
        default=".",
        help="directory where .eal/demo-runs will be written",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        experiment = read_experiment(args.experiment)
        lab = read_lab(args.lab)
        lab = _apply_overrides(lab, args)
        scheduler = Scheduler(experiment, lab, workdir=args.workdir)
        result = scheduler.run(resume_run_root=args.resume_run_dir)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.status in {"accepted", "rejected", "stopped"} else 1

    if args.command == "demo":
        experiment = read_experiment(args.experiment)
        lab = read_lab(args.lab) if args.lab else None
        run_dir = run_demo(experiment, lab, workdir=args.workdir)
        result = {
            "experiment_name": experiment.name,
            "run_dir": str(run_dir),
            "events": str(run_dir / "events.jsonl"),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


def _apply_overrides(lab: LabConfig, args: argparse.Namespace) -> LabConfig:
    return LabConfig(
        executor_command=lab.executor_command,
        checker_command=lab.checker_command,
        reviewer_command=lab.reviewer_command,
        workdir=args.workdir or lab.workdir,
        runs_dir=lab.runs_dir,
        max_iterations=(
            lab.max_iterations if args.max_iterations is None else args.max_iterations
        ),
        commit_on_accept=args.commit or lab.commit_on_accept,
        rollback_on_reject=False if args.no_rollback else lab.rollback_on_reject,
        commit_message_template=lab.commit_message_template,
        executor_permission=lab.executor_permission,
        checker_permission=lab.checker_permission,
        reviewer_permission=lab.reviewer_permission,
        executor_readme=lab.executor_readme,
        checker_readme=lab.checker_readme,
        reviewer_readme=lab.reviewer_readme,
        resource_hold_enabled=lab.resource_hold_enabled,
        resource_hold_timeout_seconds=lab.resource_hold_timeout_seconds,
        resource_hold_poll_seconds=lab.resource_hold_poll_seconds,
        resource_hold_external_memory_threshold_mib=(
            lab.resource_hold_external_memory_threshold_mib
        ),
        retest_delay_seconds=lab.retest_delay_seconds,
        gpu_retest_wait_timeout_seconds=lab.gpu_retest_wait_timeout_seconds,
        gpu_retest_wait_poll_seconds=lab.gpu_retest_wait_poll_seconds,
        gpu_retest_external_memory_threshold_mib=(
            lab.gpu_retest_external_memory_threshold_mib
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
