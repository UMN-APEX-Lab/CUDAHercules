#!/usr/bin/env python3

from class3_eval_common import (
    add_solution_target_args,
    build_parser,
    load_solution_targets,
    run_single_task_eval,
)


def main() -> int:
    parser = build_parser("icicle_zk")
    add_solution_target_args(parser)
    args = parser.parse_args()
    targets = load_solution_targets(
        manifest_path=args.solution_manifest,
        inline_targets=args.solution_target,
    )
    return run_single_task_eval(
        "icicle_zk",
        args,
        explicit_solution_targets=targets or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())

