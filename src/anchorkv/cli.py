"""Command-line entry points for lightweight AnchorKV experiments."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from .capacity import ModelGeometry, estimate_eager_capture
from .heads import discover_receiver_heads, sentence_vertical_scores
from .policy import CacheBudgetError, CacheMode, DelayedAnchorTracker, KVGeometry, plan_cache
from .types import TokenSpan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anchorkv",
        description="Thought-anchor-guided KV-cache compression research toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo",
        help="run a deterministic synthetic receiver-head and cache-planning demo",
    )
    demo.add_argument(
        "--budget-ratio",
        type=float,
        default=0.6,
        help="fraction of the full FP16 cache available to the planner (default: 0.6)",
    )
    demo.add_argument("--seed", type=int, default=7)

    estimate = subparsers.add_parser(
        "estimate-t4",
        help="estimate eager-attention capture memory for Qwen3-0.6B on a 16 GiB T4",
    )
    estimate.add_argument("--sequence-length", type=int, default=1024)
    estimate.add_argument("--batch-size", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        if not 0 < args.budget_ratio <= 1:
            raise SystemExit("--budget-ratio must be in (0, 1]")
        try:
            report = run_synthetic_demo(args.budget_ratio, seed=args.seed)
        except CacheBudgetError as error:
            raise SystemExit(str(error)) from error
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "estimate-t4":
        geometry = ModelGeometry(
            parameters=600_000_000,
            layers=28,
            query_heads=16,
            kv_heads=8,
            head_dim=64,
        )
        estimate = estimate_eager_capture(
            geometry,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
        )
        print(json.dumps(estimate.as_gib(), indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def run_synthetic_demo(budget_ratio: float = 0.6, *, seed: int = 7) -> dict[str, Any]:
    """Exercise the complete offline core without model downloads."""

    rng = np.random.default_rng(seed)
    spans = [TokenSpan(index * 4, (index + 1) * 4, f"reasoning step {index}") for index in range(4)]

    vertical_traces = []
    for _ in range(4):
        attention = rng.uniform(0.0, 0.01, size=(2, 4, 16, 16))
        # Layer 1, head 2 repeatedly broadcasts the first reasoning step.
        attention[1, 2, 4:, 0:4] += 0.25
        vertical_traces.append(sentence_vertical_scores(attention, spans))

    receiver_heads = discover_receiver_heads(vertical_traces, top_k=3)

    tracker = DelayedAnchorTracker(
        min_observations=2,
        provisional_steps=1,
        anchor_threshold=0.05,
    )
    synthetic_scores = [0.12, 0.01, 0.08, 0.02]
    for segment_id, (span, score) in enumerate(zip(spans, synthetic_scores, strict=True)):
        tracker.register(segment_id, span, emitted_step=segment_id)
        tracker.observe(segment_id, score)
        tracker.observe(segment_id, score)

    geometry = KVGeometry(layers=4, kv_heads=2, head_dim=8)
    full_cache_bytes = sum(
        geometry.bytes_for(span.length, CacheMode.FP16) for span in spans
    )
    budget_bytes = int(full_cache_bytes * budget_ratio)
    plan = plan_cache(
        tracker.states(),
        tracker.decisions(current_step=5),
        geometry=geometry,
        budget_bytes=budget_bytes,
        current_step=5,
        recent_window=0,
    )

    return {
        "receiver_heads": [
            {
                "layer": head.layer,
                "query_head": head.query_head,
                "mean_kurtosis": round(head.mean_kurtosis, 4),
                "stability": round(head.stability, 4),
            }
            for head in receiver_heads
        ],
        "cache": {
            "budget_bytes": plan.budget_bytes,
            "used_bytes": plan.used_bytes,
            "full_cache_bytes": plan.full_cache_bytes,
            "compression_ratio": round(plan.compression_ratio, 4),
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "mode": segment.mode.value,
                    "bytes": segment.bytes_used,
                    "importance": round(segment.importance, 4),
                    "reason": segment.reason,
                }
                for segment in plan.segments
            ],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
