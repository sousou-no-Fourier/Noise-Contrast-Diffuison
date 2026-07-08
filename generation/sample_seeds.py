#!/usr/bin/env python
# coding=utf-8
"""Sample the shared seed list from [1, 1024], saved once so every method reuses it.

Seeds are stored in draw order; SSR-3/10/20 use prefixes of the list.
"""

import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Sample the shared multi-seed evaluation list.")
    parser.add_argument("--num_seeds", type=int, default=10, help="How many seeds to draw (max N of SSR-N).")
    parser.add_argument("--seed_min", type=int, default=1)
    parser.add_argument("--seed_max", type=int, default=1024)
    parser.add_argument(
        "--master_seed", type=int, default=None,
        help="RNG seed for reproducible sampling; omit for a fresh random draw each run.",
    )
    parser.add_argument("--output", type=str, default="outputs/seeds.json")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.master_seed)
    seeds = rng.sample(range(args.seed_min, args.seed_max + 1), args.num_seeds)

    payload = {
        "seed_min": args.seed_min,
        "seed_max": args.seed_max,
        "master_seed": args.master_seed,
        "num_seeds": args.num_seeds,
        "seeds": seeds,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[sample_seeds] wrote {args.num_seeds} seeds from [{args.seed_min}, {args.seed_max}] "
          f"to {args.output}: {seeds}")


if __name__ == "__main__":
    main()
