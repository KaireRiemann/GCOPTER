#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"no rows found in {csv_path}")
    return rows


def to_series(rows, key):
    return [float(row[key]) for row in rows]


def main():
    parser = argparse.ArgumentParser(description="Plot GCOPTER NUBS scale scan curves from CSV.")
    parser.add_argument("csv", type=Path, help="Path to the scale scan CSV file.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    scales = to_series(rows, "scale")
    total_cost = to_series(rows, "total_cost")
    cp_cost = to_series(rows, "cp_cost")
    energy_cost = to_series(rows, "energy_cost")
    time_cost = to_series(rows, "time_cost")
    total_duration = to_series(rows, "total_duration")
    min_segment_time = to_series(rows, "min_segment_time")
    label = rows[0]["label"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(scales, total_cost, label="total", linewidth=2.2)
    axes[0].plot(scales, cp_cost, label="cp", linewidth=1.8)
    axes[0].plot(scales, energy_cost, label="energy", linewidth=1.8)
    axes[0].plot(scales, time_cost, label="time", linewidth=1.8)
    axes[0].set_ylabel("Cost")
    axes[0].set_title(f"Scale Scan: {label}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(scales, total_duration, color="tab:green", linewidth=2.0)
    axes[1].set_ylabel("Total Duration [s]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(scales, min_segment_time, color="tab:red", linewidth=2.0)
    axes[2].set_ylabel("Min Segment Time [s]")
    axes[2].set_xlabel("Global Scale")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    if args.output is None:
        result_dir = Path(__file__).resolve().parent / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        output_path = result_dir / f"{args.csv.stem}.png"
    else:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=160)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
