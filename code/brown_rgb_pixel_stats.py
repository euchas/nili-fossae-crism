"""Per-band, per-cube + 4-cube-pooled pixel-value distributions for the
Brown RGB composites. Used to pick a common stretch range across the
four cubes so cross-cube comparison preserves "identical color =
identical parameter value".

Reads the same v6 Brown .npy files used by brown_rgb_unstretched.py.
Reports percentiles {1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99} per band
per cube and over the full 4-cube pool, plus suggested stretch ranges.

Output: outputs/_brown_rgb_unstretched/brown_rgb_pixel_stats.{md,json}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import CUBE_IDS, OUTPUTS_DIR
from stage_v7c_mtr3_pipeline import MTR3_CUBE_DIRS, CUBES_DIR

PARAMS = ("centroid", "amplitude", "asymmetry", "hwhm")
PERCENTILES = (1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99)
SNAPSHOT = OUTPUTS_DIR / "brown_rgb"


def load_param(cube_id: str, name: str) -> np.ndarray:
    dir_name = MTR3_CUBE_DIRS[cube_id]
    path = CUBES_DIR / dir_name / f"{dir_name}_{name}_map_v6.npy"
    return np.load(path).astype(np.float32)


def stats_for(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    if n == 0:
        return {"n": 0}
    out = {
        "n": n,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }
    pcts = np.percentile(finite, PERCENTILES)
    for p, v in zip(PERCENTILES, pcts):
        out[f"p{p}"] = float(v)
    return out


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)

    per_cube: dict = {}
    pooled_arrays: dict = {p: [] for p in PARAMS}

    for cube_id in CUBE_IDS:
        per_cube[cube_id] = {}
        for p in PARAMS:
            arr = load_param(cube_id, p)
            per_cube[cube_id][p] = stats_for(arr)
            finite = arr[np.isfinite(arr)]
            pooled_arrays[p].append(finite)

    pooled = {p: stats_for(np.concatenate(pooled_arrays[p])) for p in PARAMS}

    # Suggested stretch ranges - 4-cube-pooled percentiles
    suggestions = {}
    for p in PARAMS:
        s = pooled[p]
        suggestions[p] = {
            "p2_p98":   [s["p2"],  s["p98"]],
            "p1_p99":   [s["p1"],  s["p99"]],
            "p5_p95":   [s["p5"],  s["p95"]],
        }
    # Centroid - apply a Brown 2020 Eq.2 floor option (>= 1.15 um)
    suggestions["centroid"]["p2_p98_brown_eq2_floor"] = [
        max(1.15, pooled["centroid"]["p2"]), pooled["centroid"]["p98"]]

    # JSON output
    json_payload = {
        "timestamp_utc": timestamp,
        "params": list(PARAMS),
        "percentiles_used": list(PERCENTILES),
        "per_cube": per_cube,
        "pooled_4cube": pooled,
        "suggested_common_stretch": suggestions,
        "notes": (
            "Stretches are common across all 4 cubes by design "
            "(identical color == identical parameter value). For "
            "centroid, 'p2_p98_brown_eq2_floor' clamps the low end "
            "to the Brown 2020 Eq.2 reliability threshold of 1.15 um, "
            "useful if you want to suppress the sub-1.15 illumination "
            "tail without dropping pixels."
        ),
    }
    json_path = SNAPSHOT / f"brown_rgb_pixel_stats_{timestamp}.json"
    json_path.write_text(json.dumps(json_payload, indent=2),
                         encoding="utf-8")

    # Print per-band tables to terminal + write a Markdown file
    md_lines = [
        f"# Brown RGB pixel-value distributions ({timestamp})",
        "",
        "Per-cube + 4-cube-pooled distributions for each band of the "
        "unstretched Brown RGB GeoTIFFs.",
        "",
        f"Bands: {', '.join(PARAMS)}",
        f"Percentiles: {', '.join(f'p{p}' for p in PERCENTILES)}",
        "",
    ]

    for p in PARAMS:
        print(f"\n=== {p} ===")
        md_lines.append(f"## {p}\n")
        header = (f"{'cube':<14}{'n':>9} "
                  f"{'min':>8} {'p1':>8} {'p2':>8} {'p5':>8} {'p10':>8} "
                  f"{'p25':>8} {'p50':>8} {'p75':>8} "
                  f"{'p90':>8} {'p95':>8} {'p98':>8} {'p99':>8} "
                  f"{'max':>8}")
        print(header)
        md_lines.append("| cube | n | min | p1 | p2 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p98 | p99 | max |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cube_id in CUBE_IDS:
            s = per_cube[cube_id][p]
            row = (f"{cube_id:<14}{s['n']:>9} "
                   f"{s['min']:>8.3f} {s['p1']:>8.3f} {s['p2']:>8.3f} "
                   f"{s['p5']:>8.3f} {s['p10']:>8.3f} "
                   f"{s['p25']:>8.3f} {s['p50']:>8.3f} {s['p75']:>8.3f} "
                   f"{s['p90']:>8.3f} {s['p95']:>8.3f} {s['p98']:>8.3f} "
                   f"{s['p99']:>8.3f} {s['max']:>8.3f}")
            print(row)
            md_row = (f"| {cube_id} | {s['n']} | {s['min']:.3f} | "
                      f"{s['p1']:.3f} | {s['p2']:.3f} | {s['p5']:.3f} | "
                      f"{s['p10']:.3f} | {s['p25']:.3f} | {s['p50']:.3f} | "
                      f"{s['p75']:.3f} | {s['p90']:.3f} | {s['p95']:.3f} | "
                      f"{s['p98']:.3f} | {s['p99']:.3f} | {s['max']:.3f} |")
            md_lines.append(md_row)
        s = pooled[p]
        row = (f"{'POOLED':<14}{s['n']:>9} "
               f"{s['min']:>8.3f} {s['p1']:>8.3f} {s['p2']:>8.3f} "
               f"{s['p5']:>8.3f} {s['p10']:>8.3f} "
               f"{s['p25']:>8.3f} {s['p50']:>8.3f} {s['p75']:>8.3f} "
               f"{s['p90']:>8.3f} {s['p95']:>8.3f} {s['p98']:>8.3f} "
               f"{s['p99']:>8.3f} {s['max']:>8.3f}")
        print(row)
        md_row = (f"| **POOLED** | {s['n']} | **{s['min']:.3f}** | "
                  f"**{s['p1']:.3f}** | **{s['p2']:.3f}** | "
                  f"**{s['p5']:.3f}** | **{s['p10']:.3f}** | "
                  f"**{s['p25']:.3f}** | **{s['p50']:.3f}** | "
                  f"**{s['p75']:.3f}** | **{s['p90']:.3f}** | "
                  f"**{s['p95']:.3f}** | **{s['p98']:.3f}** | "
                  f"**{s['p99']:.3f}** | **{s['max']:.3f}** |")
        md_lines.append(md_row)
        md_lines.append("")

    # Suggested stretches
    print("\n=== Suggested common stretches (4-cube pooled) ===")
    md_lines.append("## Suggested common stretches (4-cube pooled)\n")
    md_lines.append(
        "Use one of these as the matched stretch across all four cubes "
        "in ArcGIS so identical color = identical parameter value.\n")
    md_lines.append("| band | p2 to p98 | p1 to p99 | p5 to p95 |")
    md_lines.append("|---|---|---|---|")
    for p in PARAMS:
        s = suggestions[p]
        line = (f"  {p:<10} p2-p98 = [{s['p2_p98'][0]:.3f}, "
                f"{s['p2_p98'][1]:.3f}]   p1-p99 = "
                f"[{s['p1_p99'][0]:.3f}, {s['p1_p99'][1]:.3f}]   "
                f"p5-p95 = [{s['p5_p95'][0]:.3f}, {s['p5_p95'][1]:.3f}]")
        print(line)
        md_lines.append(
            f"| {p} | {s['p2_p98'][0]:.3f} - {s['p2_p98'][1]:.3f} | "
            f"{s['p1_p99'][0]:.3f} - {s['p1_p99'][1]:.3f} | "
            f"{s['p5_p95'][0]:.3f} - {s['p5_p95'][1]:.3f} |")

    s = suggestions["centroid"]["p2_p98_brown_eq2_floor"]
    print(f"\n  centroid alt: 4-cube p2 with Brown Eq.2 floor (>=1.15) "
          f"-> [{s[0]:.3f}, {s[1]:.3f}]")
    md_lines.append("")
    md_lines.append(
        "**Centroid alternative** (clamp low end to Brown 2020 Eq.2 "
        f"reliability threshold 1.15 um): **{s[0]:.3f} to {s[1]:.3f}**. "
        "Suppresses the sub-1.15 illumination-artifact tail without "
        "dropping pixels.")
    md_lines.append("")

    md_path = SNAPSHOT / f"brown_rgb_pixel_stats_{timestamp}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
