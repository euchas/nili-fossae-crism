"""Stage B report writer — pulls logs + summary JSONs into markdown.

Per-cube reports sit at outputs/reports/02_stageB_per_cube_<cube>.md.
The cross-cube notes summary sits at outputs/reports/02_stageB_cross_cube_notes.md.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from config import CUBE_IDS, STAGE_B_PARAM_ORDER, out_dir


def load_latest_log(cube_id: str, tag: str) -> dict | None:
    matches = sorted(glob.glob(
        str(out_dir("logs") / f"{cube_id}_stageB_{tag}_*.json")
    ))
    if not matches:
        return None
    return json.loads(Path(matches[-1]).read_text())


def format_row(values: list) -> str:
    return "| " + " | ".join(str(v) for v in values) + " |"


def per_cube_report(cube_id: str, method_summary: dict) -> Path:
    build = load_latest_log(cube_id, "build") or {}
    pca = load_latest_log(cube_id, "pca") or {}
    umap_log = load_latest_log(cube_id, "umap") or {}
    clpca = load_latest_log(cube_id, "cluster_pca") or {}
    hdb = load_latest_log(cube_id, "hdbscan") or {}
    viz = load_latest_log(cube_id, "visualize") or {}

    lines: list[str] = []
    lines.append(f"# Stage B per-cube report — {cube_id}")
    lines.append("")
    lines.append(f"Generated from logs and summary JSONs. Figures in `outputs/figures/`.")
    lines.append("")

    lines.append("## Feature stack")
    lines.append("")
    if build:
        lines.append(f"- joint_valid_15 pixels: **{build['n_pixels_joint_valid_15']}** "
                     f"({100*build['joint_valid_15_fraction']:.1f}% of frame)")
        lines.append(f"- BDI1000VIS NaN loss vs baseline: "
                     f"{100*build['bdi1000vis_loss_vs_baseline']:.2f}% "
                     f"(flag: {build['bdi1000vis_flag_over_5pct']})")
        lines.append("")
        lines.append("### Per-parameter NaN fractions")
        lines.append("")
        lines.append("| param | NaN frac |")
        lines.append("|---|---:|")
        for name in STAGE_B_PARAM_ORDER:
            nf = build.get("nan_fraction_by_param", {}).get(name, float("nan"))
            lines.append(f"| {name} | {nf:.4f} |")
        lines.append("")

    lines.append("## PCA (15-param)")
    lines.append("")
    if pca:
        evr = pca.get("explained_variance_ratio_15param", [])
        evr8 = pca.get("explained_variance_ratio_8param", [])
        if evr:
            cum5 = sum(evr[:5])
            lines.append(
                f"- PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%, PC3={evr[2]*100:.1f}%, "
                f"PC4={evr[3]*100:.1f}%, PC5={evr[4]*100:.1f}%  → cum5={cum5*100:.1f}%"
            )
        if evr8:
            lines.append(
                f"- 8-param baseline: PC1={evr8[0]*100:.1f}%, PC2={evr8[1]*100:.1f}%, "
                f"PC3={evr8[2]*100:.1f}%"
            )
        lines.append("")
        lines.append("Figures: `*_pca15_variance.png`, `*_pca15_loadings.png`, "
                     "`*_pca15_density.png`, `*_pc1_pc2_maps.png`.")
        lines.append("")

    lines.append("## UMAP")
    lines.append("")
    if umap_log:
        lines.append(f"- heuristic pick: **{umap_log['heuristic_pick']}**")
        lines.append(f"- validity_index pick: **{umap_log['validity_pick']}**")
        lines.append(f"- protocols agree: **{umap_log['picks_agree']}**")
        lines.append(f"- HDBSCAN min_cluster_size used for validity: "
                     f"{umap_log['hdbscan_min_cluster_size_used_for_validity']}")
        lines.append("")
        lines.append("Validity scores per config:")
        lines.append("")
        for k, v in umap_log.get("validity_scores", {}).items():
            lines.append(f"- {k}: {v:.3f}")
        lines.append("")
        lines.append("Figure: `*_umap_sweep.png`, `*_umap_selected_density_*.png`.")
        lines.append("")

    lines.append("## Clustering — k-means + GMM (PCA 5-D input)")
    lines.append("")
    if clpca:
        km_best = clpca.get("kmeans_best_k")
        gmm_best = clpca.get("gmm_best_k")
        lines.append(f"- k-means adaptive k (silhouette): **{km_best}**")
        lines.append(f"- GMM adaptive k (BIC): **{gmm_best}**")
        lines.append(f"- Fixed k across cubes: **{clpca.get('fixed_k')}**")
        lines.append("")
        lines.append("Figures: `*_kmeans_silhouette_sweep.png`, `*_gmm_bic_sweep.png`.")
        lines.append("")

    lines.append("## Clustering — HDBSCAN (UMAP 2-D input)")
    lines.append("")
    if hdb:
        mcs = hdb.get("min_cluster_size")
        lines.append(f"- min_cluster_size: **{mcs}**")
        for prot, res in hdb.get("results", {}).items():
            lines.append(
                f"- [{prot}] clusters={res['n_clusters']}, "
                f"noise={100*res['noise_fraction']:.1f}%"
            )
        lines.append("")

    lines.append("## Per-method cluster summary")
    lines.append("")
    lines.append("Mean spectra + label maps + param-position figures in `outputs/figures/`. "
                 "Ratioed-spectra figures use the auto-selected mafic-cap denominator "
                 "(display-only, never used for clustering).")
    lines.append("")
    if viz:
        lines.append("| method | n_clusters | mafic denom (auto) | denom source |")
        lines.append("|---|---:|---:|---|")
        for mid, m in viz.get("methods_summary", {}).items():
            lines.append(
                f"| {mid} | {m['n_clusters']} | c{m['auto_mafic_pick']} | {m['denominator_source']} |"
            )
        lines.append("")

    lines.append("## Next-stage notes")
    lines.append("")
    lines.append("- For Stage C cluster-matching across cubes, the fixed-k=6 labels "
                 "provide the nominally-comparable reference.")
    lines.append("- If any method's auto-picked mafic denominator looks wrong "
                 "(check the ratioed spectra visually), override via "
                 "`outputs/reports/denominator_overrides.json` and rerun "
                 "`stage_b_visualize.py <cube_id>`.")
    lines.append("")

    path = out_dir("reports") / f"02_stageB_per_cube_{cube_id}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def cross_cube_notes(method_summary: dict) -> Path:
    lines: list[str] = []
    lines.append("# Stage B cross-cube notes")
    lines.append("")
    lines.append("Plots: `all_cubes_median_subtracted.png`, "
                 "`all_cubes_centroid_ks_wasserstein.png`, "
                 "`all_cubes_pca15_loadings_compare.png`. Per-cube deliverables "
                 "under `02_stageB_per_cube_<cube>.md`.")
    lines.append("")

    lines.append("## Centroid pairwise distances")
    lines.append("")
    csv = out_dir("reports") / "02_stageB_centroid_pairwise.csv"
    if csv.exists():
        lines.append("From `02_stageB_centroid_pairwise.csv`:")
        lines.append("")
        lines.append("```")
        lines.append(csv.read_text())
        lines.append("```")
        lines.append("")

    lines.append("## Per-cube method outcomes")
    lines.append("")
    lines.append("| cube | km_adaptive_k | gmm_adaptive_k | hdbscan_heuristic k | noise% |")
    lines.append("|---|---:|---:|---:|---:|")
    for cube_id in CUBE_IDS:
        clpca = load_latest_log(cube_id, "cluster_pca") or {}
        hdb = load_latest_log(cube_id, "hdbscan") or {}
        km_k = clpca.get("kmeans_best_k", "?")
        gmm_k = clpca.get("gmm_best_k", "?")
        heur = hdb.get("results", {}).get("heuristic", {})
        nk = heur.get("n_clusters", "?")
        noise = heur.get("noise_fraction", float("nan"))
        lines.append(f"| {cube_id} | {km_k} | {gmm_k} | {nk} | {100*noise:.1f}% |")
    lines.append("")

    lines.append("## Open questions for review")
    lines.append("")
    lines.append("- Does the fixed-k=6 labelling produce the same six interpretable "
                 "populations across cubes, or do some cubes over- or under-split? "
                 "See per-cube `*_kmeans_k6_fixed_labels.png` side-by-side.")
    lines.append("- Where does the mafic-cap cluster sit in PC1-PC2 on each cube? "
                 "The automatic denominator should fall in the same region on each; "
                 "if A4FC's auto-pick is inconsistent, flag for override.")
    lines.append("- GMM transition maps (`*_gmm_k*_transition.png`) mark pixels with "
                 "high second-best posterior - are those spatially coincident with "
                 "OU boundaries as the Viviano 2013 talc-transition framing predicts?")
    lines.append("")

    path = out_dir("reports") / "02_stageB_cross_cube_notes.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    method_summary_path = out_dir("reports") / "02_stageB_method_summary.json"
    method_summary = (
        json.loads(method_summary_path.read_text())
        if method_summary_path.exists() else {}
    )
    for cube_id in CUBE_IDS:
        path = per_cube_report(cube_id, method_summary.get(cube_id, {}))
        print(f"wrote {path}")
    path = cross_cube_notes(method_summary)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
