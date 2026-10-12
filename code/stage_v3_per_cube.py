"""Stage v3 per-cube driver: Sections 1-6.

Run: python stage_v3_per_cube.py FRT00003E12 [FRT0000871C ...]
     python stage_v3_per_cube.py             # all four cubes
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from sklearn.mixture import GaussianMixture

from config import CUBE_IDS, RANDOM_SEED, STAGE_B_PARAM_ORDER, out_dir
from crism_io import load_cube_by_id
from display import apply_display_transform
from stage_v3_common import (
    LOBE_DEFS,
    MAFIC_LOBE,
    PARAM_INDEX,
    PC_INTERPRETATIONS,
    SMECTITE_LOBE,
    TARGETED_GMM,
    annotate_spectrum,
    backproject,
    csvpath,
    figpath,
    lobe_color,
    lobe_mask_pca,
    load_cube_artifacts,
    md5_of,
    percentile_norm,
    percentile_stretch,
    v3_dark_style,
)

GMM_PALETTE = np.array([
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
])


# ---------- Section 1: PC score maps + RGB composites ----------
def section_1(cube_art: dict, log: dict) -> None:
    cube_id = cube_art["cube_id"]
    pca = cube_art["pca"]
    valid = cube_art["valid"]
    interp = PC_INTERPRETATIONS[cube_id]
    evr = pca["explained_variance_ratio"]
    transformed = pca["transformed"]

    pc_imgs = []
    for n in range(1, 6):
        scores = transformed[:, n - 1]
        img = backproject(scores, valid)
        pc_imgs.append(img)
        with v3_dark_style():
            fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
            disp = apply_display_transform(img)
            vmin, vmax = percentile_stretch(img, 2, 98)
            im = ax.imshow(disp, cmap="viridis", vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(
                f"{cube_id}  PC{n}  ({evr[n-1]*100:.1f}% var)\n{interp[n]}",
                fontsize=11,
            )
            plt.colorbar(im, ax=ax, shrink=0.85, label="PC score")
            fig.savefig(figpath(f"{cube_id}_PC{n}_scene_v3"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # RGB composites: PC1=R, PC2=G, PC{3,4,5}=B
    for b_idx in (3, 4, 5):
        R = percentile_norm(pc_imgs[0])
        G = percentile_norm(pc_imgs[1])
        B = percentile_norm(pc_imgs[b_idx - 1])
        rgb = np.dstack([R, G, B])
        # NaN -> black so background reads as no-data
        nan_mask = np.isnan(R) | np.isnan(G) | np.isnan(B)
        rgb[nan_mask] = 0.0
        with v3_dark_style():
            fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
            ax.imshow(apply_display_transform(rgb), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(
                f"{cube_id}  RGB = PC1 / PC2 / PC{b_idx}\n"
                f"R: {interp[1]}   |   G: {interp[2]}   |   B: {interp[b_idx]}",
                fontsize=10,
            )
            fig.savefig(figpath(f"{cube_id}_RGB_PC1_PC2_PC{b_idx}_v3"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
    log["section_1"] = {"pc_maps": 5, "rgb_composites": 3}


# ---------- Section 2: PC pair plots ----------
def section_2(cube_art: dict, log: dict) -> None:
    cube_id = cube_art["cube_id"]
    transformed = cube_art["pca"]["transformed"][:, :5]
    evr = cube_art["pca"]["explained_variance_ratio"]

    # 5x5 pairmatrix
    with v3_dark_style():
        fig, axes = plt.subplots(5, 5, figsize=(18, 16),
                                 constrained_layout=True)
        for i in range(5):
            for j in range(5):
                ax = axes[i, j]
                if i == j:
                    ax.hist(transformed[:, i], bins=80, color="#7fdbff",
                            histtype="stepfilled", alpha=0.85)
                    ax.set_title(f"PC{i+1}  {evr[i]*100:.1f}%", fontsize=9)
                else:
                    ax.hexbin(transformed[:, j], transformed[:, i],
                              gridsize=60, cmap="viridis", mincnt=1,
                              norm=LogNorm())
                    ax.set_title(f"PC{j+1} ({evr[j]*100:.1f}%) vs PC{i+1} ({evr[i]*100:.1f}%)",
                                 fontsize=8)
                if i == 4:
                    ax.set_xlabel(f"PC{j+1}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"PC{i+1}", fontsize=8)
        fig.suptitle(f"{cube_id}  PC1-5 pair matrix (hexbin density, log)",
                     fontsize=14)
        fig.savefig(figpath(f"{cube_id}_PC_pairmatrix_v3"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

    # Individual scientifically important pairs
    pairs = [(1, 2), (1, 3), (3, 4), (4, 5)]
    interp = PC_INTERPRETATIONS[cube_id]
    for i, j in pairs:
        with v3_dark_style():
            fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
            hb = ax.hexbin(transformed[:, i - 1], transformed[:, j - 1],
                           gridsize=80, cmap="viridis", mincnt=1,
                           norm=LogNorm())
            ax.axhline(0, color="#888888", ls=":", lw=0.5)
            ax.axvline(0, color="#888888", ls=":", lw=0.5)
            ax.set_xlabel(f"PC{i}  ({evr[i-1]*100:.1f}%)  -- {interp[i]}")
            ax.set_ylabel(f"PC{j}  ({evr[j-1]*100:.1f}%)  -- {interp[j]}")
            ax.set_title(f"{cube_id}  PC{i} vs PC{j}", fontsize=11)
            plt.colorbar(hb, ax=ax, label="pixels (log)")
            fig.savefig(figpath(f"{cube_id}_PC{i}_vs_PC{j}_v3"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
    log["section_2"] = {"pairmatrix": 1, "pair_plots": len(pairs)}


# ---------- Section 3: extreme back-projection ----------
def section_3(cube_art: dict, log: dict) -> None:
    cube_id = cube_art["cube_id"]
    pca = cube_art["pca"]
    valid = cube_art["valid"]
    fs = cube_art["feature_stack"]
    gmm_labels = cube_art["gmm_labels"]
    transformed = pca["transformed"][:, :5]
    interp = PC_INTERPRETATIONS[cube_id]

    # Background: BD1900R2 grayscale (hydration) per spec ("BD1900R2 or MAF greyscale")
    bg = fs["BD1900R2"].astype(np.float32)
    bg_lo, bg_hi = percentile_stretch(bg, 2, 98)

    # GMM background as image
    gmm_img = backproject(gmm_labels.astype(np.float32), valid, fill=-1.0)

    shred_warnings = []
    for n in range(1, 6):
        scores = transformed[:, n - 1]
        for tag, predicate in (("high", scores >= np.percentile(scores, 95)),
                                ("low", scores <= np.percentile(scores, 5))):
            mask_1d = predicate
            mask_img = backproject(mask_1d.astype(np.float32), valid, fill=0.0)

            # Overlay on scene background
            with v3_dark_style():
                fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
                ax.imshow(apply_display_transform(bg), cmap="gray",
                          vmin=bg_lo, vmax=bg_hi, interpolation="nearest")
                overlay = np.where(apply_display_transform(mask_img) > 0,
                                   1.0, np.nan)
                color = "#ff5555" if tag == "high" else "#55aaff"
                ax.imshow(overlay, cmap=ListedColormap([color]),
                          vmin=0, vmax=1, alpha=0.85, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(
                    f"{cube_id}  PC{n} {tag}-extreme ({'>=95%' if tag=='high' else '<=5%'}) "
                    f"on BD1900R2 background\n{interp[n]}", fontsize=10
                )
                fig.savefig(figpath(f"{cube_id}_PC{n}_extreme_{tag}_v3"),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)

            # Overlay on GMM classification
            with v3_dark_style():
                fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
                gmm_disp = apply_display_transform(gmm_img)
                gmm_show = np.where(gmm_disp >= 0, gmm_disp, np.nan)
                ax.imshow(gmm_show, cmap=ListedColormap(GMM_PALETTE[:6]),
                          vmin=0, vmax=5, interpolation="nearest", alpha=0.6)
                overlay = np.where(apply_display_transform(mask_img) > 0,
                                   1.0, np.nan)
                color = "#ffffff"
                ax.imshow(overlay, cmap=ListedColormap([color]),
                          vmin=0, vmax=1, alpha=0.95, interpolation="nearest")
                # Count GMM classes covered by extreme
                gmm_in_extreme = gmm_labels[mask_1d]
                vals, counts = np.unique(gmm_in_extreme, return_counts=True)
                # Classes with > 10% share inside extreme
                share = counts / counts.sum()
                n_significant = int((share >= 0.10).sum())
                shred_flag = n_significant >= 3
                if shred_flag:
                    shred_warnings.append({
                        "pc": n, "tag": tag, "n_significant_classes": n_significant,
                        "shares": dict(zip(vals.tolist(), share.tolist())),
                    })
                ax.set_xticks([]); ax.set_yticks([])
                title = (
                    f"{cube_id}  PC{n} {tag}-extreme on k=6 GMM\n"
                    f"{interp[n]}   |   classes covered: {dict(zip(vals.tolist(), counts.tolist()))}"
                )
                if shred_flag:
                    title += "  ** mode shredded **"
                ax.set_title(title, fontsize=9)
                fig.savefig(figpath(f"{cube_id}_PC{n}_extreme_{tag}_on_gmm_v3"),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
    log["section_3"] = {
        "extreme_pairs": 5 * 2,
        "scene_overlays": 5 * 2,
        "gmm_overlays": 5 * 2,
        "mode_shred_warnings": shred_warnings,
    }


# ---------- Section 4a: whitened GMM k=6 ----------
def section_4a(cube_art: dict, log: dict) -> None:
    cube_id = cube_art["cube_id"]
    transformed = cube_art["pca"]["transformed"][:, :5].astype(np.float32)
    valid = cube_art["valid"]
    col_std = transformed.std(axis=0, ddof=1)
    whitened = transformed / col_std

    gmm = GaussianMixture(n_components=6, covariance_type="full",
                          n_init=5, random_state=RANDOM_SEED, reg_covar=1e-5)
    gmm.fit(whitened)
    labels = gmm.predict(whitened).astype(np.int16)
    np.save(out_dir("cluster_labels") / f"{cube_id}_gmm_pca5whitened_k6_v3.npy", labels)

    img = backproject(labels.astype(np.float32), valid, fill=-1.0)
    disp = apply_display_transform(img)
    with v3_dark_style():
        fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
        show = np.where(disp >= 0, disp, np.nan)
        ax.imshow(show, cmap=ListedColormap(GMM_PALETTE[:6]),
                  vmin=0, vmax=5, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        sizes = [int((labels == c).sum()) for c in range(6)]
        ax.set_title(
            f"{cube_id}  GMM k=6 on whitened PC1-5 (PC/std)\n"
            f"sizes: {sizes}   |   BIC: {gmm.bic(whitened):.0f}", fontsize=10,
        )
        fig.savefig(figpath(f"{cube_id}_gmm_k6_whitened_v3"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    log["section_4a"] = {"sizes": sizes, "bic": float(gmm.bic(whitened))}


# ---------- Section 4b: higher k on whitened ----------
def section_4b(cube_art: dict, log: dict) -> None:
    cube_id = cube_art["cube_id"]
    transformed = cube_art["pca"]["transformed"][:, :5].astype(np.float32)
    valid = cube_art["valid"]
    fs = cube_art["feature_stack"]
    col_std = transformed.std(axis=0, ddof=1)
    whitened = transformed / col_std

    bic_curve = {"k": [], "bic": [], "aic": []}
    for k in (4, 5, 6, 7, 8, 9, 10, 11, 12):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=3, random_state=RANDOM_SEED, reg_covar=1e-5)
        gm.fit(whitened)
        bic_curve["k"].append(k)
        bic_curve["bic"].append(float(gm.bic(whitened)))
        bic_curve["aic"].append(float(gm.aic(whitened)))

    with v3_dark_style():
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(bic_curve["k"], bic_curve["bic"], marker="o",
                color="#ffaa44", label="BIC")
        ax.plot(bic_curve["k"], bic_curve["aic"], marker="s", ls="--",
                color="#888888", label="AIC")
        ax.set_xlabel("k"); ax.set_ylabel("information criterion")
        ax.set_title(f"{cube_id}  GMM whitened-PC sweep")
        ax.grid(alpha=0.3); ax.legend()
        fig.savefig(figpath(f"{cube_id}_gmm_BIC_v3"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    candidates = {}
    for k in (8, 10, 12):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=3, random_state=RANDOM_SEED, reg_covar=1e-5)
        gm.fit(whitened)
        labels = gm.predict(whitened).astype(np.int16)
        np.save(out_dir("cluster_labels") / f"{cube_id}_gmm_pca5whitened_k{k}_v3.npy", labels)
        img = backproject(labels.astype(np.float32), valid, fill=-1.0)
        disp = apply_display_transform(img)
        with v3_dark_style():
            fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
            show = np.where(disp >= 0, disp, np.nan)
            cmap = ListedColormap(GMM_PALETTE[:k])
            ax.imshow(show, cmap=cmap, vmin=0, vmax=k - 1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            sizes = [int((labels == c).sum()) for c in range(k)]
            ax.set_title(
                f"{cube_id}  GMM k={k} on whitened PC1-5\nsizes: {sizes}",
                fontsize=9,
            )
            fig.savefig(figpath(f"{cube_id}_gmm_k{k}_whitened_v3"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Carbonate / ferric candidate flagging
        bd2500 = fs["BD2500_2"][valid].astype(np.float32)
        bd530 = fs["BD530_2"][valid].astype(np.float32)
        scene_bd2500_mu = float(np.nanmean(bd2500))
        scene_bd2500_sd = float(np.nanstd(bd2500))
        scene_bd530_mu = float(np.nanmean(bd530))
        scene_bd530_sd = float(np.nanstd(bd530))
        per_cluster_flags = []
        for c in range(k):
            mask = labels == c
            if not mask.any():
                continue
            mu25 = float(np.nanmean(bd2500[mask]))
            mu53 = float(np.nanmean(bd530[mask]))
            flags = []
            if mu25 > scene_bd2500_mu + scene_bd2500_sd:
                flags.append("carbonate")
            if mu53 > scene_bd530_mu + scene_bd530_sd:
                flags.append("ferric")
            if flags:
                per_cluster_flags.append({
                    "k": k, "cluster": c, "n_pixels": int(mask.sum()),
                    "BD2500_2_mean": mu25, "BD530_2_mean": mu53,
                    "flags": flags,
                })
        candidates[k] = per_cluster_flags
    log["section_4b"] = {"bic_curve": bic_curve, "candidates": candidates}


# ---------- Section 4c: targeted PC subset GMM ----------
def section_4c(cube_art: dict, log: dict) -> None:
    cube_id = cube_art["cube_id"]
    transformed = cube_art["pca"]["transformed"]
    valid = cube_art["valid"]
    fs = cube_art["feature_stack"]
    cfg = TARGETED_GMM[cube_id]
    pcs, k = cfg["pcs"], cfg["k"]
    col_std = transformed[:, :5].std(axis=0, ddof=1)
    cols = [p - 1 for p in pcs]
    X = (transformed[:, cols] / col_std[cols]).astype(np.float32)
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         n_init=5, random_state=RANDOM_SEED, reg_covar=1e-5)
    gm.fit(X)
    labels = gm.predict(X).astype(np.int16)
    np.save(out_dir("cluster_labels") / f"{cube_id}_gmm_minorPC_v3.npy", labels)

    bd2500 = fs["BD2500_2"]
    bd530 = fs["BD530_2"]
    bd2500_disp = apply_display_transform(bd2500)
    bd530_disp = apply_display_transform(bd530)
    bd2500_lo, bd2500_hi = percentile_stretch(bd2500, 2, 98)
    bd530_lo, bd530_hi = percentile_stretch(bd530, 2, 98)

    label_img = backproject(labels.astype(np.float32), valid, fill=-1.0)
    disp = apply_display_transform(label_img)

    with v3_dark_style():
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
        # Cluster map
        show = np.where(disp >= 0, disp, np.nan)
        axes[0].imshow(show, cmap=ListedColormap(GMM_PALETTE[:k]),
                       vmin=0, vmax=k - 1, interpolation="nearest")
        axes[0].set_title(
            f"GMM k={k} on PCs {pcs}  ({cfg['rationale']})", fontsize=10,
        )
        axes[0].set_xticks([]); axes[0].set_yticks([])

        # BD2500 with cluster outlines
        axes[1].imshow(bd2500_disp, cmap="gray", vmin=bd2500_lo, vmax=bd2500_hi)
        for c in range(k):
            mask_disp = (disp == c).astype(np.float32)
            mask_disp[mask_disp == 0] = np.nan
            axes[1].imshow(
                mask_disp,
                cmap=ListedColormap([GMM_PALETTE[c]]),
                vmin=0, vmax=1, alpha=0.55, interpolation="nearest",
            )
        axes[1].set_title("clusters on BD2500_2 (carbonate)", fontsize=10)
        axes[1].set_xticks([]); axes[1].set_yticks([])

        # BD530 with cluster outlines
        axes[2].imshow(bd530_disp, cmap="gray", vmin=bd530_lo, vmax=bd530_hi)
        for c in range(k):
            mask_disp = (disp == c).astype(np.float32)
            mask_disp[mask_disp == 0] = np.nan
            axes[2].imshow(
                mask_disp,
                cmap=ListedColormap([GMM_PALETTE[c]]),
                vmin=0, vmax=1, alpha=0.55, interpolation="nearest",
            )
        axes[2].set_title("clusters on BD530_2 (ferric)", fontsize=10)
        axes[2].set_xticks([]); axes[2].set_yticks([])

        fig.suptitle(
            f"{cube_id}  Targeted minor-PC GMM  (PCs {pcs}, k={k})", fontsize=12,
        )
        fig.savefig(figpath(f"{cube_id}_gmm_minorPC_v3"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    log["section_4c"] = {
        "pcs": pcs, "k": k, "rationale": cfg["rationale"],
        "sizes": [int((labels == c).sum()) for c in range(k)],
    }


# ---------- Section 5: per-cluster diagnostics ----------
def section_5(cube_art: dict, log: dict) -> None:
    cube_id = cube_art["cube_id"]
    transformed = cube_art["pca"]["transformed"]
    valid = cube_art["valid"]
    fs = cube_art["feature_stack"]
    # Preferred GMM = the standard k=6 fixed (already used everywhere).
    labels = cube_art["gmm_labels"]
    k = int(labels.max()) + 1

    param_arrays = {p: fs[p][valid].astype(np.float32) for p in STAGE_B_PARAM_ORDER}
    pc_means = transformed[:, :5]

    rows = []
    for c in range(k):
        mask = labels == c
        n = int(mask.sum())
        row = {"cluster_id": c, "n_pixels": n}
        for n_pc in range(5):
            row[f"PC{n_pc+1}_mean"] = float(np.mean(pc_means[mask, n_pc])) if n else float("nan")
        for p in STAGE_B_PARAM_ORDER:
            row[f"{p}_mean"] = float(np.nanmean(param_arrays[p][mask])) if n else float("nan")
        rows.append(row)

    csv_path = csvpath(f"{cube_id}_cluster_diagnostics_v3")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Heatmap with mineral candidate flags
    flag_table = np.zeros((k, 5), dtype=np.float32)
    flag_labels = ("olivine", "smectite", "carbonate", "ferric", "redox")
    scene = {p: float(np.nanmean(param_arrays[p])) for p in STAGE_B_PARAM_ORDER}
    sd = {p: float(np.nanstd(param_arrays[p])) for p in STAGE_B_PARAM_ORDER}
    for c, row in enumerate(rows):
        if row["n_pixels"] == 0:
            continue
        # olivine: high OLINDEX3 + low BD1900R2
        flag_table[c, 0] = ((row["OLINDEX3_mean"] - scene["OLINDEX3"]) / max(sd["OLINDEX3"], 1e-9)
                            - max(0, (row["BD1900R2_mean"] - scene["BD1900R2"]) / max(sd["BD1900R2"], 1e-9)))
        # smectite: high BD1900R2 + high D2300
        flag_table[c, 1] = ((row["BD1900R2_mean"] - scene["BD1900R2"]) / max(sd["BD1900R2"], 1e-9)
                            + (row["D2300_mean"] - scene["D2300"]) / max(sd["D2300"], 1e-9))
        # carbonate: high BD2500_2
        flag_table[c, 2] = (row["BD2500_2_mean"] - scene["BD2500_2"]) / max(sd["BD2500_2"], 1e-9)
        # ferric: high BD530_2 + high BD920_2
        flag_table[c, 3] = ((row["BD530_2_mean"] - scene["BD530_2"]) / max(sd["BD530_2"], 1e-9)
                            + (row["BD920_2_mean"] - scene["BD920_2"]) / max(sd["BD920_2"], 1e-9))
        # redox: difference of BD530 and BD920 (proxy for Fe3+/Fe2+ shape)
        flag_table[c, 4] = ((row["BD530_2_mean"] - scene["BD530_2"]) / max(sd["BD530_2"], 1e-9)
                            - (row["BD920_2_mean"] - scene["BD920_2"]) / max(sd["BD920_2"], 1e-9))

    with v3_dark_style():
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        vmax = max(2.0, float(np.nanmax(np.abs(flag_table))))
        im = ax.imshow(flag_table, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels(flag_labels, rotation=30, ha="right")
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"c{c}" for c in range(k)])
        for i in range(k):
            for j in range(5):
                ax.text(j, i, f"{flag_table[i, j]:+.2f}", ha="center", va="center",
                        color="white" if abs(flag_table[i, j]) > vmax * 0.5 else "black",
                        fontsize=8)
        ax.set_title(f"{cube_id}  k=6 cluster mineral-candidate z-scores", fontsize=11)
        plt.colorbar(im, ax=ax, label="z-score")
        fig.savefig(figpath(f"{cube_id}_cluster_diagnostics_v3"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    log["section_5"] = {"csv": str(csv_path), "n_clusters": k}


# ---------- Section 6: lobe spectra ----------
def section_6(cube_art: dict, log: dict) -> None:
    cube_id = cube_art["cube_id"]
    pca = cube_art["pca"]
    fs = cube_art["feature_stack"]
    yy = pca["line"]
    xx = pca["sample"]
    wav = cube_art["wav"]
    cube_data, _, _, _ = load_cube_by_id(cube_id)  # (bands, lines, samples)
    print(f"  loaded TER3 cube  shape={cube_data.shape}")

    lobes = LOBE_DEFS[cube_id]
    mafic_label = MAFIC_LOBE[cube_id]
    mafic_mask = lobe_mask_pca(cube_art, lobes[mafic_label])
    if mafic_mask.sum() < 20:
        print(f"  WARNING: mafic lobe {mafic_label} has only {mafic_mask.sum()} pixels - ratio will be noisy")

    mafic_yy = yy[mafic_mask]; mafic_xx = xx[mafic_mask]
    mafic_specs = cube_data[:, mafic_yy, mafic_xx]
    mafic_mean = np.nanmean(mafic_specs, axis=1)

    lobe_summary = {}
    combined_ratios = {}
    for name, lobe in lobes.items():
        mask = lobe_mask_pca(cube_art, lobe)
        n = int(mask.sum())
        if n < 5:
            print(f"  lobe {name}: only {n} pixels, skipping")
            lobe_summary[name] = {"n_pixels": n, "skipped": True, "label": lobe["label"]}
            continue
        lyy = yy[mask]; lxx = xx[mask]
        specs = cube_data[:, lyy, lxx]
        mean_spec = np.nanmean(specs, axis=1)
        p16 = np.nanpercentile(specs, 16, axis=1)
        p84 = np.nanpercentile(specs, 84, axis=1)
        ratio = mean_spec / np.where(np.isfinite(mafic_mean) & (np.abs(mafic_mean) > 1e-6),
                                      mafic_mean, np.nan)
        combined_ratios[name] = ratio

        # Raw spectrum + envelope
        with v3_dark_style():
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            ax.fill_between(wav, p16, p84, color=lobe_color(name), alpha=0.25,
                            label="16-84%")
            ax.plot(wav, mean_spec, color=lobe_color(name), lw=1.6,
                    label=f"mean (n={n})")
            ax.set_xlim(0.4, 2.6)
            annotate_spectrum(ax)
            ax.set_xlabel(r"wavelength ($\mu$m)")
            ax.set_ylabel("I/F (atmospherically corrected, TER3)")
            ax.set_title(f"{cube_id}  Lobe {name}: {lobe['label']}", fontsize=11)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.2)
            fig.savefig(figpath(f"{cube_id}_lobe_{name}_spectra_raw_v3"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Ratio to mafic
        with v3_dark_style():
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            ax.plot(wav, ratio, color=lobe_color(name), lw=1.6,
                    label=f"Lobe {name} / Lobe {mafic_label} (mafic bulk)")
            ax.axhline(1.0, color="#888888", ls=":", lw=0.8)
            ax.set_xlim(0.4, 2.6)
            annotate_spectrum(ax)
            ax.set_xlabel(r"wavelength ($\mu$m)")
            ax.set_ylabel(f"ratio to Lobe {mafic_label}")
            ax.set_title(
                f"{cube_id}  Lobe {name} / Lobe {mafic_label} ratio  ({lobe['label']})",
                fontsize=11,
            )
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.2)
            fig.savefig(figpath(f"{cube_id}_lobe_{name}_ratio_mafic_v3"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        lobe_summary[name] = {
            "n_pixels": n, "label": lobe["label"], "pcs": list(lobe["pcs"]),
            "mean_spectrum_at_2_3um": float(mean_spec[np.argmin(np.abs(wav - 2.3))]),
            "mean_spectrum_at_2_5um": float(mean_spec[np.argmin(np.abs(wav - 2.5))]),
        }

    # Combined overlay of all ratios
    with v3_dark_style():
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        for name, ratio in combined_ratios.items():
            if name == mafic_label:
                continue  # ratio of mafic to itself = 1
            ax.plot(wav, ratio, lw=1.6, color=lobe_color(name),
                    label=f"Lobe {name}: {lobes[name]['label']}")
        ax.axhline(1.0, color="#888888", ls=":", lw=0.8)
        ax.set_xlim(0.4, 2.6)
        annotate_spectrum(ax)
        ax.set_xlabel(r"wavelength ($\mu$m)")
        ax.set_ylabel(f"ratio to Lobe {mafic_label} (mafic bulk)")
        ax.set_title(f"{cube_id}  All lobe ratios (overlay)", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
        fig.savefig(figpath(f"{cube_id}_lobes_combined_v3"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    log["section_6"] = {
        "mafic_lobe": mafic_label,
        "lobes": lobe_summary,
        "wav_range_um": (float(wav[0]), float(wav[-1])),
    }


def process_cube(cube_id: str) -> dict:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n=== {cube_id} v3 per-cube ===")
    cube_art = load_cube_artifacts(cube_id)
    log = {"cube_id": cube_id, "stage": "v3_per_cube", "run_timestamp_utc": timestamp,
           "random_seed": RANDOM_SEED}

    print("  section 1 PC maps + RGB...")
    section_1(cube_art, log)
    print("  section 2 PC pair plots...")
    section_2(cube_art, log)
    print("  section 3 extreme back-projection...")
    section_3(cube_art, log)
    print("  section 4a whitened k=6...")
    section_4a(cube_art, log)
    print("  section 4b higher k whitened...")
    section_4b(cube_art, log)
    print("  section 4c targeted PC subset...")
    section_4c(cube_art, log)
    print("  section 5 cluster diagnostics...")
    section_5(cube_art, log)
    print("  section 6 lobe spectra (loads TER3 cube)...")
    section_6(cube_art, log)

    log["wall_clock_seconds"] = round(time.time() - t0, 2)
    log["source_code_md5"] = {
        "stage_v3_per_cube.py": md5_of(Path(__file__)),
        "stage_v3_common.py": md5_of(Path(__file__).with_name("stage_v3_common.py")),
        "config.py": md5_of(Path(__file__).with_name("config.py")),
    }
    log_path = out_dir("logs") / f"{cube_id}_v3_per_cube_{timestamp}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"  done in {log['wall_clock_seconds']}s -> {log_path.name}")
    return log


def main(argv: list[str]) -> None:
    targets = argv[1:] if len(argv) > 1 else list(CUBE_IDS)
    for cube_id in targets:
        process_cube(cube_id)


if __name__ == "__main__":
    main(sys.argv)
