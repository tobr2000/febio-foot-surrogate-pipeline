from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / ".codex_plot_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = PROJECT_ROOT
HERE = Path(__file__).resolve().parent
WANDB_ROOT = ROOT / "wandb_evolution" / "wandb_evolution_output"
QUALITY_ROOT = ROOT / "training" / "dataset_quality"
CONFIG = HERE / "config" / "decision_runs.csv"
DERIVED = HERE / "derived"
FIG_MAIN = HERE / "figures" / "main"
FIG_APPENDIX = HERE / "figures" / "appendix"
ASSETS = HERE / "assets"
SFEM_REGRESSION_LONG = PROJECT_ROOT.parents[1] / "VT_Project" / "sfem_regression_report" / "sfem_regression_baselines_long.csv"
RESELECTED_RAW = HERE / "derived" / "reselected"
RESELECTED_NORM = HERE / "derived" / "reselected_sfem_norm"

COLORS = {
    "Mean map": "#8A8A8A",
    "Ridge": "#7B5AA6",
    "FFN": "#2878B5",
    "GNO": "#E07A1F",
    "PINN": "#2E9D65",
    "Baseline": "#777777",
}
METHOD_ORDER = ["FFN", "GNO", "PINN"]
FULL_METHOD_ORDER = ["Mean map", "Ridge", "FFN", "GNO", "PINN"]
DATASET_ORDER = ["Simplified foot", "Anatomical pilot", "V9", "V9 model-ready", "V10 model-ready"]
REGRESSION_BASELINE_LABELS = {
    "mean_pressure_map": "Mean map",
    "ridge_params_all": "Ridge all",
    "ridge_params_without_base_id": "Ridge no base ID",
    "ridge_physics_only_no_base_profile": "Ridge physical",
    "mean_von_mises_scalar": "Mean scalar",
    "mean_von_mises_norm_scalar": "Mean scalar",
    "ridge_load_only": "Ridge load",
    "ridge_node_geometry_only": "Ridge node geom.",
    "ridge_node_and_global_features_all": "Ridge node+global",
}

HEADLINE_KEYS = {
    "Pressure NRMSE": "pressure/nrmse",
    "Pressure $R^2$": "pressure/r2",
    "Peak pressure NRMSE": "peak_pressure/nrmse",
    "Centre of pressure NRMSE": "center_of_pressure/nrmse",
    "Reaction proxy NRMSE": "reaction_proxy/nrmse",
}

QUALITY_DIRS = {
    "Simplified foot": QUALITY_ROOT / "simplified_foot_report",
    "Anatomical pilot": QUALITY_ROOT / "anatomic_foot_v2_pilot_stable3_1536_w64",
    "V9": QUALITY_ROOT / "anatomic_v9_contact_v1",
    "V10 model-ready": QUALITY_ROOT / "anatomic_v10_contact_v1",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, name: str, appendix: bool = False) -> None:
    out = FIG_APPENDIX if appendix else FIG_MAIN
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": "#6B7280", "lw": 1.4},
        xycoords="axes fraction",
    )


def plot_dataset_lineage() -> None:
    sources = {
        "SFEM": plt.imread(ASSETS / "lineage_sfem_source.png"),
        "Simplified foot": plt.imread(ASSETS / "lineage_simplified_source.png"),
        "V9": plt.imread(ASSETS / "lineage_v9_source.png"),
        "V10": plt.imread(ASSETS / "lineage_v10_source.png"),
    }

    # Fixed pixel crops preserve the supplied FE evidence while removing slide
    # backgrounds, oversized margins, and embedded captions.
    specifications = [
        (
            "fig00a_sfem_meshes",
            "SFEM",
            [(24, 145, 700, 682), (785, 145, 1459, 702)],
            ["(a) Sample geometry family", "(b) Loading, boundary conditions, and stress"],
            (10.8, 4.1),
        ),
        (
            "fig00b_simplified_foot",
            "Simplified foot",
            [(32, 123, 592, 480), (661, 123, 1147, 478), (1225, 123, 1822, 491)],
            ["(a) Exterior mesh", "(b) Material cross-section", "(c) Mechanical response"],
            (11.2, 3.1),
        ),
        (
            "fig00c_v9_mesh",
            "V9",
            [(21, 84, 557, 684), (584, 84, 1100, 676), (1114, 84, 1605, 681)],
            ["(a) Exterior mesh", "(b) Anatomical cross-section", "(c) Mechanical response"],
            (11.2, 4.0),
        ),
        (
            "fig00d_v10_mesh",
            "V10",
            [(43, 88, 647, 745), (672, 88, 1182, 745), (1234, 88, 1813, 745)],
            ["(a) Exterior mesh", "(b) Anatomical cross-section", "(c) Mechanical response"],
            (11.2, 4.0),
        ),
    ]

    for filename, stage, stage_crops, titles, figsize in specifications:
        image = sources[stage]
        widths = [x2 - x1 for x1, _, x2, _ in stage_crops]
        fig, axes = plt.subplots(
            1,
            len(stage_crops),
            figsize=figsize,
            gridspec_kw={"width_ratios": widths, "wspace": 0.035},
        )
        axes = np.atleast_1d(axes)
        for ax, (x1, y1, x2, y2), title in zip(axes, stage_crops, titles):
            ax.imshow(image[y1:y2, x1:x2])
            ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#CBD5E1")
                spine.set_linewidth(0.8)
        fig.tight_layout(pad=0.4)
        save_figure(fig, filename)


def plot_pipeline_schematic() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 7.3))
    ax.set_axis_off()

    palette = {
        "simulation": ("#DCEAF5", "#2878B5"),
        "quality": ("#FDE9D9", "#D97706"),
        "dataset": ("#E5F3EA", "#2E9D65"),
        "model": ("#EEE8F5", "#7B5AA6"),
        "decision": ("#FFF4E5", "#C2410C"),
        "neutral": ("#F3F4F6", "#6B7280"),
    }

    def box(x: float, y: float, text: str, role: str, fontsize: float = 8.0) -> None:
        face, edge = palette[role]
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=fontsize,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": face, "edgecolor": edge, "linewidth": 1.35},
        )

    def diamond(x: float, y: float, text: str) -> None:
        face, edge = palette["decision"]
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.55", "facecolor": face, "edgecolor": edge, "linewidth": 1.5},
        )

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = "#64748B",
              rad: float = 0.0, label: str | None = None, label_xy: tuple[float, float] | None = None) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.35,
                        "connectionstyle": f"arc3,rad={rad}"},
        )
        if label and label_xy:
            ax.text(*label_xy, label, transform=ax.transAxes, fontsize=7.2,
                    fontweight="bold", color=color, ha="center", va="center")

    # Section bands.
    bands = [
        (0.69, 0.24, "A  Simulation campaign", "simulation"),
        (0.37, 0.25, "B  Validation and dataset diagnosis", "quality"),
        (0.04, 0.23, "C  Model-ready learning and evaluation", "model"),
    ]
    for y, height, title, role in bands:
        face, edge = palette[role]
        ax.add_patch(plt.Rectangle((0.015, y), 0.97, height, transform=ax.transAxes,
                                   facecolor=face + "35", edgecolor=edge + "55", linewidth=0.8))
        ax.text(0.035, y + height - 0.028, title, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.2, fontweight="bold", color=edge)

    # A: generation.
    box(0.12, 0.79, "Parameter domain\n+ base templates", "simulation")
    box(0.34, 0.79, "Sample manifest\nparameter draws + base IDs", "simulation")
    box(0.57, 0.79, "FEBio input generation\n+ cluster execution", "simulation")
    box(0.80, 0.79, "Output extraction\nnodes · elements · contact", "simulation")
    for a, b in [(0.20, 0.26), (0.43, 0.49), (0.66, 0.72)]:
        arrow((a, 0.79), (b, 0.79))

    # B: per-sample acceptance, packing, and dataset-level diagnosis.
    box(0.72, 0.57, "Sample checks\nsolver termination · required fields\nfinite values · shapes · dataset ID", "quality", fontsize=7.5)
    diamond(0.90, 0.57, "Valid\nsample?")
    arrow((0.80, 0.73), (0.74, 0.63))
    arrow((0.81, 0.57), (0.85, 0.57))
    box(0.92, 0.43, "Reject + log failure\nclassify cause", "neutral", fontsize=7.4)
    arrow((0.90, 0.52), (0.92, 0.48), color="#B45309", label="No", label_xy=(0.94, 0.51))
    arrow((0.94, 0.40), (0.58, 0.73), color="#B45309", rad=0.28,
          label="revise bounds, template, or solver setup", label_xy=(0.77, 0.655))

    box(0.75, 0.43, "Pack accepted samples\ninto validated shards", "dataset")
    arrow((0.88, 0.53), (0.79, 0.47), color="#2E9D65", label="Yes", label_xy=(0.83, 0.51))
    box(0.51, 0.49, "Dataset diagnostics\nacceptance + base coverage\ntarget distributions · schema/fingerprint\nmean-map + regression baselines", "dataset", fontsize=7.35)
    arrow((0.68, 0.43), (0.59, 0.47), color="#2E9D65")
    diamond(0.29, 0.49, "Useful and\nsufficient dataset?")
    arrow((0.42, 0.49), (0.35, 0.49), color="#2E9D65")
    arrow((0.26, 0.54), (0.12, 0.73), color="#B45309", rad=-0.18,
          label="No: redesign or top-up", label_xy=(0.17, 0.655))

    # C: representation, learning, evaluation, and iteration.
    box(0.24, 0.18, "Model-ready conversion\nregion/face features\ntrain-only normalization\nhistory compression", "dataset", fontsize=7.5)
    arrow((0.29, 0.44), (0.25, 0.24), color="#2E9D65", label="Yes", label_xy=(0.29, 0.34))
    box(0.49, 0.18, "Train FFN · GNO · PINN\nW&B lineage + checkpoints", "model")
    arrow((0.33, 0.18), (0.41, 0.18), color="#7B5AA6")
    box(0.73, 0.18, "Common evaluation\npooled + per-base metrics\nheld-out bases · stability · runtime", "model", fontsize=7.5)
    arrow((0.58, 0.18), (0.65, 0.18), color="#7B5AA6")
    diamond(0.92, 0.18, "Stable and\ninformative?")
    arrow((0.82, 0.18), (0.87, 0.18), color="#7B5AA6")
    arrow((0.90, 0.23), (0.52, 0.24), color="#7B5AA6", rad=0.20,
          label="No: revise architecture, losses, or training", label_xy=(0.72, 0.29))

    ax.text(
        0.5,
        0.965,
        "Iterative FEBio-to-surrogate development workflow",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.012,
        "Provenance spans the full loop: manifests and dataset fingerprints connect accepted samples to W&B run IDs, checkpoint rules, and reported figures.",
        transform=ax.transAxes,
        ha="center",
        color="#4B5563",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.02, right=0.985, top=0.94, bottom=0.05)
    save_figure(fig, "fig00_pipeline")


def plot_model_schematic() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 6.2))
    ax.set_axis_off()

    def box(x: float, y: float, text: str, edge: str, width: float = 0.14,
            face: str = "#FFFFFF", fontsize: float = 8.2) -> None:
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            bbox={
                "boxstyle": f"round,pad=0.55,rounding_size={width / 5}",
                "facecolor": face,
                "edgecolor": edge,
                "linewidth": 1.4,
            },
        )

    lane_y = {"FFN": 0.76, "GNO": 0.48, "PINN": 0.20}
    for method, y in lane_y.items():
        color = COLORS[method]
        ax.add_patch(
            plt.Rectangle(
                (0.01, y - 0.115),
                0.98,
                0.23,
                transform=ax.transAxes,
                facecolor=color + "0B",
                edgecolor=color + "55",
                linewidth=0.8,
                zorder=0,
            )
        )
        ax.text(0.025, y, method, transform=ax.transAxes, rotation=90, ha="center",
                va="center", fontweight="bold", color=color, fontsize=10)

    common = "Per-face sample\n52 global parameters\n+ normalized coordinates\n+ 4-region one-hot code"
    output = "Common output\n1,064 contact-face\npressure values"

    # FFN lane
    y = lane_y["FFN"]
    box(0.14, y, common, COLORS["FFN"], face="#F8FAFC")
    box(0.39, y, "Concatenate global\nand face features", COLORS["FFN"], face="#EAF3FA")
    box(0.62, y, "Pointwise MLP\n5 hidden layers × 192\nSiLU + dropout", COLORS["FFN"], face="#EAF3FA")
    box(0.87, y, output, COLORS["FFN"], face="#F8FAFC")
    for a, b in [(0.23, 0.30), (0.48, 0.53), (0.72, 0.78)]:
        add_arrow(ax, (a, y), (b, y))

    # GNO lane: explicit baseline and graph residual branches.
    y = lane_y["GNO"]
    box(0.14, y, common, COLORS["GNO"], face="#F8FAFC")
    box(0.34, y + 0.055, "Pointwise baseline\npressure", COLORS["GNO"], face="#FDF1E6")
    box(0.34, y - 0.055, "Fixed 6-NN graph\nrelative edge geometry", COLORS["GNO"], face="#FDF1E6")
    box(0.57, y - 0.055, "4 message-passing blocks\nhidden width 160", COLORS["GNO"], face="#FDF1E6")
    box(0.74, y - 0.055, "Bounded graph\nresidual", COLORS["GNO"], face="#FDF1E6")
    box(0.87, y, "Baseline + residual\n→ 1,064 pressures", COLORS["GNO"], face="#F8FAFC")
    add_arrow(ax, (0.23, y), (0.27, y + 0.055))
    add_arrow(ax, (0.23, y), (0.27, y - 0.055))
    add_arrow(ax, (0.43, y - 0.055), (0.49, y - 0.055))
    add_arrow(ax, (0.66, y - 0.055), (0.68, y - 0.055))
    add_arrow(ax, (0.80, y - 0.055), (0.82, y))
    add_arrow(ax, (0.42, y + 0.055), (0.82, y + 0.015))

    # PINN lane: architecture is pressure MLP; extra information acts through training losses.
    y = lane_y["PINN"]
    box(0.14, y, common, COLORS["PINN"], face="#F8FAFC")
    box(0.39, y, "Pressure MLP\n6 hidden layers × 192\nSiLU + dropout", COLORS["PINN"], face="#EAF6F0")
    box(0.62, y, "Pressure prediction\nat contact faces", COLORS["PINN"], face="#EAF6F0")
    box(0.87, y, output, COLORS["PINN"], face="#F8FAFC")
    for a, b in [(0.23, 0.30), (0.48, 0.53), (0.71, 0.78)]:
        add_arrow(ax, (a, y), (b, y))
    ax.text(
        0.62,
        y - 0.087,
        "training only: auxiliary FE fields, histories, and physical consistency losses",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.3,
        color="#166534",
        fontstyle="italic",
    )

    ax.text(
        0.5,
        0.965,
        "Final V10 surrogate architectures",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.035,
        "All three models solve the same supervised pressure task; they differ in spatial inductive bias and training supervision.",
        transform=ax.transAxes,
        ha="center",
        color="#4B5563",
        fontsize=8.5,
    )
    fig.tight_layout()
    save_figure(fig, "fig00_model_schematic")


def plot_pinn_training_flow() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    ax.set_axis_off()

    def box(x: float, y: float, text: str, edge: str, face: str, fontsize: float = 8.0) -> None:
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": face, "edgecolor": edge, "linewidth": 1.3},
        )

    green = COLORS["PINN"]
    blue = COLORS["FFN"]
    orange = COLORS["GNO"]
    grey = "#6B7280"

    fig.suptitle("Hybrid PINN training data and loss flow", fontsize=13, fontweight="bold", y=0.97)

    # Forward prediction path.
    box(0.09, 0.84, "Model inputs\nparameters + contact-face\nfeatures", blue, "#EEF5FB")
    box(0.29, 0.84, "Pressure network\n6 × 192 MLP", green, "#EAF6F0")
    box(0.49, 0.84, "Predicted contact\npressure field", green, "#EAF6F0")
    for a, b in [(0.17, 0.22), (0.37, 0.42)]:
        add_arrow(ax, (a, 0.84), (b, 0.84))

    # Five loss families are kept in separate rows before convergence.
    box(0.70, 0.84, "Primary pressure loss\nSmooth L1", green, "#EAF6F0")
    add_arrow(ax, (0.57, 0.84), (0.63, 0.84))

    box(0.49, 0.64, "Derived quantities\npeak · reaction proxy\nmean/std · centre of pressure", orange, "#FDF1E6")
    box(0.70, 0.64, "Derived contact losses", orange, "#FDF1E6")
    add_arrow(ax, (0.49, 0.78), (0.49, 0.70))
    add_arrow(ax, (0.58, 0.64), (0.63, 0.64))

    box(0.09, 0.45, "Stored FEBio supervision\nnode displacement\nelement stress + von Mises", grey, "#F3F4F6")
    box(0.32, 0.45, "Masked auxiliary losses\n512 nodes · 512 elements", grey, "#F3F4F6")
    add_arrow(ax, (0.17, 0.45), (0.24, 0.45))

    box(0.09, 0.24, "Compressed histories\n3 of 51 stored times\nup to 256 contact faces", grey, "#F3F4F6")
    box(0.32, 0.24, "History supervision\npressure · displacement\nstress · von Mises", grey, "#F3F4F6")
    add_arrow(ax, (0.17, 0.24), (0.24, 0.24))

    box(0.58, 0.31, "Physical consistency losses\nnon-negativity · equilibrium\nconstitutive consistency\ncontact projection/complementarity", green, "#EAF6F0", fontsize=7.5)
    ax.annotate("", xy=(0.58, 0.39), xytext=(0.49, 0.78), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": green, "lw": 1.1})

    box(0.84, 0.48, "Scheduled and adaptive weighting\nwarm-up · ramps · cadence\nReLoBRaLo-style weights + caps", "#7B5AA6", "#F3EFF8")
    convergence = [
        ((0.77, 0.84), (0.78, 0.54)),
        ((0.77, 0.64), (0.78, 0.51)),
        ((0.40, 0.45), (0.77, 0.48)),
        ((0.40, 0.24), (0.78, 0.43)),
        ((0.67, 0.31), (0.78, 0.45)),
    ]
    for start, end in convergence:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                    arrowprops={"arrowstyle": "-|>", "color": "#8B8B8B", "lw": 1.0})

    box(0.95, 0.48, "Weighted\ntotal loss", "#7B5AA6", "#F3EFF8")
    add_arrow(ax, (0.90, 0.48), (0.92, 0.48))
    box(0.84, 0.16, "Back-propagation\nupdates pressure network", green, "#EAF6F0")
    ax.annotate("", xy=(0.84, 0.22), xytext=(0.95, 0.42), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": grey, "lw": 1.2,
                            "connectionstyle": "arc3,rad=-0.15"})
    ax.annotate("", xy=(0.29, 0.78), xytext=(0.76, 0.16), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": green, "lw": 1.3, "connectionstyle": "arc3,rad=-0.22"})

    ax.text(
        0.5,
        0.03,
        "Auxiliary fields and histories supervise training but are not required to produce the final pressure field at inference.",
        transform=ax.transAxes,
        ha="center",
        color="#4B5563",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.08)
    save_figure(fig, "fig00_pinn_training_flow")


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(CONFIG)
    manifest["method"] = manifest["method"].str.upper()
    manifest["include_main"] = manifest["include_main"].astype(str).str.lower().eq("true")
    return manifest


def history_path(project: str, run_id: str) -> Path:
    return WANDB_ROOT / "runs" / project / run_id / "history.jsonl.gz"


def load_history(project: str, run_id: str) -> list[dict]:
    path = history_path(project, run_id)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_histories(manifest: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    audit: list[dict] = []
    for row in manifest.itertuples(index=False):
        rows = load_history(row.project, row.run_id)
        scalar_count = 0
        for hist in rows:
            step = hist.get("_step")
            epoch = hist.get("epoch")
            runtime = hist.get("_runtime")
            for key, value in hist.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    records.append(
                        {
                            "run_id": row.run_id,
                            "project": row.project,
                            "dataset_label": row.dataset_label,
                            "method": row.method,
                            "role": row.role,
                            "selection_status": row.selection_status,
                            "step": step,
                            "epoch": epoch,
                            "runtime_seconds": runtime,
                            "metric": key,
                            "value": float(value),
                        }
                    )
                    scalar_count += 1
        audit.append(
            {
                "run_id": row.run_id,
                "project": row.project,
                "dataset_label": row.dataset_label,
                "method": row.method,
                "history_rows": len(rows),
                "scalar_values": scalar_count,
                "history_file": str(history_path(row.project, row.run_id)),
            }
        )
    pd.DataFrame(audit).to_csv(DERIVED / "availability_audit.csv", index=False)
    result = pd.DataFrame(records)
    result.to_csv(DERIVED / "run_metrics_long.csv", index=False)
    return result


def last_metric(metrics: pd.DataFrame, run_id: str, key: str) -> float:
    subset = metrics[(metrics.run_id == run_id) & (metrics.metric == key)].sort_values("step")
    return float(subset.value.iloc[-1]) if not subset.empty else np.nan


def metric_series(metrics: pd.DataFrame, run_id: str, key: str) -> pd.DataFrame:
    return metrics[(metrics.run_id == run_id) & (metrics.metric == key)].sort_values("step")


def select_checkpoint_row(project: str, run_id: str) -> tuple[dict, str]:
    rows = load_history(project, run_id)
    candidates = [
        row for row in rows
        if isinstance(row.get("val/pooled/pressure/nrmse"), (int, float))
        and math.isfinite(row["val/pooled/pressure/nrmse"])
    ]
    if candidates:
        return min(candidates, key=lambda row: row["val/pooled/pressure/nrmse"]), "minimum_val_pressure_nrmse"
    best_rows = [
        row for row in rows
        if isinstance(row.get("best_val/pooled/pressure/nrmse"), (int, float))
        and math.isfinite(row["best_val/pooled/pressure/nrmse"])
    ]
    if best_rows:
        return best_rows[-1], "exported_best_val_fallback"
    return (rows[-1] if rows else {}), "final_available_row"


def selected_value(row: dict, prefix: str, suffix: str) -> float:
    preferred = prefix + suffix
    value = row.get(preferred)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    fallback = preferred.replace("best_val/", "val/", 1)
    value = row.get(fallback)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else np.nan


def load_quality() -> pd.DataFrame:
    rows: list[dict] = []
    regression_rows: list[dict] = []
    for label, folder in QUALITY_DIRS.items():
        summary_path = folder / "summary.json"
        baseline_path = folder / "regression_baselines.json"
        baseline_long_path = folder / "regression_baselines_long.csv"
        counts_path = folder / "valid_counts_by_base.csv"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "dataset_label": label,
            "attempted": summary.get("attempted_samples_from_json"),
            "valid": summary.get("valid_samples_in_shards"),
            "success_rate": summary.get("apparent_success_rate"),
            "failed": summary.get("failed_samples_not_packed"),
            "shards": summary.get("shards_found"),
        }
        rows.append(row)
        if counts_path.exists():
            counts = pd.read_csv(counts_path)
            counts["dataset_label"] = label
            counts.to_csv(DERIVED / f"quality_counts_{slug(label)}.csv", index=False)
        if baseline_long_path.exists():
            baseline_long = pd.read_csv(baseline_long_path)
            baseline_long.insert(0, "dataset_label", label)
            regression_rows.extend(baseline_long.to_dict("records"))
        elif baseline_path.exists():
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
            for split_name, models in data.get("baselines", {}).items():
                for model_name, values in models.items():
                    if isinstance(values, dict):
                        regression_rows.append(
                            {
                                "dataset_label": label,
                                "split": split_name,
                                "baseline": model_name,
                                "pressure_r2": values.get("pressure_r2"),
                                "pressure_rmse": values.get("pressure_rmse"),
                                "peak_pressure_r2": values.get("peak_pressure_r2"),
                            }
                        )
    quality = pd.DataFrame(rows)
    regression = pd.DataFrame(regression_rows)
    quality.to_csv(DERIVED / "dataset_quality_summary.csv", index=False)
    regression.to_csv(DERIVED / "regression_baselines_long.csv", index=False)
    if SFEM_REGRESSION_LONG.exists():
        sfem_regression = pd.read_csv(SFEM_REGRESSION_LONG)
        sfem_regression.to_csv(DERIVED / "sfem_regression_baselines_long.csv", index=False)
    return quality


def slug(value: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in value).strip("_")


def plot_dataset_quality(quality: pd.DataFrame) -> None:
    if quality.empty:
        return
    q = quality.copy()
    q["dataset_label"] = pd.Categorical(q.dataset_label, DATASET_ORDER, ordered=True)
    q = q.sort_values("dataset_label")
    q.to_csv(DERIVED / "figure_01_dataset_quality.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    x = np.arange(len(q))
    display_labels = [
        "Anatomical pilot\n(generator only)" if value == "Anatomical pilot" else value
        for value in q["dataset_label"].astype(str)
    ]
    campaign_colors = ["#9CA3AF" if value == "Anatomical pilot" else "#4C78A8"
                       for value in q["dataset_label"].astype(str)]
    point_colors = ["#9CA3AF" if value == "Anatomical pilot" else "#2E9D65"
                    for value in q["dataset_label"].astype(str)]
    axes[0].bar(x, q["valid"], color=campaign_colors)
    axes[0].set_ylabel("Valid samples")
    axes[0].set_xticks(x, display_labels, rotation=25, ha="right")
    axes[0].set_title("(a) Accepted dataset size")
    axes[1].scatter(x, 100 * q["success_rate"], s=55, color=point_colors, zorder=3)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Apparent success rate (%)")
    axes[1].set_xticks(x, display_labels, rotation=25, ha="right")
    axes[1].set_title("(b) FEBio generation reliability")
    fig.tight_layout()
    save_figure(fig, "fig01_dataset_quality")


def copy_checkpoint_reselection_outputs() -> None:
    copy_map = {
        RESELECTED_RAW / "unified_checkpoints_foot.csv": "unified_checkpoints_foot.csv",
        RESELECTED_RAW / "unified_checkpoint_metrics_foot_long.csv": "unified_checkpoint_metrics_foot_long.csv",
        RESELECTED_RAW / "unified_checkpoints_sfem_all.csv": "unified_checkpoints_sfem_raw_all.csv",
        RESELECTED_RAW / "unified_checkpoints_sfem_best_by_method.csv": "unified_checkpoints_sfem_raw_best_by_method.csv",
        RESELECTED_RAW / "unified_checkpoint_metrics_sfem_long.csv": "unified_checkpoint_metrics_sfem_raw_long.csv",
        RESELECTED_NORM / "unified_checkpoints_sfem_all.csv": "unified_checkpoints_sfem_norm_all.csv",
        RESELECTED_NORM / "unified_checkpoints_sfem_best_by_method.csv": "unified_checkpoints_sfem_norm_best_by_method.csv",
        RESELECTED_NORM / "unified_checkpoint_metrics_sfem_long.csv": "unified_checkpoint_metrics_sfem_norm_long.csv",
    }
    for src, dest_name in copy_map.items():
        if src.exists():
            pd.read_csv(src).to_csv(DERIVED / dest_name, index=False)


def plot_febio_regression_baselines() -> None:
    path = DERIVED / "regression_baselines_long.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    if "base" in data:
        data = data[data["base"].isna()].copy()
    data = data[data["dataset_label"].isin(["Simplified foot", "Anatomical pilot", "V9", "V10 model-ready"])]
    data = data.dropna(subset=["pressure_nrmse"])
    if data.empty:
        return
    data["split_label"] = data["split"].map(
        {
            "random_seen_base_split": "Represented/random split",
            "unseen_base_10_11_split": "Held-out bases 10-11",
        }
    ).fillna(data["split"])
    data["baseline_label"] = data["baseline"].map(REGRESSION_BASELINE_LABELS).fillna(data["baseline"])
    data["dataset_label"] = pd.Categorical(data["dataset_label"], ["Simplified foot", "Anatomical pilot", "V9", "V10 model-ready"], ordered=True)
    data.to_csv(DERIVED / "figure_08_febio_regression_baselines.csv", index=False)

    splits = ["Represented/random split", "Held-out bases 10-11"]
    baselines = ["Mean map", "Ridge all", "Ridge no base ID", "Ridge physical"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)
    width = 0.18
    x_labels = ["Simplified foot", "Anatomical pilot", "V9", "V10 model-ready"]
    x = np.arange(len(x_labels))
    colors = ["#8A8A8A", "#7B5AA6", "#A57BC2", "#4C78A8"]
    for ax, split_label in zip(axes, splits):
        sub = data[data["split_label"] == split_label]
        for i, baseline in enumerate(baselines):
            vals = []
            for dataset in x_labels:
                row = sub[(sub["dataset_label"].astype(str) == dataset) & (sub["baseline_label"] == baseline)]
                vals.append(float(row["pressure_nrmse"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width=width, label=baseline, color=colors[i])
        ax.set_xticks(x, x_labels, rotation=25, ha="right")
        ax.set_title(split_label)
        ax.set_ylabel("Pressure NRMSE")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Linear FEBio baselines reveal dataset difficulty and simple-structure predictability", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig08_febio_regression_baselines")


def plot_sfem_regression_baselines() -> None:
    path = DERIVED / "sfem_regression_baselines_long.csv"
    if not path.exists():
        return
    data = pd.read_csv(path).dropna(subset=["r2"]).copy()
    if data.empty:
        return
    data["baseline_label"] = data["baseline"].map(REGRESSION_BASELINE_LABELS).fillna(data["baseline"])
    data["target_label"] = data["target"].map({"von_mises": "Raw von Mises", "von_mises_norm": "Normalized von Mises"})
    data["split_label"] = data["split"].map({"val": "Validation", "report_holdout": "Report holdout"}).fillna(data["split"])
    data.to_csv(DERIVED / "figure_09_sfem_regression_baselines.csv", index=False)

    targets = ["Raw von Mises", "Normalized von Mises"]
    splits = ["Validation", "Report holdout"]
    baseline_order = ["Mean scalar", "Ridge load", "Ridge node geom.", "Ridge node+global"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), sharey=False)
    width = 0.18
    x = np.arange(len(splits))
    colors = ["#8A8A8A", "#D97706", "#4C78A8", "#7B5AA6"]
    for ax, target in zip(axes, targets):
        sub = data[data["target_label"] == target]
        for i, baseline in enumerate(baseline_order):
            vals = []
            for split in splits:
                row = sub[(sub["split_label"] == split) & (sub["baseline_label"] == baseline)]
                vals.append(float(row["r2"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width=width, label=baseline, color=colors[i])
        ax.axhline(0, color="#6B7280", lw=0.8)
        ax.set_xticks(x, splits)
        ax.set_title(target)
        ax.set_ylabel("$R^2$")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("SFEM node-level baselines separate raw and normalized stress structure", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig09_sfem_regression_baselines")


def build_headline_table(metrics: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    records = []
    checkpoints = []
    for run in manifest[manifest.include_main].itertuples(index=False):
        checkpoint, rule = select_checkpoint_row(run.project, run.run_id)
        checkpoints.append(
            {
                "run_id": run.run_id,
                "dataset_label": run.dataset_label,
                "method": run.method,
                "selection_rule": rule,
                "step": checkpoint.get("_step"),
                "epoch": checkpoint.get("epoch"),
                "runtime_seconds": checkpoint.get("_runtime"),
            }
        )
        for display, suffix in HEADLINE_KEYS.items():
            for split, prefix in [
                ("Pooled", "best_val/pooled/"),
                ("Seen bases", "best_val/seen_bases_00_09/"),
                ("Unseen bases", "best_val/unseen_bases_10_11/"),
            ]:
                records.append(
                    {
                        "run_id": run.run_id,
                        "dataset_label": run.dataset_label,
                        "method": run.method,
                        "role": run.role,
                        "split": split,
                        "metric_label": display,
                        "metric_key": prefix + suffix,
                        "value": selected_value(checkpoint, prefix, suffix),
                    }
                )
    table = pd.DataFrame(records)
    table.to_csv(DERIVED / "headline_metrics.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(DERIVED / "selected_checkpoints.csv", index=False)
    return table


def plot_cross_dataset(headline: pd.DataFrame) -> None:
    rows = [
        ("SFEM stress", "FFN", -5.501, "initial baseline"),
        ("SFEM stress", "GNO", 0.634, "finished graph run"),
        ("SFEM stress", "PINN", 0.817, "later specialist"),
        ("Simplified foot", "FFN", 0.204, "pressure"),
        ("Simplified foot", "GNO", 0.697, "pressure"),
        ("Simplified foot", "PINN", 0.594, "pressure"),
        ("V9", "FFN", 0.680, "pressure"),
        ("V9", "GNO", 0.674, "pressure"),
        ("V9", "PINN", 0.666, "pressure"),
        ("V10", "Mean map", 0.451, "pressure baseline"),
        ("V10", "Ridge", 0.461, "pressure baseline"),
        ("V10", "FFN", 0.444, "pressure"),
        ("V10", "GNO", 0.446, "pressure"),
        ("V10", "PINN", 0.453, "pressure"),
    ]
    data = pd.DataFrame(rows, columns=["dataset_label", "method", "r2", "comparison_role"])
    data.to_csv(DERIVED / "figure_02_cross_dataset_ranking.csv", index=False)
    datasets = ["SFEM stress", "Simplified foot", "V9", "V10"]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.8))
    for ax, dataset in zip(axes, datasets):
        sub = data[data.dataset_label == dataset]
        x = np.arange(len(sub))
        ax.bar(x, sub.r2, color=[COLORS[m] for m in sub.method], width=0.68)
        ax.axhline(0, color="#6B7280", lw=0.8)
        ax.set_xticks(x, sub.method, rotation=45, ha="right")
        ax.set_title(dataset)
        ax.set_ylim(min(-0.35, sub.r2.min() * 1.06), max(1.05, sub.r2.max() * 1.10))
        for xi, value in zip(x, sub.r2):
            ax.text(xi, value + (0.035 if value >= 0 else -0.10), f"{value:.2f}",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
        ax.set_ylabel("$R^2$" if dataset == datasets[0] else "")
    axes[0].text(0.5, -0.34, "Different target;\nlater PINN specialist", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=7, color="#4B5563")
    fig.suptitle("Architecture ranking changed across the trained dataset lineages", y=1.01)
    fig.text(0.5, 0.01, "Compare methods within a panel only; SFEM predicts stress, whereas the FEBio datasets predict contact pressure.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig, "fig02_cross_dataset_ranking")


def plot_final_benchmark(headline: pd.DataFrame) -> None:
    r2_seen = {"Mean map": 0.451, "Ridge": 0.461, "FFN": 0.453, "GNO": 0.447, "PINN": 0.450}
    r2_unseen = {"Mean map": 0.466, "Ridge": 0.480, "FFN": 0.439, "GNO": 0.448, "PINN": 0.454}
    pooled_nrmse = {"FFN": 0.13624, "GNO": 0.13573, "PINN": 0.13511}
    runtime_min = {"FFN": 1637 / 60, "GNO": 1495 / 60, "PINN": 2663 / 60}
    rows = []
    for method in FULL_METHOD_ORDER:
        rows.extend([
            {"panel": "Represented-base $R^2$", "method": method, "value": r2_seen[method]},
            {"panel": "Held-out-base $R^2$", "method": method, "value": r2_unseen[method]},
        ])
    for method in METHOD_ORDER:
        rows.extend([
            {"panel": "Pooled pressure NRMSE", "method": method, "value": pooled_nrmse[method]},
            {"panel": "Checkpoint runtime (min)", "method": method, "value": runtime_min[method]},
        ])
    pd.DataFrame(rows).to_csv(DERIVED / "figure_03_final_benchmark.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.8))
    panel_data = [
        ("Represented-base $R^2$", r2_seen, FULL_METHOD_ORDER),
        ("Held-out-base $R^2$", r2_unseen, FULL_METHOD_ORDER),
        ("Pooled pressure NRMSE", pooled_nrmse, METHOD_ORDER),
        ("Checkpoint runtime (min)", runtime_min, METHOD_ORDER),
    ]
    for ax, (title, values, order) in zip(axes, panel_data):
        x = np.arange(len(order))
        bars = ax.bar(x, [values[m] for m in order], color=[COLORS[m] for m in order], width=0.7)
        ax.set_xticks(x, order, rotation=45, ha="right")
        ax.set_title(title)
        ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=7)
        if "$R^2$" in title:
            ax.set_ylim(0.40, 0.50)
        elif "NRMSE" in title:
            ax.set_ylim(0.13, 0.14)
        else:
            ax.set_ylim(0, 240)
    fig.suptitle("V10: similar neural accuracy, competitive simple baselines, and unequal computational cost", y=1.01)
    fig.tight_layout()
    save_figure(fig, "fig03_final_benchmark")


def plot_per_base_heatmap(metrics: pd.DataFrame, manifest: pd.DataFrame) -> None:
    final_runs = manifest[(manifest.dataset_label == "V10 model-ready") & manifest.include_main]
    records = []
    for run in final_runs.itertuples(index=False):
        checkpoint, _ = select_checkpoint_row(run.project, run.run_id)
        for base in range(12):
            suffix = f"by_base/base_{base:02d}/pressure/nrmse"
            records.append(
                {
                    "run_id": run.run_id,
                    "method": run.method,
                    "base": base,
                    "value": selected_value(checkpoint, "best_val/", suffix),
                }
            )
    data = pd.DataFrame(records)
    data.to_csv(DERIVED / "figure_04_per_base_nrmse.csv", index=False)
    matrix = data.pivot(index="method", columns="base", values="value").reindex(METHOD_ORDER)
    if matrix.dropna(how="all").empty:
        return
    fig, ax = plt.subplots(figsize=(9.2, 2.7))
    sns.heatmap(matrix, cmap="mako_r", annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Pressure NRMSE"}, ax=ax)
    ax.axvline(10, color="white", lw=3)
    ax.text(10.98, -0.28, "held out", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Base model")
    ax.set_ylabel("")
    ax.set_title("V10 pressure error by base geometry")
    fig.tight_layout()
    save_figure(fig, "fig04_per_base_generalization")


def plot_generalization_gap(headline: pd.DataFrame) -> None:
    data = headline[
        (headline.dataset_label == "V10 model-ready")
        & (headline.metric_label == "Pressure NRMSE")
        & (headline.split.isin(["Seen bases", "Unseen bases"]))
    ].dropna(subset=["value"])
    data.to_csv(DERIVED / "figure_05_generalization_gap.csv", index=False)
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for method in METHOD_ORDER:
        sub = data[data.method == method].set_index("split")
        vals = [sub.value.get("Seen bases", np.nan), sub.value.get("Unseen bases", np.nan)]
        ax.plot([0, 1], vals, marker="o", lw=2, color=COLORS[method], label=method)
    ax.set_xticks([0, 1], ["Represented bases", "Held-out bases"])
    ax.set_ylabel("Pressure NRMSE")
    ax.set_title("Generalization gap on V10")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig05_generalization_gap")


def plot_training_stability(metrics: pd.DataFrame, manifest: pd.DataFrame) -> None:
    final_runs = manifest[(manifest.dataset_label == "V10 model-ready") & manifest.include_main]
    source = []
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=False)
    for ax, method in zip(axes, METHOD_ORDER):
        selected = final_runs[final_runs.method == method]
        if selected.empty:
            continue
        run_id = selected.iloc[0].run_id
        for split, key, style in [
            ("Pooled", "val/pooled/pressure/r2", "-"),
            ("Unseen bases", "val/unseen_bases_10_11/pressure/r2", "--"),
        ]:
            series = metric_series(metrics, run_id, key)
            if series.empty:
                continue
            source.append(series.assign(method=method, split=split)[["run_id", "method", "split", "step", "epoch", "value"]])
            ax.plot(series["epoch"].fillna(series["step"]), series.value, style, color=COLORS[method], alpha=0.8, label=split)
        ax.axhline(0, color="#999999", lw=0.7)
        ax.set_ylabel("$R^2$")
        ax.set_title(f"{method} ({run_id})", loc="left")
        ax.legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("V10 pressure validation dynamics", y=0.995)
    fig.tight_layout()
    if source:
        pd.concat(source).to_csv(DERIVED / "figure_06_training_stability.csv", index=False)
    save_figure(fig, "fig06_training_stability")


def plot_pinn_loss_governance(metrics: pd.DataFrame, manifest: pd.DataFrame) -> None:
    selected = manifest[(manifest.dataset_label == "V9 model-ready") & (manifest.method == "PINN")]
    if selected.empty:
        return
    run_id = selected.iloc[0].run_id
    candidate_groups = {
        "Pressure": ["train/pressure", "train/loss_pressure"],
        "Equilibrium": ["train/equilibrium"],
        "Contact": ["train/contact_projection", "train/contact_complementarity"],
        "Node auxiliary": ["train/node_displacement_aux"],
        "History": ["train/history_contact_pressure", "train/history_node_aux", "train/history_element_aux"],
    }
    available = set(metrics[metrics.run_id == run_id].metric.unique())
    chosen = {}
    for label, candidates in candidate_groups.items():
        for key in candidates:
            if key in available:
                chosen[label] = key
                break
    weight_keys = sorted(k for k in available if k.startswith("train/robalrs_weight/"))
    weight_keys = [k for k in weight_keys if any(x in k for x in ["pressure", "equilibrium", "contact", "history"])][:6]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.2), sharex=True)
    val = metric_series(metrics, run_id, "val/pooled/pressure/r2")
    if not val.empty:
        axes[0].plot(val.epoch.fillna(val.step), val.value, color=COLORS["PINN"])
    axes[0].set_ylabel("Pressure $R^2$")
    axes[0].set_title(f"PINN loss governance ({run_id})", loc="left")
    source = []
    for label, key in chosen.items():
        series = metric_series(metrics, run_id, key)
        if not series.empty:
            axes[1].plot(series.epoch.fillna(series.step), series.value, label=label)
            source.append(series.assign(panel="loss", display=label)[["run_id", "panel", "display", "metric", "step", "epoch", "value"]])
    axes[1].set_yscale("symlog", linthresh=1e-6)
    axes[1].set_ylabel("Training loss")
    axes[1].legend(frameon=False, ncol=2)
    for key in weight_keys:
        series = metric_series(metrics, run_id, key)
        if not series.empty:
            label = key.split("/")[-1].replace("_", " ")
            axes[2].plot(series.epoch.fillna(series.step), series.value, label=label)
            source.append(series.assign(panel="weight", display=label)[["run_id", "panel", "display", "metric", "step", "epoch", "value"]])
    axes[2].set_ylabel("Adaptive weight")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(frameon=False, ncol=2)
    fig.tight_layout()
    if source:
        pd.concat(source).to_csv(DERIVED / "figure_07_pinn_loss_governance.csv", index=False)
    save_figure(fig, "fig07_pinn_loss_governance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report result tables and figures.")
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Optional self-contained report folder. Figures go here and CSVs go under derived/.",
    )
    parser.add_argument("--derived-dir", type=Path, default=None)
    parser.add_argument("--fig-main-dir", type=Path, default=None)
    parser.add_argument("--fig-appendix-dir", type=Path, default=None)
    parser.add_argument("--sfem-regression-long", type=Path, default=SFEM_REGRESSION_LONG)
    return parser.parse_args()


def configure_outputs(args: argparse.Namespace) -> None:
    global DERIVED, FIG_MAIN, FIG_APPENDIX, SFEM_REGRESSION_LONG
    if args.report_root is not None:
        report_root = args.report_root.resolve()
        DERIVED = report_root / "derived"
        FIG_MAIN = report_root
        FIG_APPENDIX = report_root / "appendix_figures"
    if args.derived_dir is not None:
        DERIVED = args.derived_dir.resolve()
    if args.fig_main_dir is not None:
        FIG_MAIN = args.fig_main_dir.resolve()
    if args.fig_appendix_dir is not None:
        FIG_APPENDIX = args.fig_appendix_dir.resolve()
    SFEM_REGRESSION_LONG = args.sfem_regression_long.resolve()


def main() -> None:
    args = parse_args()
    configure_outputs(args)
    configure_style()
    DERIVED.mkdir(parents=True, exist_ok=True)
    FIG_MAIN.mkdir(parents=True, exist_ok=True)
    FIG_APPENDIX.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    metrics = normalize_histories(manifest)
    quality = load_quality()
    headline = build_headline_table(metrics, manifest)
    plot_dataset_lineage()
    plot_pipeline_schematic()
    plot_model_schematic()
    plot_pinn_training_flow()
    plot_dataset_quality(quality)
    copy_checkpoint_reselection_outputs()
    plot_febio_regression_baselines()
    plot_sfem_regression_baselines()
    plot_cross_dataset(headline)
    plot_final_benchmark(headline)
    plot_per_base_heatmap(metrics, manifest)
    plot_generalization_gap(headline)
    plot_training_stability(metrics, manifest)
    plot_pinn_loss_governance(metrics, manifest)
    print(f"Wrote derived tables to {DERIVED}")
    print(f"Wrote main figures to {FIG_MAIN}")


if __name__ == "__main__":
    main()
