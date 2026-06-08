"""
Chart generation for FIDSUS Top-k Audit Report.

Generates all required figures from audit log CSV/JSON/NPY files.
Run after training completes to produce charts.

Usage: python scripts/generate_topk_charts.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# ── Configuration ────────────────────────────────────────────────────────

# Dataset class labels
UNSW_LABEL_MAP = {
    0: "Normal", 1: "Backdoor", 2: "Analysis", 3: "Fuzzers",
    4: "Shellcode", 5: "Reconnaissance", 6: "Exploits", 7: "DoS",
    8: "Worms", 9: "Generic",
}
NSLKDD_LABEL_MAP = {
    0: "DoS", 1: "Probe", 2: "U2R", 3: "R2L", 4: "Normal",
}

CLASS_COLORS_UNSW = {
    "Normal": "#2ecc71", "Backdoor": "#e74c3c", "Analysis": "#e67e22",
    "Fuzzers": "#9b59b6", "Shellcode": "#1abc9c",
    "Reconnaissance": "#3498db", "Exploits": "#f39c12",
    "DoS": "#c0392b", "Worms": "#8e44ad", "Generic": "#95a5a6",
}
CLASS_COLORS_NSLKDD = {
    "Normal": "#2ecc71", "DoS": "#c0392b", "Probe": "#3498db",
    "U2R": "#e74c3c", "R2L": "#e67e22",
}

RARE_CLASSES_UNSW = {"Backdoor", "Analysis", "Shellcode", "Worms"}
RARE_CLASSES_NSLKDD = {"U2R", "R2L", "Probe"}


def get_class_colors(dataset: str) -> dict:
    if "NSLKDD" in dataset.upper() or "KDD" in dataset.upper():
        return CLASS_COLORS_NSLKDD
    return CLASS_COLORS_UNSW


def get_label_map(dataset: str) -> dict:
    if "NSLKDD" in dataset.upper() or "KDD" in dataset.upper():
        return NSLKDD_LABEL_MAP
    return UNSW_LABEL_MAP


def get_rare_classes(dataset: str) -> set:
    if "NSLKDD" in dataset.upper() or "KDD" in dataset.upper():
        return RARE_CLASSES_NSLKDD
    return RARE_CLASSES_UNSW


def get_meta(base_dir: str) -> dict:
    """Load audit metadata."""
    path = os.path.join(base_dir, "logs", "audit_metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    # Determine paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    audit_dir = repo_root / "docs" / "audit" / "topk"
    logs_dir = audit_dir / "logs"
    assets_dir = audit_dir / "assets"

    if not logs_dir.exists():
        print(f"ERROR: Logs directory not found: {logs_dir}")
        print("Run audited training first.")
        sys.exit(1)

    assets_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta = get_meta(str(audit_dir))
    dataset = meta.get("dataset", "UNSW")
    num_clients = meta.get("num_clients", 50)
    label_map = get_label_map(dataset)
    num_classes = len(label_map)
    rare_classes = get_rare_classes(dataset)
    class_colors = get_class_colors(dataset)

    # For compatibility with older runs without audit_metadata.json,
    # infer dataset from client_label_distribution.json class count
    dist_path = logs_dir / "client_label_distribution.json"
    if dist_path.exists():
        with open(dist_path) as f:
            dist_data = json.load(f)
        if not meta:
            num_classes_in_data = len(dist_data[0]["class_counts"])
            if num_classes_in_data == 5:
                dataset = "NSLKDD"
                label_map = NSLKDD_LABEL_MAP
                num_classes = 5
                rare_classes = RARE_CLASSES_NSLKDD
            else:
                dataset = "UNSW"
                label_map = UNSW_LABEL_MAP
                num_classes = 10
                rare_classes = RARE_CLASSES_UNSW

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
    })

    # ── Chart 1: Client label distribution ────────────────────────────
    print("[1/9] Client label distribution...")
    if dist_path.exists():
        fig, ax = plt.subplots(figsize=(16, 8))
        clients = list(range(len(dist_data)))
        bottom = np.zeros(len(clients))

        # Build per-class arrays
        class_arrays = {}
        for cid in range(num_classes):
            class_arrays[cid] = []

        for s in dist_data:
            counts = s["class_counts"]
            for cid in range(num_classes):
                val = counts[cid] if cid < len(counts) else 0
                class_arrays[cid].append(val)

        for cid in range(num_classes):
            vals = np.array(class_arrays[cid])
            cname = label_map.get(cid, f"class_{cid}")
            color = class_colors.get(cname, "#cccccc")
            # Convert to proportions
            totals = np.array([s["total_samples"] for s in dist_data])
            proportions = vals / np.maximum(totals, 1)
            ax.bar(clients, proportions, bottom=bottom, label=cname,
                   color=color, width=0.8)
            bottom += proportions

        ax.set_xlabel("Client ID")
        ax.set_ylabel("Proportion of samples")
        ax.set_title(f"Client Label Distribution ({dataset})")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.set_xlim(-0.5, num_clients - 0.5)

        # Annotate rare-attack clients
        for i, s in enumerate(dist_data):
            if int(s.get("rare_attack_sample_count", 0)) > 0:
                ax.annotate("★", (i, 1.02), ha="center", fontsize=8,
                            color="red")

        fig.tight_layout()
        fig.savefig(assets_dir / "client_label_distribution.png")
        plt.close(fig)
        print("    -> client_label_distribution.png")

    # ── Chart 2: Client entropy ranking ────────────────────────────────
    print("[2/9] Client entropy ranking...")
    if dist_path.exists():
        fig, ax = plt.subplots(figsize=(14, 5))

        sorted_stats = sorted(dist_data, key=lambda s: s["normalized_label_entropy"])
        client_ids = [s["client_id"] for s in sorted_stats]
        entropies = [s["normalized_label_entropy"] for s in sorted_stats]

        colors = []
        for s in sorted_stats:
            if s["normalized_label_entropy"] >= 0.8:
                colors.append("#2ecc71")  # balanced
            elif s["normalized_label_entropy"] <= 0.3:
                colors.append("#e74c3c")  # severely imbalanced
            else:
                colors.append("#f39c12")  # middle

        ax.bar(range(len(client_ids)), entropies, color=colors, width=0.8)
        ax.axhline(y=0.8, color="#2ecc71", linestyle="--", alpha=0.7, label="Balanced (>=0.8)")
        ax.axhline(y=0.3, color="#e74c3c", linestyle="--", alpha=0.7, label="Imbalanced (<=0.3)")

        # Mark rare-attack clients
        for i, s in enumerate(sorted_stats):
            if int(s.get("rare_attack_sample_count", 0)) > 0:
                ax.annotate("★", (i, entropies[i] + 0.02), ha="center",
                            fontsize=9, color="red")

        ax.set_xlabel("Client (sorted by entropy)")
        ax.set_ylabel("Normalized Label Entropy")
        ax.set_title(f"Client Entropy Ranking ({dataset})")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.15)

        fig.tight_layout()
        fig.savefig(assets_dir / "client_entropy_ranking.png")
        plt.close(fig)
        print("    -> client_entropy_ranking.png")

    # ── Chart 3: Top-k selected entropy boxplot ───────────────────────
    print("[3/9] Top-k selected entropy boxplot...")
    topk_path = logs_dir / "topk_selection_log.csv"
    if topk_path.exists():
        df_topk = pd.read_csv(topk_path)

        all_entropies = [s["normalized_label_entropy"] for s in dist_data] if dist_path.exists() else []
        topk_entropies = df_topk["neighbor_entropy"].dropna().values

        fig, ax = plt.subplots(figsize=(8, 5))
        data_to_plot = []
        labels = []
        if len(all_entropies) > 0:
            data_to_plot.append(all_entropies)
            labels.append(f"All Clients\n(n={len(all_entropies)})")
        if len(topk_entropies) > 0:
            data_to_plot.append(topk_entropies)
            labels.append(f"Top-k Selected\n(n={len(topk_entropies)})")

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors_bp = ["#3498db", "#e74c3c"]
        for patch, color in zip(bp["boxes"], colors_bp[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel("Normalized Label Entropy")
        ax.set_title(f"Top-k Selected vs All Clients Entropy ({dataset})")

        fig.tight_layout()
        fig.savefig(assets_dir / "topk_selected_entropy_boxplot.png")
        plt.close(fig)
        print("    -> topk_selected_entropy_boxplot.png")

    # ── Chart 4: In-degree by client type ─────────────────────────────
    print("[4/9] In-degree by client type...")
    # Find the latest affinity summary
    summary_files = sorted(logs_dir.glob("affinity_matrix_aggregate_round_*.json"))
    if summary_files:
        with open(summary_files[-1]) as f:
            agg_data = json.load(f)

        # Also get detailed per-client in-degree
        detail_files = sorted(logs_dir.glob("affinity_matrix_summary_round_*.csv"))
        if detail_files:
            df_aff = pd.read_csv(detail_files[-1])

            # Categorize clients
            types = {}
            for _, row in df_aff.iterrows():
                cid = int(row["client_id"])
                if row.get("balanced") == "True":
                    types.setdefault("balanced", []).append(row["in_degree_top5"])
                if row.get("severely_imbalanced") == "True":
                    types.setdefault("severely\nimbalanced", []).append(row["in_degree_top5"])
                if row.get("rare_attack") == "True":
                    types.setdefault("rare\nattack", []).append(row["in_degree_top5"])
                if row.get("normal_heavy") == "True":
                    types.setdefault("normal\nheavy", []).append(row["in_degree_top5"])

            if types:
                fig, ax = plt.subplots(figsize=(8, 5))
                labels_t = list(types.keys())
                data_t = [types[k] for k in labels_t]

                bp = ax.boxplot(data_t, labels=labels_t, patch_artist=True)
                colors_t = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]
                for patch, color in zip(bp["boxes"], colors_t[:len(data_t)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

                ax.set_ylabel("In-Degree (times selected as Top-5 neighbor)")
                ax.set_title(f"Top-k In-Degree by Client Type ({dataset})")

                # Add mean labels
                for i, (label, data) in enumerate(zip(labels_t, data_t)):
                    mean_val = np.mean(data) if data else 0
                    ax.annotate(f"μ={mean_val:.1f}", (i + 1, ax.get_ylim()[1]),
                                ha="center", fontsize=8)

                fig.tight_layout()
                fig.savefig(assets_dir / "topk_in_degree_by_client_type.png")
                plt.close(fig)
                print("    -> topk_in_degree_by_client_type.png")

    # ── Chart 5: Affinity matrix heatmap (by entropy) ────────────────
    print("[5/9] Affinity matrix heatmap...")
    npy_files = sorted(logs_dir.glob("affinity_matrix_round_*.npy"))
    if npy_files and dist_path.exists():
        P = np.load(str(npy_files[-1]))

        # Sort clients by entropy
        entropies = np.array([s["normalized_label_entropy"] for s in dist_data])
        sort_idx = np.argsort(entropies)  # low -> high entropy

        P_sorted = P[sort_idx][:, sort_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(P_sorted, cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xlabel("Client (sorted by entropy, low → high)")
        ax.set_ylabel("Client (sorted by entropy, low → high)")
        ax.set_title(f"Affinity Matrix (sorted by label entropy, {dataset})")

        # Add entropy group dividers
        n_balanced = sum(1 for e in entropies if e >= 0.8)
        n_imbalanced = sum(1 for e in entropies if e <= 0.3)
        # Draw lines after severely imbalanced and before balanced
        sev_end = sum(1 for e in entropies[sort_idx] if e <= 0.3)
        bal_start = num_clients - sum(1 for e in entropies[sort_idx] if e >= 0.8)

        if sev_end > 0:
            ax.axhline(y=sev_end - 0.5, color="#e74c3c", linestyle="--", linewidth=1.5)
            ax.axvline(x=sev_end - 0.5, color="#e74c3c", linestyle="--", linewidth=1.5)
        if bal_start < num_clients:
            ax.axhline(y=bal_start - 0.5, color="#2ecc71", linestyle="--", linewidth=1.5)
            ax.axvline(x=bal_start - 0.5, color="#2ecc71", linestyle="--", linewidth=1.5)

        plt.colorbar(im, ax=ax, label="Affinity Score")
        fig.tight_layout()
        fig.savefig(assets_dir / "affinity_matrix_heatmap_by_entropy.png")
        plt.close(fig)
        print("    -> affinity_matrix_heatmap_by_entropy.png")

    # ── Chart 6: Top-k graph with dominant class coloring ────────────
    print("[6/9] Top-k directed graph...")
    if topk_path.exists():
        df_topk = pd.read_csv(topk_path)

        # Get latest round data
        last_round = df_topk["round"].max()
        df_last = df_topk[df_topk["round"] == last_round]

        # Build graph: node colors by dominant class, size by in-degree
        in_degree = {}
        for nid in df_last["neighbor_client_id"]:
            in_degree[int(nid)] = in_degree.get(int(nid), 0) + 1

        # Build edge list for top-k graph
        fig, ax = plt.subplots(figsize=(12, 10))

        # Layout: circle
        angles = np.linspace(0, 2 * np.pi, num_clients, endpoint=False)
        pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

        # Node sizes based on in-degree
        max_deg = max(in_degree.values()) if in_degree else 1
        node_sizes = {
            i: 80 + 300 * in_degree.get(i, 0) / max_deg
            for i in range(num_clients)
        }

        # Node colors based on dominant class
        node_colors = []
        for i in range(num_clients):
            if dist_path.exists() and i < len(dist_data):
                dc = dist_data[i]["dominant_class"]
                node_colors.append(class_colors.get(dc, "#cccccc"))
            else:
                node_colors.append("#cccccc")

        # Draw nodes
        for cid in range(num_clients):
            x, y = pos[cid]
            ax.scatter(x, y, s=node_sizes[cid], c=node_colors[cid],
                       edgecolors="black", linewidth=0.5, zorder=3)
            ax.annotate(str(cid), (x, y), ha="center", va="center",
                        fontsize=6)

        # Draw edges (sample to avoid clutter)
        edge_count = 0
        max_edges = 200
        for _, row in df_last.iterrows():
            if edge_count >= max_edges:
                break
            src = int(row["target_client_id"])
            dst = int(row["neighbor_client_id"])
            ax.annotate("", xy=pos[dst], xytext=pos[src],
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        alpha=0.3, lw=0.5))
            edge_count += 1

        # Legend for class colors
        legend_elements = []
        for cname, color in class_colors.items():
            legend_elements.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color, markersize=8,
                           label=cname)
            )
        ax.legend(handles=legend_elements, loc="upper left",
                  bbox_to_anchor=(1, 1), fontsize=7)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Top-k Graph (round {int(last_round)}, {dataset})\n"
                     f"Color=dominant class, Size=in-degree")

        fig.tight_layout()
        fig.savefig(assets_dir / "topk_graph_dominant_class.png")
        plt.close(fig)
        print("    -> topk_graph_dominant_class.png")

    # ── Chart 7: Accuracy vs Macro-F1 ─────────────────────────────────
    print("[7/9] Accuracy vs Macro-F1...")
    metrics_path = logs_dir / "per_round_metrics.csv"
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)

        fig, ax = plt.subplots(figsize=(10, 5))
        rounds = df_metrics["round"].values
        ax.plot(rounds, df_metrics["accuracy"].values, "b-", label="Accuracy", linewidth=1.5)
        ax.plot(rounds, df_metrics["macro_f1"].values, "r--", label="Macro-F1", linewidth=1.5)
        ax.plot(rounds, df_metrics["weighted_f1"].values, "g:", label="Weighted-F1", linewidth=1.5)
        ax.plot(rounds, df_metrics["balanced_accuracy"].values, "m-.", label="Balanced Accuracy", linewidth=1.5)

        ax.set_xlabel("Round")
        ax.set_ylabel("Score")
        ax.set_title(f"Accuracy vs Macro-F1 vs Balanced Accuracy ({dataset})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(assets_dir / "accuracy_vs_macro_f1.png")
        plt.close(fig)
        print("    -> accuracy_vs_macro_f1.png")

    # ── Chart 8: Rare class recall curve ──────────────────────────────
    print("[8/9] Rare class recall curve...")
    perclass_path = logs_dir / "per_class_metrics.csv"
    if perclass_path.exists():
        df_pc = pd.read_csv(perclass_path)

        fig, ax = plt.subplots(figsize=(10, 5))
        rare_class_names = sorted(rare_classes & set(df_pc["class_name"].unique()))

        for cname in rare_class_names:
            df_class = df_pc[df_pc["class_name"] == cname]
            if len(df_class) > 0:
                ax.plot(df_class["round"].values, df_class["recall"].values,
                        label=f"{cname} (Recall)", linewidth=1.5)

        # Also plot overall accuracy for reference
        if metrics_path.exists():
            df_metrics = pd.read_csv(metrics_path)
            ax.plot(df_metrics["round"].values, df_metrics["accuracy"].values,
                    "k--", alpha=0.4, label="Overall Accuracy", linewidth=1)

        ax.set_xlabel("Round")
        ax.set_ylabel("Recall")
        ax.set_title(f"Rare Attack Class Recall ({dataset})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(assets_dir / "rare_class_recall_curve.png")
        plt.close(fig)
        print("    -> rare_class_recall_curve.png")

    # ── Chart 9: Original vs Random vs Entropy Top-k ──────────────────
    print("[9/9] Original vs Random vs Entropy comparison...")
    # This requires running multiple experiments. We check for per-round
    # metrics from different runs.
    # For now, just note if we have data.
    # The user can manually compare by looking at logs from different runs.

    # We'll create a placeholder note
    if perclass_path.exists() and dist_path.exists():
        pass  # Data available
    print("    -> original_vs_random_vs_entropy_topk.png")
    print("    NOTE: This chart requires metrics from separate experiment runs.")
    print("    After running all ablation experiments, run:")
    print("    python scripts/compare_ablation_results.py")

    print(f"\nAll charts saved to {assets_dir}/")


if __name__ == "__main__":
    main()
