"""
UAV-NIDD Audit Script
=====================
Audits the UAV-NIDD dataset across four dimensions:
  1. Sample counts per attack type, GCS side vs UAV side
  2. Feature counts per attack type, GCS side vs UAV side
  3. Cross-view analysis: same attack, GCS vs UAV — how severely each
     side is affected (using normalized deviation from its own Normal)
  4. Within-view analysis: different attacks on the same platform —
     which features shift most between attack types

All output goes to docs/audit/uav_nidd_audit_report.txt
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import variation

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "docs" / "audit" / "uav_nidd_audit_report.txt"

GCS_PATH = ROOT / "dataset" / "UAV-NIDD" / "UAV-NDD CSV" / "GSC Case3 Label .csv"
UAV_PATH = ROOT / "dataset" / "UAV-NIDD" / "UAV-NDD CSV" / "UAV-Case1-Label.csv"

# ── Helper ──────────────────────────────────────────────────────────────────

def sepline(f, title, char="="):
    f.write(f"\n{char * 70}\n{title}\n{char * 70}\n\n")


def load_gcs():
    """Load GCS Case3 (xlsx despite .csv extension)."""
    df = pd.read_excel(GCS_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    # Normalize labels
    label_map = {
        "Bruteforce": "Brute Force", "Brute-Force": "Brute Force",
        "Benign": "Normal", "Normal": "Normal",
        "Fake Landing ": "Fake Landing",
    }
    df["Class"] = df["Class"].str.strip().replace(label_map)
    return df


def load_uav():
    """Load UAV-Case1 (real CSV)."""
    df = pd.read_csv(UAV_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    label_map = {
        "BruteForce": "Brute Force", "Brute-Force": "Brute Force",
        "replay": "Replay", "FakeLanding": "Fake Landing",
        "Reconnassiance": "Reconnaissance",
    }
    df["Label"] = df["Label"].str.strip().replace(label_map)
    return df


def get_numeric_features(df, label_col):
    """Return only numeric feature columns (exclude label, uid, IPs)."""
    skip = {label_col}
    extra_skip = {"uid", "id.orig_h", "id.resp_h", "frame.number",
                  "frame.time_epoch", "frame.time_relative"}
    cols = []
    for c in df.columns:
        if c in skip or c in extra_skip:
            continue
        if df[c].dtype in ("int64", "float64"):
            cols.append(c)
    return cols


def normalized_deviation(group_mean, baseline_mean, group_std, baseline_std):
    """
    Cohen's-d-like normalized deviation:
      d = |group_mean - baseline_mean| / pooled_std
    pooled_std = sqrt((s1^2 + s2^2) / 2)
    Returns a pd.Series with original index preserved.
    """
    pooled_var = (group_std ** 2 + baseline_std ** 2) / 2
    pooled_std = np.sqrt(pooled_var)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.abs(group_mean - baseline_mean) / pooled_std
    result = pd.Series(np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0),
                       index=group_mean.index)
    return result


def top_k_features(series, k=10):
    """Return top-k (feature_name, value) from a Series, descending."""
    return series.sort_values(ascending=False).head(k)


# ── Main audit ──────────────────────────────────────────────────────────────

def main():
    print("Loading GCS Case3 ...")
    gcs = load_gcs()
    print("Loading UAV-Case1 ...")
    uav = load_uav()

    gcs_label = "Class"
    uav_label = "Label"

    gcs_feat_cols = get_numeric_features(gcs, gcs_label)
    uav_feat_cols = get_numeric_features(uav, uav_label)

    gcs_attacks = sorted(gcs[gcs_label].unique())
    uav_attacks = sorted(uav[uav_label].unique())

    all_attacks = sorted(set(gcs_attacks) | set(uav_attacks))

    # Pre-compute per-attack statistics
    gcs_stats = {}  # attack -> {mean, std, count}
    for att in gcs_attacks:
        sub = gcs[gcs[gcs_label] == att][gcs_feat_cols]
        gcs_stats[att] = {
            "mean": sub.mean(), "std": sub.std(ddof=0), "count": len(sub)
        }

    uav_stats = {}
    for att in uav_attacks:
        sub = uav[uav[uav_label] == att][uav_feat_cols]
        uav_stats[att] = {
            "mean": sub.mean(), "std": sub.std(ddof=0), "count": len(sub)
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # ── 1. Sample counts ────────────────────────────────────────────
        sepline(f, "1. SAMPLE COUNTS PER ATTACK (GCS vs UAV)")

        f.write(f"{'Attack':<22s} {'GCS Count':>10s} {'UAV Count':>10s} {'Total':>10s}\n")
        f.write("-" * 55 + "\n")
        gcs_total = 0
        uav_total = 0
        for att in all_attacks:
            gc = gcs_stats.get(att, {}).get("count", 0)
            uc = uav_stats.get(att, {}).get("count", 0)
            gcs_total += gc
            uav_total += uc
            gcs_str = f"{gc:,}" if gc else "-"
            uav_str = f"{uc:,}" if uc else "-"
            f.write(f"{att:<22s} {gcs_str:>10s} {uav_str:>10s} {gc+uc:>10,}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'TOTAL':<22s} {gcs_total:>10,} {uav_total:>10,} {gcs_total+uav_total:>10,}\n")

        f.write(f"\nAttacks present ONLY in GCS: {sorted(set(gcs_attacks) - set(uav_attacks))}\n")
        f.write(f"Attacks present ONLY in UAV: {sorted(set(uav_attacks) - set(gcs_attacks))}\n")
        f.write(f"Attacks present in BOTH:    {sorted(set(gcs_attacks) & set(uav_attacks))}\n")

        # ── 2. Feature counts ───────────────────────────────────────────
        sepline(f, "2. FEATURE COUNTS PER ATTACK (GCS vs UAV)")

        gcs_feat_n = len(gcs_feat_cols)
        uav_feat_n = len(uav_feat_cols)
        f.write(f"GCS — {gcs_feat_n} numeric features (flow-level: Zeek conn log + flowmeter)\n")
        f.write(f"UAV — {uav_feat_n} numeric features (packet-level: radiotap / wlan / IP / UDP / TCP)\n\n")
        f.write("Note: GCS and UAV feature spaces are completely disjoint.\n")
        f.write("      Feature types are listed in the canonical audit report.\n")
        f.write(f"      All {gcs_feat_n} GCS features and all {uav_feat_n} UAV features are shared\n")
        f.write("      across all attack types on that side (uniform schema per side).\n")

        # ── 3. Cross-view: same attack, GCS vs UAV ─────────────────────
        sepline(f, "3. CROSS-VIEW: SAME ATTACK – GCS vs UAV IMPACT COMPARISON")
        f.write("For each attack present on BOTH sides, we compute the per-feature\n")
        f.write("normalized deviation from Normal/Benign (Cohen's d against baseline).\n")
        f.write("We then report the Top-10 most-deviated features per side to show\n")
        f.write("HOW each platform experiences the same attack.\n\n")

        common_attacks = sorted(set(gcs_attacks) & set(uav_attacks) - {"Normal"})

        for att in common_attacks:
            f.write(f"── Attack: {att} ──\n\n")

            # GCS side
            gcs_normal = gcs_stats["Normal"]
            gcs_att = gcs_stats[att]
            gcs_dev = normalized_deviation(
                gcs_att["mean"], gcs_normal["mean"],
                gcs_att["std"], gcs_normal["std"]
            )
            gcs_top = top_k_features(gcs_dev, 10)
            f.write(f"  GCS (flow-level) — Top-10 most-deviated features vs Normal:\n")
            for feat, val in gcs_top.items():
                f.write(f"    {feat:<45s} d={val:.3f}\n")
            f.write(f"  GCS mean |d| across all features: {gcs_dev.mean():.4f}\n\n")

            # UAV side
            uav_normal = uav_stats["Normal"]
            uav_att = uav_stats[att]
            uav_dev = normalized_deviation(
                uav_att["mean"], uav_normal["mean"],
                uav_att["std"], uav_normal["std"]
            )
            uav_top = top_k_features(uav_dev, 10)
            f.write(f"  UAV (packet-level) — Top-10 most-deviated features vs Normal:\n")
            for feat, val in uav_top.items():
                f.write(f"    {feat:<45s} d={val:.3f}\n")
            f.write(f"  UAV mean |d| across all features: {uav_dev.mean():.4f}\n\n")

            f.write("  Interpretation:\n")
            f.write(f"    GCS mean |d| = {gcs_dev.mean():.4f}  |  UAV mean |d| = {uav_dev.mean():.4f}\n")
            if gcs_dev.mean() > uav_dev.mean():
                f.write(f"    → GCS shows LARGER overall deviation under {att} than UAV.\n")
            else:
                f.write(f"    → UAV shows LARGER overall deviation under {att} than GCS.\n")
            f.write(f"    The deviated features differ: GCS sees flow-metric shifts,\n")
            f.write(f"    UAV sees packet/radio-level shifts.\n\n")

        # Attacks only on one side — compare to its own Normal
        sepline(f, "3b. CROSS-VIEW: ATTACKS ON ONLY ONE SIDE (deviation vs own Normal)")
        for att in all_attacks:
            if att in common_attacks or att == "Normal":
                continue
            f.write(f"── Attack: {att} ──\n")
            if att in gcs_stats:
                normal = gcs_stats["Normal"]
                att_s = gcs_stats[att]
                dev = normalized_deviation(
                    att_s["mean"], normal["mean"],
                    att_s["std"], normal["std"]
                )
                top = top_k_features(dev, 10)
                f.write(f"  GCS only — Top-10 most-deviated features vs Normal:\n")
                for feat, val in top.items():
                    f.write(f"    {feat:<45s} d={val:.3f}\n")
                f.write(f"  Mean |d| across all features: {dev.mean():.4f}\n")
            else:
                normal = uav_stats["Normal"]
                att_s = uav_stats[att]
                dev = normalized_deviation(
                    att_s["mean"], normal["mean"],
                    att_s["std"], normal["std"]
                )
                top = top_k_features(dev, 10)
                f.write(f"  UAV only — Top-10 most-deviated features vs Normal:\n")
                for feat, val in top.items():
                    f.write(f"    {feat:<45s} d={val:.3f}\n")
                f.write(f"  Mean |d| across all features: {dev.mean():.4f}\n")
            f.write("\n")

        # ── 4. Within-view: different attacks, same platform ──────────
        sepline(f, "4. WITHIN-VIEW: DIFFERENT ATTACKS ON THE SAME PLATFORM")
        f.write("For each platform (GCS / UAV), we compute the per-feature deviation\n")
        f.write("from Normal for every attack, then aggregate across attacks to find\n")
        f.write("which features are MOST sensitive to attacks in general (high variance\n")
        f.write("of deviation across attacks) and which attacks cause the LARGEST\n")
        f.write("overall shift from Normal.\n\n")

        for side_name, stats, feat_cols in [
            ("GCS (flow-level)", gcs_stats, gcs_feat_cols),
            ("UAV (packet-level)", uav_stats, uav_feat_cols),
        ]:
            f.write(f"{'=' * 50}\n")
            f.write(f"  {side_name}\n")
            f.write(f"{'=' * 50}\n\n")

            normal = stats["Normal"]
            attacks = [a for a in sorted(stats) if a != "Normal"]

            # deviation matrix: attack x feature
            dev_matrix = {}
            for att in attacks:
                att_s = stats[att]
                dev = normalized_deviation(
                    att_s["mean"], normal["mean"],
                    att_s["std"], normal["std"]
                )
                dev_matrix[att] = dev

            dev_df = pd.DataFrame(dev_matrix).T  # attack × feature

            # 4a. Which attacks cause the largest overall shift?
            f.write("4a. Attack impact ranking (mean |d| across all features):\n\n")
            attack_impact = dev_df.mean(axis=1).sort_values(ascending=False)
            for rank, (att, mean_d) in enumerate(attack_impact.items(), 1):
                f.write(f"  {rank:2d}. {att:<22s} mean|d| = {mean_d:.4f}\n")

            # 4b. Which features are most sensitive (high std of d across attacks)?
            f.write(f"\n4b. Top-15 features with highest VARIANCE of d across attacks\n")
            f.write(f"    (features whose deviation changes most depending on attack type):\n\n")
            feature_std = dev_df.std(axis=0).sort_values(ascending=False)
            for rank, (feat, std_val) in enumerate(feature_std.head(15).items(), 1):
                f.write(f"  {rank:2d}. {feat:<45s} std(d)={std_val:.4f}\n")

            # 4c. Which features are MOST consistently deviated across all attacks?
            f.write(f"\n4c. Top-15 features with highest MEAN d across attacks\n")
            f.write(f"    (features that are universally shifted regardless of attack type):\n\n")
            feature_mean = dev_df.mean(axis=0).sort_values(ascending=False)
            for rank, (feat, mean_val) in enumerate(feature_mean.head(15).items(), 1):
                f.write(f"  {rank:2d}. {feat:<45s} mean(d)={mean_val:.4f}\n")

            # 4d. Attack similarity matrix (correlation of d vectors)
            f.write(f"\n4d. Attack similarity matrix (Pearson r of deviation vectors):\n\n")
            if len(dev_df) >= 2:
                # Compute cross-correlation
                f.write(f"    {'':<22s} " + " ".join(f"{a[:10]:>10s}" for a in attacks) + "\n")
                for a1 in attacks:
                    row = []
                    for a2 in attacks:
                        r = np.corrcoef(dev_df.loc[a1], dev_df.loc[a2])[0, 1]
                        row.append(f"{r:10.3f}")
                    f.write(f"    {a1:<22s} " + " ".join(row) + "\n")

            f.write("\n")

    print(f"Audit report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
