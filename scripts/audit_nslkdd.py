"""
NSLKDD Audit Script
===================
Audits the NSLKDD dataset (single UAV-Client perspective):
  0. Feature and label inventory
  1. Sample counts per attack type
  2. Feature counts and description
  4. Within-view — attack impact ranking, feature sensitivity, attack similarity

Usage: uv run python scripts/audit_nslkdd.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "docs" / "audit" / "nslkdd_audit_report.txt"

TRAIN_PATH = ROOT / "dataset" / "NSLKDD" / "rawdata" / "datasets" / "KDDTrain5.csv"
TEST_PATH = ROOT / "dataset" / "NSLKDD" / "rawdata" / "datasets" / "KDDTest5.csv"

NON_FEATURE = {"label"}

LABEL_MAP = {5: "Normal", 1: "DoS", 2: "Probe", 3: "U2R", 4: "R2L"}


def sepline(f, title, char="="):
    f.write(f"\n{char * 70}\n{title}\n{char * 70}\n\n")


def get_numeric_cols(df, non_feature_set):
    numeric = []
    for c in df.columns:
        c_str = str(c).strip()
        if c_str in non_feature_set:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c_str)
    return numeric


def compute_stats(df, numeric_cols):
    if len(df) == 0:
        return {"count": 0, "mean": pd.Series(dtype="float64"),
                "std": pd.Series(dtype="float64"),
                "min": pd.Series(dtype="float64"),
                "max": pd.Series(dtype="float64")}
    sub = df[numeric_cols].fillna(0)
    sub_arr = sub.values.astype("float32")
    count = len(sub_arr)
    mean_vals = np.mean(sub_arr, axis=0)
    std_vals = np.std(sub_arr, axis=0, ddof=0)
    min_vals = np.min(sub_arr, axis=0)
    max_vals = np.max(sub_arr, axis=0)
    return {
        "count": count,
        "mean": pd.Series(mean_vals, index=numeric_cols),
        "std": pd.Series(std_vals, index=numeric_cols),
        "min": pd.Series(min_vals, index=numeric_cols),
        "max": pd.Series(max_vals, index=numeric_cols),
    }


def normalized_deviation(attack_mean, baseline_mean,
                          attack_std, baseline_std,
                          baseline_min, baseline_max):
    baseline_range = baseline_max - baseline_min
    safe_range = np.where(baseline_range > 1e-12, baseline_range, np.inf)
    attack_norm = (attack_mean - baseline_min) / safe_range
    baseline_norm = (baseline_mean - baseline_min) / safe_range
    attack_std_norm = attack_std / safe_range
    baseline_std_norm = baseline_std / safe_range
    attack_norm = attack_norm.fillna(0).replace([np.inf, -np.inf], 0)
    baseline_norm = baseline_norm.fillna(0).replace([np.inf, -np.inf], 0)
    attack_std_norm = attack_std_norm.fillna(0).replace([np.inf, -np.inf], 0)
    baseline_std_norm = baseline_std_norm.fillna(0).replace([np.inf, -np.inf], 0)
    pooled_var = (attack_std_norm ** 2 + baseline_std_norm ** 2) / 2
    pooled_std = np.sqrt(np.maximum(pooled_var, 1e-12))
    d = np.abs(attack_norm - baseline_norm) / pooled_std
    return d.fillna(0).replace([np.inf, -np.inf], 0)


def top_k(series, k=10):
    return series.sort_values(ascending=False).head(k)


def main():
    print("=" * 60)
    print("NSLKDD Audit")
    print("=" * 60)

    print("\n[1/2] Loading data ...")
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)
    df = pd.concat([train, test], ignore_index=True, copy=False)
    print(f"  Train: {len(train):,} rows, Test: {len(test):,} rows")
    print(f"  Total: {len(df):,} rows")

    numeric_cols = get_numeric_cols(df, NON_FEATURE)
    print(f"  {len(numeric_cols)} numeric features detected")

    print("\n[2/2] Computing per-attack statistics ...")
    stats = {}
    for label_val in sorted(df["label"].unique()):
        label_int = int(label_val)
        attack_name = LABEL_MAP.get(label_int, f"Unknown({label_int})")
        sub = df[df["label"] == label_int]
        s = compute_stats(sub, numeric_cols)
        s["numeric_cols"] = numeric_cols
        stats[attack_name] = s
        print(f"  {attack_name:<12s}: {s['count']:>10,} rows")

    attacks = sorted(stats.keys())
    non_normal_attacks = [a for a in attacks if a != "Normal"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # ── 0. Feature and Label Inventory ──────────────────────────────
        sepline(f, "0. FEATURE AND LABEL INVENTORY")

        f.write("Dataset: NSLKDD (KDDTrain5.csv + KDDTest5.csv)\n")
        f.write("Source: https://www.unb.ca/cic/datasets/nsl.html\n")
        f.write("Perspective: UAV-Client (single-view, no GCS)\n\n")

        f.write(f"Features ({len(numeric_cols)} columns, all numeric after preprocessing):\n\n")

        continuous_features = [
            "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent",
            "hot", "num_failed_logins", "num_compromised", "num_root",
            "num_file_creations", "num_shells", "num_access_files",
            "num_outbound_cmds", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
        ]
        binary_features = [
            "land", "logged_in", "root_shell", "su_attempted",
            "is_host_login", "is_guest_login",
        ]
        categorical_features = ["protocol_type", "service", "flag"]

        f.write("  Continuous (32):\n")
        for name in continuous_features:
            desc = {
                "duration": "Length (seconds) of the connection",
                "src_bytes": "Bytes sent from source to destination",
                "dst_bytes": "Bytes sent from destination to source",
                "wrong_fragment": "Number of wrong fragments",
                "urgent": "Number of urgent packets",
                "hot": "Number of 'hot' indicators",
                "num_failed_logins": "Number of failed login attempts",
                "num_compromised": "Number of 'compromised' conditions",
                "num_root": "Number of 'root' accesses",
                "num_file_creations": "Number of file creation operations",
                "num_shells": "Number of shell prompts",
                "num_access_files": "Number of operations on access control files",
                "num_outbound_cmds": "Number of outbound commands in FTP session",
                "count": "Connections to same host in past 2 seconds",
                "srv_count": "Connections to same service in past 2 seconds",
                "serror_rate": "% of connections with SYN errors (same host)",
                "srv_serror_rate": "% of connections with SYN errors (same service)",
                "rerror_rate": "% of connections with REJ errors (same host)",
                "srv_rerror_rate": "% of connections with REJ errors (same service)",
                "same_srv_rate": "% of connections to same service (same host)",
                "diff_srv_rate": "% of connections to different services (same host)",
                "srv_diff_host_rate": "% of connections to different hosts (same service)",
                "dst_host_count": "Connections to same destination host (100 conns)",
                "dst_host_srv_count": "Connections to same service on dest host",
                "dst_host_same_srv_rate": "% to same service (dest host)",
                "dst_host_diff_srv_rate": "% to different services (dest host)",
                "dst_host_same_src_port_rate": "% to same src port (dest host)",
                "dst_host_srv_diff_host_rate": "% to different hosts (dest service)",
                "dst_host_serror_rate": "% with SYN errors (dest host)",
                "dst_host_srv_serror_rate": "% with SYN errors (dest service)",
                "dst_host_rerror_rate": "% with REJ errors (dest host)",
                "dst_host_srv_rerror_rate": "% with REJ errors (dest service)",
            }.get(name, "")
            f.write(f"    {name:<35s} {desc}\n")

        f.write("\n  Binary (6):\n")
        for name in binary_features:
            desc = {
                "land": "1 if connection from/to same host/port, else 0",
                "logged_in": "1 if successfully logged in, else 0",
                "root_shell": "1 if root shell obtained, else 0",
                "su_attempted": "1 if 'su root' attempted, else 0",
                "is_host_login": "1 if login belongs to 'host' list, else 0",
                "is_guest_login": "1 if guest login, else 0",
            }.get(name, "")
            f.write(f"    {name:<35s} {desc}\n")

        f.write("\n  Categorical, label-encoded (3):\n")
        f.write("    protocol_type:  tcp=0, udp=1, icmp=2\n")
        f.write("    service:        70 network service categories\n")
        f.write("    flag:           11 connection status flags\n")

        f.write(f"\n  Total: {len(numeric_cols)} numeric feature columns.\n")
        f.write("  Note: Categorical features were label-encoded in preprocessing;\n")
        f.write("  the deviation analysis (Section 4) treats all columns as numeric,\n")
        f.write("  matching the actual input to the federated model.\n")

        f.write(f"\nLabels ({len(LABEL_MAP)} classes):\n\n")
        f.write("  Label  Class    Original attack types\n")
        f.write("  ─────  ─────    ───────────────────────────────────────────\n")
        f.write("  5      Normal   normal\n")
        f.write("  1      DoS      neptune, teardrop, smurf, pod, back, land,\n")
        f.write("                  apache2, processtable, mailbomb, worm, udpstorm\n")
        f.write("  2      Probe    ipsweep, portsweep, nmap, satan, saint, mscan\n")
        f.write("  3      U2R      rootkit, buffer_overflow, loadmodule, perl, ps,\n")
        f.write("                  sqlattack, xterm\n")
        f.write("  4      R2L      warezclient, guess_passwd, ftp_write, multihop,\n")
        f.write("                  imap, warezmaster, phf, spy, snmpgetattack,\n")
        f.write("                  httptunnel, snmpguess, named, sendmail, xlock, xsnoop\n")

        # ── 1. Sample counts ────────────────────────────────────────────
        sepline(f, "1. SAMPLE COUNTS PER ATTACK")

        f.write(f"{'Attack':<12s} {'Train':>12s} {'Test':>12s} {'Total':>12s} "
                f"{'Fraction':>10s}\n")
        f.write("-" * 58 + "\n")
        total_all = len(df)
        for att in attacks:
            label_int = {v: k for k, v in LABEL_MAP.items()}[att]
            train_cnt = len(train[train["label"] == label_int])
            test_cnt = len(test[test["label"] == label_int])
            total = train_cnt + test_cnt
            pct = 100 * total / total_all
            f.write(f"{att:<12s} {train_cnt:>12,} {test_cnt:>12,} {total:>12,} "
                    f"{pct:>9.1f}%\n")
        f.write("-" * 58 + "\n")
        f.write(f"{'TOTAL':<12s} {len(train):>12,} {len(test):>12,} "
                f"{total_all:>12,} {100.0:>9.1f}%\n")

        f.write(f"\nClass balance note:\n")
        f.write(f"  NSLKDD is highly imbalanced. Normal dominates, while U2R and\n")
        f.write(f"  R2L have very few samples, making them challenging for FL.\n")

        # ── 2. Feature counts ───────────────────────────────────────────
        sepline(f, "2. FEATURE COUNTS AND DESCRIPTION")

        f.write(f"Columns in CSV:            {len(df.columns)}\n")
        f.write(f"Numeric feature columns:   {len(numeric_cols)}\n")
        f.write(f"Label column:             1 (label, integer 1–5)\n\n")

        f.write("Feature type breakdown:\n")
        f.write(f"  Continuous:              {len(continuous_features)}\n")
        f.write(f"  Binary (0/1 flags):     {len(binary_features)}\n")
        f.write(f"  Categorical (encoded):   {len(categorical_features)}\n")
        f.write(f"  Total features:          {len(numeric_cols)}\n\n")

        f.write("Feature space description:\n")
        f.write("  - Basic connection features (duration, bytes, fragments, urgent)\n")
        f.write("  - Content features (login attempts, root access, shells, files)\n")
        f.write("  - Time-based traffic features (count, srv_count, error rates)\n")
        f.write("    computed over 2-second windows\n")
        f.write("  - Host-based traffic features (dst_host_*), computed over\n")
        f.write("    100-connection windows to same destination host\n")
        f.write("  - All features share a uniform schema across all attack types\n")
        f.write("  - Data is Z-score normalized in FL preprocessing\n")

        # ── 4. Within-view analysis ─────────────────────────────────────
        sepline(f, "4. WITHIN-VIEW: DIFFERENT ATTACKS ON NSLKDD")

        f.write("Method: For each attack type, compute per-feature deviation from\n")
        f.write("Normal using range-normalized Cohen's d (each feature normalized\n")
        f.write("to [0,1] by Normal's observed range before computing d).\n\n")
        f.write("This reveals which attacks cause the largest shift from Normal,\n")
        f.write("which features are most attack-sensitive, and which attacks have\n")
        f.write("similar deviation patterns.\n\n")

        normal = stats["Normal"]

        dev_records = {}
        for att in non_normal_attacks:
            att_s = stats[att]
            nc = sorted(set(att_s["numeric_cols"]) & set(normal["numeric_cols"]))
            dev = normalized_deviation(
                att_s["mean"][nc], normal["mean"][nc],
                att_s["std"][nc], normal["std"][nc],
                normal["min"][nc], normal["max"][nc],
            )
            dev_records[att] = dev

        dev_df = pd.DataFrame(dev_records).T
        ref_cols = list(dev_df.columns)

        # 4a. Attack impact ranking
        f.write("4a. Attack impact ranking (mean |d| across all features):\n\n")
        attack_impact = dev_df.mean(axis=1).sort_values(ascending=False)
        for rank, (att, mean_d) in enumerate(attack_impact.items(), 1):
            f.write(f"  {rank:2d}. {att:<12s} mean|d| = {mean_d:.4f}\n")

        # 4b. Features with highest variance of d
        f.write(f"\n4b. Top-15 features with highest VARIANCE of d across attacks\n")
        f.write(f"    (features whose deviation changes most depending on "
                f"attack type):\n\n")
        feature_std = dev_df.std(axis=0).sort_values(ascending=False)
        for rank, (feat, std_val) in enumerate(feature_std.head(15).items(), 1):
            f.write(f"  {rank:2d}. {feat:<40s} std(d)={std_val:.4f}\n")

        # 4c. Features with highest mean d
        f.write(f"\n4c. Top-15 features with highest MEAN d across attacks\n")
        f.write(f"    (features universally shifted regardless of attack type):\n\n")
        feature_mean = dev_df.mean(axis=0).sort_values(ascending=False)
        for rank, (feat, mean_val) in enumerate(feature_mean.head(15).items(), 1):
            f.write(f"  {rank:2d}. {feat:<40s} mean(d)={mean_val:.4f}\n")

        # 4d. Attack similarity matrix
        f.write(f"\n4d. Attack similarity matrix "
                f"(Pearson r of deviation vectors):\n\n")
        if len(dev_df) >= 2:
            att_list = list(dev_df.index)
            f.write(f"    {'':<12s}"
                    + "".join(f"{a[:8]:>9s}" for a in att_list) + "\n")
            for a1 in att_list:
                row = []
                for a2 in att_list:
                    r = np.corrcoef(dev_df.loc[a1], dev_df.loc[a2])[0, 1]
                    row.append(f"{r:9.3f}")
                f.write(f"    {a1:<12s}" + "".join(row) + "\n")

        # 4e. Largest d per attack
        f.write(f"\n4e. Quick-reference — largest single-feature "
                f"deviation per attack:\n\n")
        for att in non_normal_attacks:
            max_feat = dev_df.loc[att].idxmax()
            max_val = dev_df.loc[att].max()
            f.write(f"  {att:<12s} {max_feat:<40s} d={max_val:.3f}\n")

    print(f"\nAudit report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
