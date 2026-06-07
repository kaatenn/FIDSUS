"""
UNSW-NB15 Audit Script
======================
Audits the UNSW-NB15 dataset (single UAV-Client perspective):
  0. Feature and label inventory
  1. Sample counts per attack type
  2. Feature counts and description
  4. Within-view — attack impact ranking, feature sensitivity, attack similarity

Usage: uv run python scripts/audit_unsw.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "docs" / "audit" / "unsw_audit_report.txt"

TRAIN_PATH = ROOT / "dataset" / "UNSW" / "rawdata" / "datasets" / "unsw_train10.csv"
TEST_PATH = ROOT / "dataset" / "UNSW" / "rawdata" / "datasets" / "unsw_test10.csv"

NON_FEATURE = {"id", "proto", "service", "state", "attack_cat", "label"}

LABEL_MAP = {
    1: "Normal", 2: "Backdoor", 3: "Analysis", 4: "Fuzzers",
    5: "Shellcode", 6: "Reconnaissance", 7: "Exploits", 8: "DoS",
    9: "Worms", 10: "Generic",
}

NUMERIC_FEATURES_LIST = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb",
    "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]

CATEGORICAL_FEATURES = ["proto", "service", "state"]


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
    print("UNSW-NB15 Audit")
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
        print(f"  {attack_name:<16s}: {s['count']:>10,} rows")

    attacks = sorted(stats.keys())
    non_normal_attacks = [a for a in attacks if a != "Normal"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # ── 0. Feature and Label Inventory ──────────────────────────────
        sepline(f, "0. FEATURE AND LABEL INVENTORY")

        f.write("Dataset: UNSW-NB15 (unsw_train10.csv + unsw_test10.csv)\n")
        f.write("Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset\n")
        f.write("Perspective: UAV-Client (single-view, no GCS)\n\n")

        f.write(f"Features ({len(NUMERIC_FEATURES_LIST)} numeric, "
                f"{len(CATEGORICAL_FEATURES)} categorical, "
                f"1 id column):\n\n")

        feature_desc = {
            "dur": "Record total duration",
            "spkts": "Source-to-destination packet count",
            "dpkts": "Destination-to-source packet count",
            "sbytes": "Source-to-destination transaction bytes",
            "dbytes": "Destination-to-source transaction bytes",
            "rate": "Total packets per second",
            "sttl": "Source-to-destination time-to-live value",
            "dttl": "Destination-to-source time-to-live value",
            "sload": "Source bits per second",
            "dload": "Destination bits per second",
            "sloss": "Source packets retransmitted or dropped",
            "dloss": "Destination packets retransmitted or dropped",
            "sinpkt": "Source inter-packet arrival time (ms)",
            "dinpkt": "Destination inter-packet arrival time (ms)",
            "sjit": "Source jitter (ms)",
            "djit": "Destination jitter (ms)",
            "swin": "Source TCP window advertisement value",
            "stcpb": "Source TCP base sequence number",
            "dtcpb": "Destination TCP base sequence number",
            "dwin": "Destination TCP window advertisement value",
            "tcprtt": "TCP connection setup RTT",
            "synack": "TCP connection setup time (SYN→SYN_ACK)",
            "ackdat": "TCP connection setup time (SYN_ACK→ACK)",
            "smean": "Mean packet size transmitted by source",
            "dmean": "Mean packet size transmitted by destination",
            "trans_depth": "Pipelined depth into HTTP connection",
            "response_body_len": "Actual uncompressed HTTP response body size",
            "ct_srv_src": "No. of connections of same service from same source (100 conns)",
            "ct_state_ttl": "No. of connections per state_ttl range",
            "ct_dst_ltm": "No. of connections to same dest address (100 conns)",
            "ct_src_dport_ltm": "No. of connections from same src→dest port (100 conns)",
            "ct_dst_sport_ltm": "No. of connections to same dest→src port (100 conns)",
            "ct_dst_src_ltm": "No. of connections from same src→dest pair (100 conns)",
            "is_ftp_login": "1 if FTP session logged in, else 0",
            "ct_ftp_cmd": "No. of FTP commands in session",
            "ct_flw_http_mthd": "No. of HTTP methods in session",
            "ct_src_ltm": "No. of connections from same source (100 conns)",
            "ct_srv_dst": "No. of connections of same service to same dest (100 conns)",
            "is_sm_ips_ports": "1 if src=dest IP and src=dest port, else 0",
        }

        f.write("  Numeric features (39):\n")
        for name in NUMERIC_FEATURES_LIST:
            desc = feature_desc.get(name, "")
            f.write(f"    {name:<25s} {desc}\n")

        f.write("\n  Categorical features (3, excluded from numeric analysis):\n")
        f.write("    proto:     Transport protocol (label-encoded: 0=TCP, etc.)\n")
        f.write("    service:   Network service type (label-encoded: 70=dns, etc.)\n")
        f.write("    state:     Connection state (label-encoded)\n")

        f.write("\n  Meta/ID columns (2, excluded from analysis):\n")
        f.write("    id:         Row identifier\n")
        f.write("    attack_cat: Original attack category string (used for label mapping)\n")

        f.write(f"\n  Total: {len(NUMERIC_FEATURES_LIST)} numeric features analyzed, "
                f"{len(CATEGORICAL_FEATURES)} categorical excluded.\n")
        f.write("  Note: Categorical features were label-encoded in preprocessing;\n")
        f.write("  the deviation analysis (Section 4) uses only the 39 numeric features,\n")
        f.write("  matching the input to the federated model (id & attack_cat excluded).\n")

        f.write(f"\nLabels ({len(LABEL_MAP)} classes):\n\n")
        f.write("  Label  Class             attack_cat value\n")
        f.write("  ─────  ────────────────  ──────────────\n")
        f.write("  1      Normal            Normal\n")
        f.write("  2      Backdoor          Backdoor\n")
        f.write("  3      Analysis          Analysis\n")
        f.write("  4      Fuzzers           Fuzzers\n")
        f.write("  5      Shellcode         Shellcode\n")
        f.write("  6      Reconnaissance    Reconnaissance\n")
        f.write("  7      Exploits          Exploits\n")
        f.write("  8      DoS               DoS\n")
        f.write("  9      Worms             Worms\n")
        f.write("  10     Generic           Generic\n")

        # ── 1. Sample counts ────────────────────────────────────────────
        sepline(f, "1. SAMPLE COUNTS PER ATTACK")

        f.write(f"{'Attack':<16s} {'Train':>12s} {'Test':>12s} {'Total':>12s} "
                f"{'Fraction':>10s}\n")
        f.write("-" * 62 + "\n")
        total_all = len(df)
        for att in attacks:
            label_int = {v: k for k, v in LABEL_MAP.items()}[att]
            train_cnt = len(train[train["label"] == label_int])
            test_cnt = len(test[test["label"] == label_int])
            total = train_cnt + test_cnt
            pct = 100 * total / total_all
            f.write(f"{att:<16s} {train_cnt:>12,} {test_cnt:>12,} {total:>12,} "
                    f"{pct:>9.1f}%\n")
        f.write("-" * 62 + "\n")
        f.write(f"{'TOTAL':<16s} {len(train):>12,} {len(test):>12,} "
                f"{total_all:>12,} {100.0:>9.1f}%\n")

        f.write(f"\nClass balance note:\n")
        f.write(f"  UNSW-NB15 has moderate class imbalance. Normal is the largest\n")
        f.write(f"  class; Analysis, Backdoor, and Worms have relatively few samples.\n")
        f.write(f"  The 50-client Dirichlet partition introduces additional skew\n")
        f.write(f"  for federated learning experiments.\n")

        # ── 2. Feature counts ───────────────────────────────────────────
        sepline(f, "2. FEATURE COUNTS AND DESCRIPTION")

        f.write(f"Columns in CSV:                {len(df.columns)}\n")
        f.write(f"Numeric feature columns:       39 (continuous + counters)\n")
        f.write(f"Categorical columns:            3 (proto, service, state)\n")
        f.write(f"Meta/ID columns:                2 (id, attack_cat)\n")
        f.write(f"Label column:                   1 (label, integer 1–10)\n")
        f.write(f"Total columns:                  {len(df.columns)}\n\n")

        f.write("Feature type breakdown:\n")
        f.write(f"  Flow-level statistics:       9 (dur, rate, sttl, dttl, sload, dload, sloss, dloss, tcprtt)\n")
        f.write(f"  Packet/byte counters:         7 (spkts, dpkts, sbytes, dbytes, sinpkt, dinpkt, smean, dmean)\n")
        f.write(f"  Jitter/window/seq:           7 (sjit, djit, swin, stcpb, dtcpb, dwin, synack, ackdat)\n")
        f.write(f"  HTTP features:                2 (trans_depth, response_body_len)\n")
        f.write(f"  Connection-time features:    12 (ct_*: various 100-connection counters)\n")
        f.write(f"  Binary flags:                 2 (is_ftp_login, is_sm_ips_ports)\n\n")

        f.write("Feature space description:\n")
        f.write("  - Flow-based statistics similar to NSLKDD but with more detail\n")
        f.write("  - Includes jitter, TCP window, and HTTP response metrics\n")
        f.write("  - Time-based connection counters over 100-connection windows\n")
        f.write("  - All 39 numeric features share a uniform schema across attacks\n")
        f.write("  - Data is Z-score normalized in FL preprocessing\n")
        f.write("  - Categorical features (proto, service, state) are label-encoded\n")
        f.write("    but excluded from the federated model input\n")

        # ── 4. Within-view analysis ─────────────────────────────────────
        sepline(f, "4. WITHIN-VIEW: DIFFERENT ATTACKS ON UNSW-NB15")

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

        # 4a. Attack impact ranking
        f.write("4a. Attack impact ranking (mean |d| across all features):\n\n")
        attack_impact = dev_df.mean(axis=1).sort_values(ascending=False)
        for rank, (att, mean_d) in enumerate(attack_impact.items(), 1):
            f.write(f"  {rank:2d}. {att:<16s} mean|d| = {mean_d:.4f}\n")

        # 4b. Features with highest variance of d
        f.write(f"\n4b. Top-15 features with highest VARIANCE of d across attacks\n")
        f.write(f"    (features whose deviation changes most depending on "
                f"attack type):\n\n")
        feature_std = dev_df.std(axis=0).sort_values(ascending=False)
        for rank, (feat, std_val) in enumerate(feature_std.head(15).items(), 1):
            f.write(f"  {rank:2d}. {feat:<30s} std(d)={std_val:.4f}\n")

        # 4c. Features with highest mean d
        f.write(f"\n4c. Top-15 features with highest MEAN d across attacks\n")
        f.write(f"    (features universally shifted regardless of attack type):\n\n")
        feature_mean = dev_df.mean(axis=0).sort_values(ascending=False)
        for rank, (feat, mean_val) in enumerate(feature_mean.head(15).items(), 1):
            f.write(f"  {rank:2d}. {feat:<30s} mean(d)={mean_val:.4f}\n")

        # 4d. Attack similarity matrix
        f.write(f"\n4d. Attack similarity matrix "
                f"(Pearson r of deviation vectors):\n\n")
        if len(dev_df) >= 2:
            att_list = list(dev_df.index)
            f.write(f"    {'':<16s}"
                    + "".join(f"{a[:8]:>9s}" for a in att_list) + "\n")
            for a1 in att_list:
                row = []
                for a2 in att_list:
                    r = np.corrcoef(dev_df.loc[a1], dev_df.loc[a2])[0, 1]
                    row.append(f"{r:9.3f}")
                f.write(f"    {a1:<16s}" + "".join(row) + "\n")

        # 4e. Largest d per attack
        f.write(f"\n4e. Quick-reference — largest single-feature "
                f"deviation per attack:\n\n")
        for att in non_normal_attacks:
            max_feat = dev_df.loc[att].idxmax()
            max_val = dev_df.loc[att].max()
            f.write(f"  {att:<16s} {max_feat:<30s} d={max_val:.3f}\n")

    print(f"\nAudit report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
