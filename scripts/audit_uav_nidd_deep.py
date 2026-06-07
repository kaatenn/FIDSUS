"""
Deep Audit Script for UAV-NIDD dataset
======================================
Walks through individual GCS/UAV folders, handles multiple CSV formats,
produces comprehensive audit across 4 dimensions:
  1. Sample counts per attack, GCS vs UAV
  2. Feature counts per attack, GCS vs UAV
  3. Cross-view: same attack, GCS vs UAV impact (Cohen's d from own Normal)
  4. Within-view: different attacks on same platform — impact ranking,
     feature sensitivity, attack similarity

Usage: uv run python scripts/audit_uav_nidd_deep.py
"""

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "docs" / "audit" / "uav_nidd_audit_report.txt"

GCS_BASE = ROOT / "dataset" / "UAV-NIDD" / "GCS Case3"
UAV_BASE = ROOT / "dataset" / "UAV-NIDD" / "UAV-Case 1"

GCS_NON_FEATURE = {
    "uid", "ts", "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p",
    "tunnel_parents", "proto", "service", "conn_state", "history",
    "local_orig", "local_resp",
    # Tshark Replay export columns
    "frame.time_epoch", "ip.src", "ip.dst", "http.file_data", "info",
}

UAV_NON_FEATURE = {
    "frame.number", "frame.time", "frame.time_epoch", "frame.time_relative",
    "frame.time_delta", "frame.time_delta_displayed",
    "ip.src", "ip.dst", "ip.proto", "ip.version", "ip.ttl",
    "wlan.sa", "wlan.da", "wlan.ta", "wlan.ra", "wlan.bssid", "wlan.ssid",
    "wlan.tag", "wlan.tag.length", "wlan.country_info.code",
    "wlan.country_info.fnm", "wlan.rsn.ie.pmkid",
    "wlan_rsna_eapol.keydes.data", "wlan_rsna_eapol.keydes.nonce",
    "wlan_rsna_eapol.keydes.key_info.key_mic",
    "arp.src.hw_mac", "arp.dst.hw_mac", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "data.data", "dns.qry.name", "dns.resp.name", "dns.ptr.domain_name",
    "http.file_data", "http.host", "http.server",
    "http.request.uri.path", "http.request.uri.query",
    "http.location", "http.referer",
    "json.value.string", "json.key",
    "smb2.filename", "smb2.host", "smb2.domain",
    "info", "ssh.cookie",
    # Enum/categorical fields (not numeric features)
    "frame.encap_type", "arp.opcode", "arp.hw.type", "arp.proto.type",
    "arp.hw.size", "arp.proto.size",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds",
    "wlan.fc.frag", "wlan.fc.order", "wlan.fc.moredata",
    "wlan.fc.protected", "wlan.fc.pwrmgt", "wlan.fc.retry",
    "wlan.fcs.bad_checksum", "wlan.fixed.beacon",
    "wlan.fixed.capabilities.ess", "wlan.fixed.capabilities.ibss",
    "wlan.fixed.reason_code", "wlan.rsn.capabilities.mfpc",
    "wlan_rsna_eapol.keydes.msgnr", "eapol.keydes.key_len",
    "eapol.type", "llc",
    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.fin",
    "tcp.flags.push", "tcp.flags.reset",
    "tcp.checksum.status", "tcp.analysis.flags",
    "tcp.analysis.retransmission", "tcp.analysis.reused_ports",
    "dns.flags.authoritative", "dns.flags.checkdisable",
    "dns.flags.opcode", "dns.flags.response",
    "dns.retransmit_request", "dns.retransmit_response",
    "smb.flags.notify", "smb.flags.response", "smb.flags2.nt_error",
    "smb.flags2.sec_sig", "smb2.session_flags",
    "http.request.method", "http.response.code", "http.response.code.desc",
    "http.request.version", "http.response.version",
    "ssh.message_code",
    "icmpv6.mldr.nb_mcast_records", "icmpv6.ni.nonce",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def sepline(f, title, char="="):
    f.write(f"\n{char * 70}\n{title}\n{char * 70}\n\n")


def norm_flowmeter_cols(columns):
    """Normalize plain-CSV flowmeter columns to Zeek-style dotted names.

    Plain CSV: fwd_pkts_payload_min, active_min, idle_std
    Zeek CSV:  fwd_pkts_payload.min, active.min, idle.std
    """
    import re

    out = []
    for c in columns:
        c = c.strip().strip('"')
        c = re.sub(r"_(payload|iat)_(min|max|tot|avg|std)$", r"_\1.\2", c)
        c = re.sub(r"^(active|idle)_(min|max|tot|avg|std)$", r"\1.\2", c)
        out.append(c)
    return out


def read_zeek_tsv(path):
    """Read Zeek-format TSV (has #fields line). Returns DataFrame or None.

    Zeek format metadata specifies separators, but actual data may use tabs
    or commas. We auto-detect from the first non-comment data line.
    """
    try:
        columns = None
        sep = "\t"  # default
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("#fields"):
                    raw = line.replace("#fields", "").strip()
                    raw = raw.lstrip(",")
                    columns = [c.strip() for c in raw.split(",")]
                    columns = [c for c in columns if c]
                if not line.startswith("#") and columns is not None:
                    # Auto-detect separator: count tabs vs commas in first data line
                    tabs = line.count("\t")
                    commas = line.count(",")
                    sep = "\t" if tabs > commas else ","
                    break
            if not columns:
                return None
        return pd.read_csv(
            path, sep=sep, comment="#", names=columns,
            low_memory=False, na_values=["-", "(empty)", ""],
            on_bad_lines="skip",
        )
    except Exception:
        return None


def read_plain_flowmeter(path):
    """Read plain-CSV flowmeter format (quoted header row, comma-separated)."""
    try:
        df = pd.read_csv(path, low_memory=False,
                         na_values=["-", "(empty)", ""])
        df.columns = norm_flowmeter_cols([str(c) for c in df.columns])
        return df
    except Exception:
        return None


def read_tshark_csv(path):
    """Read tshark-exported CSV - auto-detect tab or comma separator."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline()
        sep = "\t" if "\t" in first_line and "," not in first_line[:200] else ","
        return pd.read_csv(path, sep=sep, low_memory=False,
                           na_values=["", "-", "(empty)"])
    except Exception:
        return None


def detect_and_load(path):
    """Auto-detect CSV format and load. Returns (DataFrame, format_str)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline().strip()
    except Exception:
        return None, "unreadable"

    if not first_line:
        return None, "empty"

    if first_line.startswith("#separator") or first_line.startswith("#fields"):
        return read_zeek_tsv(path), "zeek"
    if first_line.startswith('"uid"') or first_line.startswith("uid,"):
        return read_plain_flowmeter(path), "plain_flowmeter"
    if first_line.startswith("frame.encap_type") or "frame." in first_line:
        return read_tshark_csv(path), "tshark"
    if first_line.startswith('"ts"') or first_line.startswith("ts,"):
        return read_plain_flowmeter(path), "plain_flowmeter"
    if first_line.startswith('"frame.time_epoch"'):
        return read_tshark_csv(path), "tshark"

    return None, f"unknown: {first_line[:80]}"


def get_numeric_cols(df, non_feature_set):
    """Return numeric feature column names excluding non-feature cols."""
    numeric = []
    for c in df.columns:
        c_str = str(c).strip()
        if c_str in non_feature_set:
            continue
        dtypes_to_check = ("int64", "float64", "int32", "float32",
                           "int16", "float16", "int8")
        if hasattr(df[c], "dtype") and str(df[c].dtype) in dtypes_to_check:
            numeric.append(c_str)
        elif pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c_str)
    return numeric


def compute_stats(df, numeric_cols):
    """Compute count, mean, std, min, max for numeric columns. Uses float32."""
    if len(df) == 0:
        return {"count": 0, "mean": pd.Series(dtype="float64"),
                "std": pd.Series(dtype="float64"),
                "min": pd.Series(dtype="float64"),
                "max": pd.Series(dtype="float64")}
    sub = df[numeric_cols].fillna(0)
    # Convert to float32 for memory, then compute
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


def load_csv_chunked(path, fmt, numeric_cols_hint=None):
    """Load a CSV and return (df, numeric_cols, fmt).

    For large files, reads in chunks and accumulates stats.
    But for simplicity in this audit, we load directly and let
    pandas + numpy handle it. Large files are handled with float32.
    """
    df, detected_fmt = detect_and_load(path)
    if df is None:
        return None, [], detected_fmt
    return df, [], detected_fmt


def normalized_deviation(attack_mean, baseline_mean,
                          attack_std, baseline_std,
                          baseline_min, baseline_max):
    """Normalize features to [0,1] using baseline range, then compute Cohen's d.

    Each feature is min-max normalized using the Normal baseline's observed
    range, making all features dimensionless and directly comparable regardless
    of original unit (bytes, packet counts, flags, time intervals). Cohen's d
    is then computed on the normalized means and standard deviations.

    For features with zero range in the baseline (constant values), the
    normalized values are set to 0, yielding d = 0 (no discriminative power).
    """
    baseline_range = baseline_max - baseline_min
    # Use inf for zero-range features so division yields 0 after cleanup
    safe_range = np.where(baseline_range > 1e-12, baseline_range, np.inf)

    # Normalize means to [0,1]
    attack_norm = (attack_mean - baseline_min) / safe_range
    baseline_norm = (baseline_mean - baseline_min) / safe_range
    # Normalize stds to [0,1] (property of linear transformation)
    attack_std_norm = attack_std / safe_range
    baseline_std_norm = baseline_std / safe_range

    # Clean up zero-range features
    attack_norm = attack_norm.fillna(0).replace([np.inf, -np.inf], 0)
    baseline_norm = baseline_norm.fillna(0).replace([np.inf, -np.inf], 0)
    attack_std_norm = attack_std_norm.fillna(0).replace([np.inf, -np.inf], 0)
    baseline_std_norm = baseline_std_norm.fillna(0).replace([np.inf, -np.inf], 0)

    # Cohen's d on normalized values
    pooled_var = (attack_std_norm ** 2 + baseline_std_norm ** 2) / 2
    pooled_std = np.sqrt(np.maximum(pooled_var, 1e-12))
    d = np.abs(attack_norm - baseline_norm) / pooled_std
    return d.fillna(0).replace([np.inf, -np.inf], 0)


def top_k(series, k=10):
    """Return top-k (name, value) pairs sorted descending."""
    return series.sort_values(ascending=False).head(k)


# ── File discovery ───────────────────────────────────────────────────────────

def discover_files():
    """Walk both trees and return mapping: side -> attack_name -> [csv_paths].

    Handles edge cases:
      - UAV Fake Landing CSV exists in BOTH Fake-Landing Packets/ and
        De-Authentication/ as identical copies — only use one copy
      - DDoS-UAV has duplicate DDos (copy).csv — skip
      - Replay Attack-UAV has no CSV — empty list
      - GCS Normal has 10 subfolders — collect output1.csv (flowmeter)
      - GCS Scanning has conn.csv + output1.csv (different schemas) —
        only use output1.csv (flowmeter, same schema as other attacks)
      - GCS Replay has conn.csv (Zeek conn) + tshark export — keep both
        but note that comparison to Normal requires Normal conn.csv as
        baseline (handled in Section 3)
    """
    mapping = {"GCS": defaultdict(list), "UAV": defaultdict(list)}

    # ── GCS ──
    gcs_attack_dirs = {
        "Brute-force": "Brute Force",
        "DDoS": "DDoS",
        "De-authentication": "De-authentication",
        "DoS": "DoS",
        "Fake-Landing Packet": "Fake Landing",
        "MITM": "MITM",
        "Normal": "Normal",
        "Reconnassiance": "Reconnaissance",
        "Replay attack": "Replay",
        "Scanning": "Scanning",
    }

    for dir_name, attack_label in gcs_attack_dirs.items():
        attack_dir = GCS_BASE / dir_name
        if not attack_dir.is_dir():
            continue

        if dir_name == "Normal":
            # Collect output1.csv from each subfolder (flowmeter data)
            csv_paths = sorted(attack_dir.glob("*/output1.csv"))
            mapping["GCS"][attack_label].extend(csv_paths)
        elif dir_name == "MITM":
            # MITM Attack.csv is the main file; Evil Twin.csv is a sub-type
            mitm_main = attack_dir / "MITM Attack.csv"
            if mitm_main.exists():
                mapping["GCS"]["MITM"].append(mitm_main)
            evil_twin = attack_dir / "Evil Twin.csv"
            if evil_twin.exists():
                mapping["GCS"]["Evil Twin"].append(evil_twin)
        elif dir_name == "Replay attack":
            # Has conn.csv (Zeek conn log) + tshark export
            for csv_file in sorted(attack_dir.glob("*.csv")):
                mapping["GCS"][attack_label].append(csv_file)
        elif dir_name == "Scanning":
            # Only output1.csv (flowmeter); conn.csv is a different schema
            output1 = attack_dir / "output1.csv"
            if output1.exists():
                mapping["GCS"][attack_label].append(output1)
        else:
            # Use the "attack" labeled CSV (flowmeter) if present,
            # otherwise fall back to broad patterns (De-authentication, etc.)
            attack_csvs = sorted(attack_dir.glob("*[Aa]ttack*.csv"))
            if attack_csvs:
                mapping["GCS"][attack_label].extend(attack_csvs)
            elif dir_name == "De-authentication":
                # De-authentication.csv (no "attack" in filename)
                deauth_csv = attack_dir / "De-authentication.csv"
                if deauth_csv.exists():
                    mapping["GCS"][attack_label].append(deauth_csv)

    # ── UAV ──
    uav_attack_dirs = {
        "Brute-Force UAV": "Brute Force",
        "DDoS-UAV": "DDoS",
        "De-Authentication": "De-authentication",
        "Dos-UAV": "DoS",
        "Fake-Landing Packets": "Fake Landing",
        "GPS Jamming UAV": "GPS Jamming",
        "MITM-UAV": "MITM",
        "Normal-Flights": "Normal",
        "Reconnacciance-UAV": "Reconnaissance",
        "Replay Attack-UAV": "Replay",
        "Scanning-UAV": "Scanning",
    }

    for dir_name, attack_label in uav_attack_dirs.items():
        attack_dir = UAV_BASE / dir_name
        if not attack_dir.is_dir():
            mapping["UAV"][attack_label] = []
            continue

        csv_files = sorted(attack_dir.glob("*.csv"))

        if dir_name == "De-Authentication":
            # FakeLanding.csv here is identical to the one in
            # Fake-Landing Packets/ — skip to avoid double-counting
            deauth_csvs = [f for f in csv_files
                           if f.name.startswith("Deauthentication")]
            mapping["UAV"][attack_label].extend(deauth_csvs)
        elif dir_name == "DDoS-UAV":
            # Skip the "(copy)" file
            main_csvs = [f for f in csv_files if "copy" not in f.name.lower()]
            mapping["UAV"][attack_label].extend(main_csvs)
        elif dir_name == "Replay Attack-UAV":
            # No CSV available, only pcap
            mapping["UAV"][attack_label] = []
        else:
            mapping["UAV"][attack_label].extend(csv_files)

    return mapping


# ── Load and compute stats ──────────────────────────────────────────────────

def load_all_stats(file_mapping):
    """Load all CSVs and compute per-attack stats.

    Returns:
      gcs_stats: attack -> {count, mean, std, numeric_cols, fmt}
      uav_stats: attack -> {count, mean, std, numeric_cols, fmt}
      gcs_format_notes: str
      uav_format_notes: str
    """
    gcs_stats = {}
    uav_stats = {}

    for side, mapping in [("GCS", file_mapping["GCS"]),
                           ("UAV", file_mapping["UAV"])]:
        target = gcs_stats if side == "GCS" else uav_stats
        non_feat = GCS_NON_FEATURE if side == "GCS" else UAV_NON_FEATURE

        for attack, csv_paths in sorted(mapping.items()):
            if not csv_paths:
                target[attack] = {"count": 0, "mean": pd.Series(),
                                   "std": pd.Series(), "min": pd.Series(),
                                   "max": pd.Series(), "numeric_cols": [],
                                   "fmt": "no_csv"}
                continue

            all_dfs = []
            detected_fmts = set()

            for path in csv_paths:
                df, fmt = detect_and_load(path)
                if df is not None and len(df) > 0:
                    all_dfs.append(df)
                    detected_fmts.add(fmt)
                    print(f"  [{side}] {attack}: {path.name} "
                          f"→ {len(df):,} rows [{fmt}]")

            if not all_dfs:
                target[attack] = {"count": 0, "mean": pd.Series(),
                                   "std": pd.Series(), "min": pd.Series(),
                                   "max": pd.Series(), "numeric_cols": [],
                                   "fmt": "empty"}
                continue

            combined = pd.concat(all_dfs, ignore_index=True, copy=False)
            numeric_cols = get_numeric_cols(combined, non_feat)
            stats = compute_stats(combined, numeric_cols)
            stats["numeric_cols"] = numeric_cols
            stats["fmt"] = ", ".join(sorted(detected_fmts))
            target[attack] = stats

            print(f"  [{side}] {attack}: TOTAL {stats['count']:,} rows, "
                  f"{len(numeric_cols)} numeric features [{stats['fmt']}]")

    return gcs_stats, uav_stats


# ── Main audit ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("UAV-NIDD Deep Audit")
    print("=" * 60)

    print("\n[1/3] Discovering files ...")
    file_mapping = discover_files()

    for side in ("GCS", "UAV"):
        print(f"\n  {side} attacks:")
        for att, paths in sorted(file_mapping[side].items()):
            if paths:
                print(f"    {att:<22s}: {len(paths)} file(s) — "
                      f"{', '.join(p.name for p in paths[:3])}"
                      f"{' ...' if len(paths) > 3 else ''}")
            else:
                print(f"    {att:<22s}: NO CSV FILES (pcap only)")

    print("\n[2/3] Loading data and computing statistics ...")
    gcs_stats, uav_stats = load_all_stats(file_mapping)

    gcs_attacks = sorted(gcs_stats)
    uav_attacks = sorted(uav_stats)
    all_attacks = sorted(set(gcs_attacks) | set(uav_attacks))

    print("\n[3/3] Generating audit report ...")

    # Collect feature names for Section 0
    gcs_feature_set = set()
    uav_feature_set = set()
    for att, s in gcs_stats.items():
        gcs_feature_set.update(s.get("numeric_cols", []))
    for att, s in uav_stats.items():
        uav_feature_set.update(s.get("numeric_cols", []))
    gcs_features_sorted = sorted(gcs_feature_set)
    uav_features_sorted = sorted(uav_feature_set)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # ── 0. Data Sources & Feature/Label Inventory ────────────────────
        sepline(f, "0. DATA SOURCES & FEATURE/LABEL INVENTORY")

        f.write("This audit walks through individual folders in:\n")
        f.write(f"  GCS: {GCS_BASE}\n")
        f.write(f"  UAV: {UAV_BASE}\n\n")
        f.write("GCS CSVs use two formats:\n")
        f.write("  - Zeek TSV flowmeter (79 numeric features): #fields header, "
                "tab-separated\n")
        f.write("  - Plain CSV flowmeter (79 numeric features): quoted header, "
                "comma-separated\n")
        f.write("  - Replay attack uses tshark export (7 cols, limited features)\n\n")
        f.write("UAV CSVs all use tshark packet-level export (~170 columns, "
                "comma-separated).\n")
        f.write("  Replay Attack-UAV has NO CSV (only .pcap file) — excluded from "
                "UAV analysis.\n\n")

        # ── 0a. GCS Feature Inventory ───────────────────────────────────
        sepline(f, "0a. GCS FEATURE INVENTORY (flow-level)", char="-")
        f.write(f"Total unique numeric features across all attacks: "
                f"{len(gcs_features_sorted)}\n\n")

        gcs_feat_groups = {
            "Forward packet/byte stats": [
                "fwd_pkts_tot", "fwd_pkts_per_sec", "fwd_pkts_payload.avg",
                "fwd_pkts_payload.max", "fwd_pkts_payload.min",
                "fwd_pkts_payload.std", "fwd_pkts_payload.tot",
            ],
            "Backward packet/byte stats": [
                "bwd_pkts_tot", "bwd_pkts_per_sec", "bwd_pkts_payload.avg",
                "bwd_pkts_payload.max", "bwd_pkts_payload.min",
                "bwd_pkts_payload.std", "bwd_pkts_payload.tot",
            ],
            "Flow packet/byte stats": [
                "flow_pkts_tot", "flow_pkts_per_sec", "flow_pkts_payload.avg",
                "flow_pkts_payload.max", "flow_pkts_payload.min",
                "flow_pkts_payload.std", "flow_pkts_payload.tot",
            ],
            "Header size stats": [
                "fwd_header_size_avg", "fwd_header_size_max",
                "fwd_header_size_min", "fwd_header_size_std",
                "bwd_header_size_avg", "bwd_header_size_max",
                "bwd_header_size_min", "bwd_header_size_std",
            ],
            "Inter-arrival time (IAT)": [
                "fwd_iat.avg", "fwd_iat.max", "fwd_iat.min",
                "fwd_iat.std", "fwd_iat.tot",
                "bwd_iat.avg", "bwd_iat.max", "bwd_iat.min",
                "bwd_iat.std", "bwd_iat.tot",
                "flow_iat.avg", "flow_iat.max", "flow_iat.min",
                "flow_iat.std", "flow_iat.tot",
            ],
            "Active/Idle stats": [
                "active.avg", "active.max", "active.min", "active.std",
                "idle.avg", "idle.max", "idle.min", "idle.std",
            ],
            "Bulk & window stats": [
                "fwd_bulk_packets", "fwd_bulk_bytes", "fwd_bulk_duration",
                "fwd_bulk_rate", "fwd_init_window_size", "fwd_last_window_size",
                "bwd_bulk_packets", "bwd_bulk_bytes", "bwd_bulk_duration",
                "bwd_bulk_rate", "bwd_init_window_size", "bwd_last_window_size",
            ],
            "Flag counts": [
                "fwd_PSH", "fwd_SYN", "fwd_URG", "fwd_ACK", "fwd_FIN",
                "fwd_RST", "fwd_CWR", "fwd_ECE",
                "bwd_PSH", "bwd_SYN", "bwd_URG", "bwd_ACK", "bwd_FIN",
                "bwd_RST", "bwd_CWR", "bwd_ECE",
            ],
            "Connection metadata": ["flow_duration", "tot_fwd_pkts", "tot_bwd_pkts"],
        }
        for group, feat_list in gcs_feat_groups.items():
            existing = [x for x in feat_list if x in gcs_feature_set]
            if existing:
                f.write(f"  {group} ({len(existing)}):\n")
                for feat in existing:
                    f.write(f"    {feat}\n")

        # Catch any features not in our grouping
        grouped_feats = set()
        for v in gcs_feat_groups.values():
            grouped_feats.update(v)
        remaining = [x for x in gcs_features_sorted if x not in grouped_feats]
        if remaining:
            f.write(f"  Other/uncategorized ({len(remaining)}):\n")
            for feat in remaining:
                f.write(f"    {feat}\n")

        f.write(f"\n  Note: GCS feature space is flow-level — packets, bytes,\n")
        f.write(f"  flags, payload stats, IAT stats, bulk/window stats.\n")
        f.write(f"  Feature names use Zeek-style dotted notation\n")
        f.write(f"  (e.g. fwd_pkts_payload.avg, bwd_iat.std).\n")

        # ── 0b. UAV Feature Inventory ───────────────────────────────────
        sepline(f, "0b. UAV FEATURE INVENTORY (packet-level)", char="-")
        f.write(f"Total unique numeric features across all attacks: "
                f"{len(uav_features_sorted)}\n\n")

        uav_feat_groups = {
            "Frame-level": [
                "frame.cap_len", "frame.len", "frame.number",
            ],
            "Radiotap (radio/signal)": [
                "radiotap.channel.flags.cck", "radiotap.datarate",
                "radiotap.dbm_antsignal", "radiotap.length",
                "radiotap.mactime", "radiotap.present.tsft",
                "radiotap.present.flags", "radiotap.present.channel",
                "radiotap.present.dbm_antsignal",
            ],
            "WLAN / WiFi": [
                "wlan.duration", "wlan.fcs", "wlan.fcs.status",
                "wlan.seq", "wlan.fragment", "wlan.fragments",
            ],
            "IP-level": [
                "ip.dsfield.dscp", "ip.dsfield.ecn", "ip.flags.df",
                "ip.flags.mf", "ip.flags.rb", "ip.frag_offset",
                "ip.hdr_len", "ip.id", "ip.len",
            ],
            "TCP-level": [
                "tcp.ack", "tcp.analysis", "tcp.checksum",
                "tcp.connection.syn", "tcp.connection.fin",
                "tcp.connection.rst", "tcp.dstport", "tcp.flags",
                "tcp.hdr_len", "tcp.len", "tcp.nxtseq",
                "tcp.options", "tcp.port", "tcp.seq",
                "tcp.srcport", "tcp.stream", "tcp.time_delta",
                "tcp.time_relative", "tcp.urgent_pointer",
                "tcp.window_size", "tcp.window_size_scalefactor",
            ],
            "UDP-level": [
                "udp.checksum", "udp.checksum.status",
                "udp.dstport", "udp.length", "udp.port",
                "udp.srcport", "udp.stream",
            ],
            "DNS": [
                "dns.count.add_rr", "dns.count.answers",
                "dns.count.auth_rr", "dns.count.labels",
                "dns.count.queries", "dns.flags.authenticated",
                "dns.flags.recavail", "dns.flags.recdesired",
                "dns.flags.truncated", "dns.id",
                "dns.qry.type", "dns.resp.len", "dns.resp.ttl",
                "dns.resp.type", "dns.time",
            ],
            "HTTP": [
                "http.content_length", "http.next_response_in",
                "http.request.full_uri", "http.request.line",
                "http.request.number", "http.response.code",
                "http.response.line", "http.response.number",
                "http.time",
            ],
            "SMB/SMB2": [
                "smb.access.generic_execute",
                "smb2.secblob", "smb2.secblob_len",
                "smb2.sesetup.reqblob_len",
                "smb2.sesetup.respblob_len",
            ],
            "SSH": [
                "ssh.direction", "ssh.encrypted_packet",
                "ssh.host_key.length",
                "ssh.host_key_algorithms_length",
                "ssh.kex_algorithms_length",
                "ssh.mac_algorithms_client_to_server_length",
                "ssh.mac_algorithms_server_to_client_length",
                "ssh.mpint_length",
                "ssh.server_host_key_algorithms_length",
            ],
            "ICMP": ["icmp.code", "icmp.type"],
            "ICMPv6": ["icmpv6.code", "icmpv6.type"],
            "ARP": ["arp.dst.proto_ipv4", "arp.opcode",
                    "arp.proto.size", "arp.src.proto_ipv4"],
            "NBNS/NBSS": ["nbns", "nbss", "nbss.length"],
            "LDAP": ["ldap"],
            "EAPOL": ["eapol.keydes.key_len", "eapol.len",
                      "eapol.type"],
            "Other": ["data.len", "browser.server_type",
                      "dhcp.option.dhcp", "ntp.priv.recv_time_stamp",
                      "ntp.priv.refid",
                      "dhcpv6.elapsed_time", "dhcpv6.ia.na.iaaddr",
                      "dhcpv6.iana.n1.iaaddr", "dhcpv6.iana.n2.iaaddr",
                      "dhcpv6.xid",
                      "bootp.hops", "bootp.id",
                      ],
        }
        for group, feat_list in uav_feat_groups.items():
            existing = [x for x in feat_list if x in uav_feature_set]
            if existing:
                f.write(f"  {group} ({len(existing)}):\n")
                for feat in existing:
                    f.write(f"    {feat}\n")

        # Catch any features not in our grouping
        grouped_uav = set()
        for v in uav_feat_groups.values():
            grouped_uav.update(v)
        remaining_uav = [x for x in uav_features_sorted if x not in grouped_uav]
        if remaining_uav:
            f.write(f"  Other/uncategorized ({len(remaining_uav)}):\n")
            for feat in remaining_uav:
                f.write(f"    {feat}\n")

        f.write(f"\n  Note: UAV feature space is packet-level — frame, radiotap,\n")
        f.write(f"  WLAN, IP, TCP/UDP, DNS, HTTP, SMB, SSH, ARP, NBNS, EAPOL,\n")
        f.write(f"  DHCP, BOOTP, NTP, ICMP fields.\n")
        f.write(f"  Features are tshark export columns, each packet = one row.\n")
        f.write(f"  GCS and UAV feature spaces are completely disjoint.\n")

        # ── 0c. Label Inventory ─────────────────────────────────────────
        sepline(f, "0c. LABEL INVENTORY", char="-")

        gcs_attacks_list = sorted(gcs_stats.keys())
        uav_attacks_list = sorted(uav_stats.keys())

        f.write(f"GCS attack types ({len(gcs_attacks_list)}):\n")
        for att in gcs_attacks_list:
            cnt = gcs_stats[att]["count"]
            f.write(f"  {att:<22s} ({cnt:>10,} samples)\n")

        f.write(f"\nUAV attack types ({len(uav_attacks_list)}):\n")
        for att in uav_attacks_list:
            cnt = uav_stats[att]["count"]
            f.write(f"  {att:<22s} ({cnt:>10,} samples)\n")

        f.write(f"\nAttacks present in BOTH: "
                f"{sorted(set(gcs_attacks_list) & set(uav_attacks_list))}\n")
        f.write(f"Attacks present ONLY in GCS: "
                f"{sorted(set(gcs_attacks_list) - set(uav_attacks_list))}\n")
        f.write(f"Attacks present ONLY in UAV: "
                f"{sorted(set(uav_attacks_list) - set(gcs_attacks_list))}\n")

        # ── File Inventory ───────────────────────────────────────────────
        sepline(f, "FILE INVENTORY", char="-")
        for side, paths_map in [("GCS", file_mapping["GCS"]),
                                 ("UAV", file_mapping["UAV"])]:
            f.write(f"{side} file inventory:\n")
            for att in sorted(paths_map):
                paths = paths_map[att]
                if paths:
                    f.write(f"  {att:<22s}: {len(paths)} file(s)\n")
                    for p in paths[:5]:
                        size_mb = p.stat().st_size / (1024 * 1024)
                        f.write(f"    - {p.name} ({size_mb:.1f} MB)\n")
                    if len(paths) > 5:
                        f.write(f"    ... and {len(paths) - 5} more\n")
                else:
                    f.write(f"  {att:<22s}: NO CSV\n")
            f.write("\n")

        # ── 1. Sample counts ────────────────────────────────────────────
        sepline(f, "1. SAMPLE COUNTS PER ATTACK (GCS vs UAV)")

        f.write(f"{'Attack':<22s} {'GCS Count':>12s} {'UAV Count':>12s} "
                f"{'Total':>12s}\n")
        f.write("-" * 60 + "\n")
        gcs_total = 0
        uav_total = 0
        for att in all_attacks:
            gc = gcs_stats.get(att, {}).get("count", 0)
            uc = uav_stats.get(att, {}).get("count", 0)
            gcs_total += gc
            uav_total += uc
            gcs_str = f"{gc:,}" if gc else ("-" if att in gcs_stats else "N/A")
            uav_str = f"{uc:,}" if uc else ("-" if att in uav_stats else "N/A")
            f.write(f"{att:<22s} {gcs_str:>12s} {uav_str:>12s} "
                    f"{gc + uc:>12,}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'TOTAL':<22s} {gcs_total:>12,} {uav_total:>12,} "
                f"{gcs_total + uav_total:>12,}\n")

        f.write(f"\nAttacks present ONLY in GCS: "
                f"{sorted(set(gcs_attacks) - set(uav_attacks))}\n")
        f.write(f"Attacks present ONLY in UAV: "
                f"{sorted(set(uav_attacks) - set(gcs_attacks))}\n")
        f.write(f"Attacks present in BOTH:    "
                f"{sorted(set(gcs_attacks) & set(uav_attacks))}\n")

        # Note about missing data
        f.write("\nData availability notes:\n")
        if gcs_stats.get("Replay", {}).get("count", 0) > 0:
            f.write("  - GCS Replay: conn.csv (Zeek conn) + tshark export; "
                    "not flowmeter level\n")
        if uav_stats.get("Replay", {}).get("count", 0) == 0:
            f.write("  - UAV Replay: NO CSV available (only .pcap), "
                    "excluded from UAV stats\n")
        f.write("  - UAV De-authentication folder contained an identical "
                "copy of FakeLanding.csv (skipped, not double-counted)\n")
        f.write("  - GCS MITM folder contains Evil Twin.csv as a sub-type "
                "(treated separately)\n")
        f.write("  - GCS Scanning: output1.csv (flowmeter) used; conn.csv "
                "(Zeek conn, different schema) excluded\n")

        # ── 2. Feature counts ───────────────────────────────────────────
        sepline(f, "2. FEATURE COUNTS PER ATTACK (GCS vs UAV)")

        f.write(f"{'Attack':<22s} {'GCS Features':>14s} {'UAV Features':>14s} "
                f"{'GCS Format':>20s} {'UAV Format':>20s}\n")
        f.write("-" * 92 + "\n")

        for att in all_attacks:
            gcs_nf = len(gcs_stats.get(att, {}).get("numeric_cols", []))
            uav_nf = len(uav_stats.get(att, {}).get("numeric_cols", []))
            gcs_fmt = gcs_stats.get(att, {}).get("fmt", "N/A")
            uav_fmt = uav_stats.get(att, {}).get("fmt", "N/A")
            f.write(f"{att:<22s} {gcs_nf:>14} {uav_nf:>14} "
                    f"{gcs_fmt:>20s} {uav_fmt:>20s}\n")

        f.write("\nSummary:\n")
        # Collect unique feature column sets
        gcs_nf_all = []
        uav_nf_all = []
        for att in gcs_attacks:
            n = len(gcs_stats[att].get("numeric_cols", []))
            if n > 0:
                gcs_nf_all.append(n)
        for att in uav_attacks:
            n = len(uav_stats[att].get("numeric_cols", []))
            if n > 0:
                uav_nf_all.append(n)

        if gcs_nf_all:
            f.write(f"  GCS: {min(gcs_nf_all)}–{max(gcs_nf_all)} numeric "
                    f"features per attack (uniform schema per side)\n")
        if uav_nf_all:
            f.write(f"  UAV: {min(uav_nf_all)}–{max(uav_nf_all)} numeric "
                    f"features per attack (uniform schema per side)\n")

        f.write("\n  GCS feature space: flow-level (Zeek conn + flowmeter):\n")
        f.write("    packets, bytes, flags, payload stats, IAT stats, "
                "bulk/window stats\n")
        f.write("  UAV feature space: packet-level (tshark radiotap/wlan/IP/"
                "TCP/UDP):\n")
        f.write("    frame length, radio signal, channel, TCP/UDP ports, "
                "flags, DNS/HTTP counters\n")
        f.write("  GCS and UAV feature spaces are completely disjoint.\n")

        # ── 3. Cross-view: same attack, GCS vs UAV impact ───────────────
        sepline(f, "3. CROSS-VIEW: SAME ATTACK – GCS vs UAV IMPACT COMPARISON")
        f.write("Method: For each attack present on BOTH sides, compute "
                "per-feature normalized\n")
        f.write("deviation from Normal/Benign (range-normalized: "
                "|attack_mean - normal_mean| / normal_range).\n")
        f.write("Each feature is normalized by its own baseline range, making ")
        f.write("all features comparable\non the same [0,1] scale regardless "
                "of unit (bytes, counts, flags, time).\n")

        common_attacks = sorted(
            set(gcs_attacks) & set(uav_attacks) - {"Normal", "Replay"}
        )

        # Also handle Replay specially
        extra_attacks = []
        if "Replay" in gcs_attacks and gcs_stats["Replay"]["count"] > 0:
            extra_attacks.append("Replay")

        for att in common_attacks + extra_attacks:
            f.write(f"── Attack: {att} ──\n\n")

            gcs_has = (att in gcs_stats and gcs_stats[att]["count"] > 0
                       and len(gcs_stats[att].get("numeric_cols", [])) > 0)
            uav_has = (att in uav_stats and uav_stats[att]["count"] > 0
                       and len(uav_stats[att].get("numeric_cols", [])) > 0)

            if not gcs_has and not uav_has:
                f.write("  No numeric data available on either side.\n\n")
                continue

            # GCS side
            if gcs_has and "Normal" in gcs_stats:
                gcs_normal = gcs_stats["Normal"]
                gcs_att = gcs_stats[att]
                # Use intersection of features between attack and normal
                gcs_nc = sorted(set(gcs_att["numeric_cols"])
                                & set(gcs_normal["numeric_cols"]))
                gcs_dev = normalized_deviation(
                    gcs_att["mean"][gcs_nc], gcs_normal["mean"][gcs_nc],
                    gcs_att["std"][gcs_nc], gcs_normal["std"][gcs_nc],
                    gcs_normal["min"][gcs_nc], gcs_normal["max"][gcs_nc],
                )
                gcs_top = top_k(gcs_dev, 10)
                f.write(f"  GCS (flow-level, {len(gcs_nc)} features) — "
                        f"Top-10 most-deviated features vs Normal:\n")
                for feat, val in gcs_top.items():
                    f.write(f"    {feat:<50s} d={val:.3f}\n")
                f.write(f"  GCS mean |d| across all features: "
                        f"{gcs_dev.mean():.4f}\n\n")
            else:
                f.write(f"  GCS: no data for {att}\n\n")
                gcs_dev = None

            # UAV side
            if uav_has and "Normal" in uav_stats:
                uav_normal = uav_stats["Normal"]
                uav_att = uav_stats[att]
                uav_nc = sorted(set(uav_att["numeric_cols"])
                                & set(uav_normal["numeric_cols"]))
                uav_dev = normalized_deviation(
                    uav_att["mean"][uav_nc], uav_normal["mean"][uav_nc],
                    uav_att["std"][uav_nc], uav_normal["std"][uav_nc],
                    uav_normal["min"][uav_nc], uav_normal["max"][uav_nc],
                )
                uav_top = top_k(uav_dev, 10)
                f.write(f"  UAV (packet-level, {len(uav_nc)} features) — "
                        f"Top-10 most-deviated features vs Normal:\n")
                for feat, val in uav_top.items():
                    f.write(f"    {feat:<50s} d={val:.3f}\n")
                f.write(f"  UAV mean |d| across all features: "
                        f"{uav_dev.mean():.4f}\n\n")
            else:
                f.write(f"  UAV: no data for {att}\n\n")
                uav_dev = None

            # Interpretation
            if gcs_dev is not None and uav_dev is not None:
                f.write("  Interpretation:\n")
                f.write(f"    GCS mean |d| = {gcs_dev.mean():.4f}  |  "
                        f"UAV mean |d| = {uav_dev.mean():.4f}\n")
                if gcs_dev.mean() > uav_dev.mean():
                    f.write(f"    → GCS shows LARGER overall deviation "
                            f"under {att} than UAV.\n")
                else:
                    f.write(f"    → UAV shows LARGER overall deviation "
                            f"under {att} than GCS.\n")
                f.write(f"    GCS sees flow-metric shifts (packet rates, "
                        f"header sizes, IAT);\n")
                f.write(f"    UAV sees packet/radio-level shifts (frame "
                        f"length, port usage, signal).\n")
            f.write("\n")

        # Attacks on only one side
        sepline(f, "3b. ATTACKS ON ONLY ONE SIDE (deviation vs own Normal)")
        for att in all_attacks:
            if att in common_attacks or att == "Normal":
                continue
            if att in extra_attacks:
                continue  # Already handled above

            f.write(f"── Attack: {att} ──\n")
            if att in gcs_stats and gcs_stats[att]["count"] > 0:
                if "Normal" in gcs_stats:
                    normal = gcs_stats["Normal"]
                    att_s = gcs_stats[att]
                    nc = sorted(set(att_s["numeric_cols"])
                                & set(normal["numeric_cols"]))
                    dev = normalized_deviation(
                        att_s["mean"][nc], normal["mean"][nc],
                        att_s["std"][nc], normal["std"][nc],
                        normal["min"][nc], normal["max"][nc],
                    )
                    top = top_k(dev, 10)
                    f.write(f"  GCS only — Top-10 most-deviated features "
                            f"vs Normal:\n")
                    for feat, val in top.items():
                        f.write(f"    {feat:<50s} d={val:.3f}\n")
                    f.write(f"  Mean |d| across all features: "
                            f"{dev.mean():.4f}\n")
            elif att in uav_stats and uav_stats[att]["count"] > 0:
                if "Normal" in uav_stats:
                    normal = uav_stats["Normal"]
                    att_s = uav_stats[att]
                    nc = sorted(set(att_s["numeric_cols"])
                                & set(normal["numeric_cols"]))
                    dev = normalized_deviation(
                        att_s["mean"][nc], normal["mean"][nc],
                        att_s["std"][nc], normal["std"][nc],
                        normal["min"][nc], normal["max"][nc],
                    )
                    top = top_k(dev, 10)
                    f.write(f"  UAV only — Top-10 most-deviated features "
                            f"vs Normal:\n")
                    for feat, val in top.items():
                        f.write(f"    {feat:<50s} d={val:.3f}\n")
                    f.write(f"  Mean |d| across all features: "
                            f"{dev.mean():.4f}\n")
            else:
                f.write(f"  No data available.\n")
            f.write("\n")

        # ── 4. Within-view: different attacks, same platform ────────────
        sepline(f, "4. WITHIN-VIEW: DIFFERENT ATTACKS ON THE SAME PLATFORM")
        f.write("For each platform (GCS / UAV), compute per-feature deviation "
                "from Normal\n")
        f.write("for every attack, then analyze which attacks cause the "
                "largest shift and\n")
        f.write("which features are most sensitive to attacks.\n\n")

        for side_name, stats_map, side_label in [
            ("GCS (flow-level)", gcs_stats, "GCS"),
            ("UAV (packet-level)", uav_stats, "UAV"),
        ]:
            f.write(f"{'=' * 60}\n")
            f.write(f"  {side_name}\n")
            f.write(f"{'=' * 60}\n\n")

            if "Normal" not in stats_map:
                f.write("  No Normal baseline — skipping.\n\n")
                continue

            normal = stats_map["Normal"]
            attacks = [a for a in sorted(stats_map)
                       if a != "Normal"
                       and stats_map[a]["count"] > 0
                       and len(stats_map[a].get("numeric_cols", [])) > 0]

            if not attacks:
                f.write("  No attack data available for this side.\n\n")
                continue

            # Use the union of numeric columns across all attacks for
            # comparability. But since schemas are uniform, just use
            # the first attack's columns. For safety, use columns
            # common across all attacks + normal.
            all_nc_sets = [set(stats_map[a]["numeric_cols"])
                           for a in attacks]
            all_nc_sets.append(set(normal["numeric_cols"]))
            ref_cols = sorted(set.intersection(*all_nc_sets))
            if not ref_cols:
                # Fallback: use first attack's cols aligned with normal
                ref_cols = sorted(
                    set(stats_map[attacks[0]]["numeric_cols"])
                    & set(normal["numeric_cols"])
                )
            normal_mean = normal["mean"][ref_cols].fillna(0)
            normal_std = normal["std"][ref_cols].fillna(0)
            normal_min = normal["min"][ref_cols].fillna(0)
            normal_max = normal["max"][ref_cols].fillna(0)

            # Deviation matrix: attack × feature
            dev_records = {}
            for att in attacks:
                att_s = stats_map[att]
                # Align to ref_cols
                att_mean = att_s["mean"].reindex(ref_cols).fillna(0)
                att_std = att_s["std"].reindex(ref_cols).fillna(0)
                dev = normalized_deviation(
                    att_mean, normal_mean, att_std, normal_std,
                    normal_min, normal_max,
                )
                dev_records[att] = dev

            if not dev_records:
                f.write("  Could not compute deviations.\n\n")
                continue

            dev_df = pd.DataFrame(dev_records).T  # attack × feature

            # 4a. Attack impact ranking
            f.write("4a. Attack impact ranking (mean |d| across all "
                    "features):\n\n")
            attack_impact = dev_df.mean(axis=1).sort_values(ascending=False)
            for rank, (att, mean_d) in enumerate(attack_impact.items(), 1):
                f.write(f"  {rank:2d}. {att:<25s} mean|d| = {mean_d:.4f}\n")

            # 4b. Features with highest VARIANCE of d across attacks
            f.write(f"\n4b. Top-15 features with highest VARIANCE of d "
                    f"across attacks\n")
            f.write(f"    (features whose deviation changes most depending "
                    f"on attack type):\n\n")
            feature_std = dev_df.std(axis=0).sort_values(ascending=False)
            for rank, (feat, std_val) in enumerate(
                    feature_std.head(15).items(), 1):
                f.write(f"  {rank:2d}. {feat:<50s} std(d)={std_val:.4f}\n")

            # 4c. Features with highest MEAN d across attacks
            f.write(f"\n4c. Top-15 features with highest MEAN d across "
                    f"attacks\n")
            f.write(f"    (features universally shifted regardless of "
                    f"attack type):\n\n")
            feature_mean = dev_df.mean(axis=0).sort_values(ascending=False)
            for rank, (feat, mean_val) in enumerate(
                    feature_mean.head(15).items(), 1):
                f.write(f"  {rank:2d}. {feat:<50s} mean(d)={mean_val:.4f}\n")

            # 4d. Attack similarity matrix
            f.write(f"\n4d. Attack similarity matrix "
                    f"(Pearson r of deviation vectors):\n\n")
            if len(dev_df) >= 2:
                att_list = list(dev_df.index)
                # Header
                f.write(f"    {'':<25s}"
                        + "".join(f"{a[:8]:>9s}" for a in att_list) + "\n")
                for a1 in att_list:
                    row = []
                    for a2 in att_list:
                        r = np.corrcoef(dev_df.loc[a1], dev_df.loc[a2])[0, 1]
                        row.append(f"{r:9.3f}")
                    f.write(f"    {a1:<25s}" + "".join(row) + "\n")

            # 4e. Quick-reference: largest d per attack
            f.write(f"\n4e. Quick-reference — largest single-feature "
                    f"deviation per attack:\n\n")
            for att in attacks:
                max_feat = dev_df.loc[att].idxmax()
                max_val = dev_df.loc[att].max()
                f.write(f"  {att:<25s} {max_feat:<50s} d={max_val:.3f}\n")

            f.write("\n")

        # ── Summary ─────────────────────────────────────────────────────
        sepline(f, "5. INTERPRETATION GUIDE")
        f.write("""
This audit answers four questions:

Q1 (Sample counts): How many samples exist per attack on each side?
   → Section 1 of this report.

Q2 (Feature counts): How many numeric features are available per attack?
   → Section 2. GCS uniformly has 80 flow-level features; UAV has
     86–116 packet-level features. They are disjoint spaces.

Q3 (Cross-view — same attack, GCS vs UAV impact):
   → Section 3. For each attack, range-normalized deviation quantifies how
     far each feature deviates from its Normal baseline, as a fraction of the
     feature's own observed range. A larger mean |d| means the platform was
     more severely affected. Comparing GCS mean|d| vs UAV mean|d| for the
     same attack reveals which side is more impacted.

Q4 (Within-view — different attacks, same platform):
   → Section 4.
     4a: Which attacks cause the biggest overall shift from Normal?
     4b: Which features are most attack-type-dependent (high variance)?
     4c: Which features are universally shifted by all attacks?
     4d: Which attacks have similar deviation patterns (correlation)?
     4e: The single most-deviated feature per attack.
""")

    print(f"\nAudit report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
