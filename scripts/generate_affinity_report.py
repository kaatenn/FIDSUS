"""Generate final FIDSUS affinity diagnosis report.

Reads all processed CSV metrics and raw logs, generates:
  audit/fidsus_real_training_diagnosis/fidsus_real_training_affinity_report.md

Usage:
  uv run python scripts/generate_affinity_report.py \
      --input-dir audit/fidsus_real_training_diagnosis
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path):
    if not path.exists():
        return []
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Generate FIDSUS affinity diagnosis report")
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to diagnosis output directory')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    raw_dir = input_dir / "raw"
    processed_dir = input_dir / "processed"
    metrics_dir = input_dir / "metrics"
    report_path = input_dir / "fidsus_real_training_affinity_report.md"

    # Read data
    profiles = read_csv(raw_dir / "client_label_profile.csv")
    topn_raw = read_csv(raw_dir / "topn_selection_log.csv")
    affinity_raw = read_csv(raw_dir / "affinity_update_log.csv")
    pred_raw = read_csv(raw_dir / "prediction_log_fine.csv")
    topn_diag = read_csv(processed_dir / "topn_diagnosis_by_round.csv")
    fine_metrics = read_csv(metrics_dir / "fine_grained_metrics.csv")
    family_metrics = read_csv(metrics_dir / "family_level_metrics.csv")
    gap = read_csv(metrics_dir / "family_fine_gap.csv")
    intra = read_csv(metrics_dir / "intra_family_confusion_pairs.csv")
    sim_match = read_csv(metrics_dir / "confusion_pair_similarity_match.csv")

    with open(report_path, 'w') as f:
        f.write("# FIDSUS Affinity/Top-N Real Training Diagnosis Report\n\n")
        f.write(f"**Source**: real_training\n")
        f.write(f"**Generated from**: `{input_dir}`\n\n")

        # ── 1. Status ──────────────────────────────────────────────────────
        f.write("## 1. Logging Status\n\n")

        checks = [
            ("topn_selection_log.csv", len(topn_raw) > 0),
            ("affinity_update_log.csv", len(affinity_raw) > 0),
            ("client_label_profile.csv", len(profiles) > 0),
            ("prediction_log_fine.csv", len(pred_raw) > 0),
            ("topn_diagnosis_by_round.csv", len(topn_diag) > 0),
            ("fine_grained_metrics.csv", len(fine_metrics) > 0),
            ("family_level_metrics.csv", len(family_metrics) > 0),
            ("family_fine_gap.csv", len(gap) > 0),
            ("intra_family_confusion_pairs.csv", len(intra) > 0),
        ]
        for name, ok in checks:
            status = "SUCCESS" if ok else "MISSING"
            f.write(f"- [{status}] `{name}` ({len(locals().get(name.replace('.csv','')[:20], [])) if ok else 0} rows)\n")

        # ── 2. Sanity Check ────────────────────────────────────────────────
        f.write("\n## 2. Sanity Check (placeholder)\n\n")
        f.write("Compare test accuracy curves with/without diagnosis logging:\n\n")
        f.write("- Control run results should be found in `../results/` directory\n")
        f.write("- Diagnosis run results should be in `../results/` (same naming, with `_diagnosis` goal suffix)\n")
        f.write("- If accuracy curves match within floating point tolerance (1e-6), logging does not alter training.\n\n")

        if fine_metrics:
            last = fine_metrics[-1]
            f.write(f"Final fine-grained accuracy (diagnosis run): {last.get('accuracy', 'N/A')}\n")
            f.write(f"Final fine-grained macro F1 (diagnosis run): {last.get('macro_f1', 'N/A')}\n")

        # ── 3. Top-N vs Random Neighbors Comparison ────────────────────────
        f.write("\n## 3. FIDSUS Top-N vs Random Neighbors\n\n")

        if topn_diag:
            _write_section3(f, topn_diag)

        # ── 4. Affinity Correlations ───────────────────────────────────────
        f.write("\n## 4. Affinity Correlations\n\n")

        if affinity_raw and profiles:
            _write_section4(f, affinity_raw, profiles)

        # ── 5. Fine-Grained vs Family-Level Results ────────────────────────
        f.write("\n## 5. Fine-Grained vs Family-Level Results\n\n")

        if fine_metrics and family_metrics:
            _write_section5(f, fine_metrics, family_metrics)

        # ── 6. Family-Fine Gap ─────────────────────────────────────────────
        f.write("\n## 6. Family-Fine Accuracy Gap\n\n")

        if gap:
            _write_section6(f, gap)

        # ── 7. Intra-Family Confusion Ratio ────────────────────────────────
        f.write("\n## 7. Intra-Family Confusion Ratio\n\n")

        if intra:
            _write_section7(f, intra)

        # ── 8. Top Confused Attack Pairs ───────────────────────────────────
        f.write("\n## 8. Top Confused Attack Pairs\n\n")

        if intra:
            _write_section8(f, intra, sim_match)

        # ── 9. Baseline Comparison ─────────────────────────────────────────
        f.write("\n## 9. Baseline Comparison\n\n")
        f.write("(Requires running FedAvg/FedProx/FedProto with same eval logging.)\n")

        # ── 10. Conclusion ─────────────────────────────────────────────────
        f.write("\n## 10. Conclusion\n\n")
        _write_section10(f, topn_diag, intra, gap, affinity_raw, profiles)

        # ── Appendix ───────────────────────────────────────────────────────
        f.write("\n## Appendix: Experiment Commands\n\n")
        f.write("```bash\n")
        f.write("# Debug run (5 rounds)\n")
        f.write("uv run python system/main.py -c system/experiments/diagnosis_debug.json\n\n")
        f.write("# Full run (100 rounds)\n")
        f.write("uv run python system/main.py -c system/experiments/diagnosis_full.json\n\n")
        f.write("# Analysis\n")
        f.write("uv run python scripts/diagnosis_analysis.py \\\n")
        f.write("    --input-dir audit/fidsus_real_training_diagnosis \\\n")
        f.write("    --audit-dir docs/audit \\\n")
        f.write("    --family-mapping configs/attack_family_mapping.yaml\n\n")
        f.write("# Report\n")
        f.write("uv run python scripts/generate_affinity_report.py \\\n")
        f.write("    --input-dir audit/fidsus_real_training_diagnosis\n")
        f.write("```\n")

    print(f"Report written to {report_path}")


def _write_section3(f, topn_diag):
    """Top-N vs Random neighbors comparison."""
    # Aggregate across rounds
    metrics = ['mean_neighbor_entropy', 'mean_neighbor_normalized_entropy',
               'mean_neighbor_observed_class_count', 'mean_neighbor_js_to_global',
               'same_dominant_label_ratio', 'same_dominant_family_ratio',
               'neighbor_family_coverage', 'neighbor_class_coverage']
    rand_metrics = ['random_baseline_mean_entropy', 'random_baseline_class_coverage',
                    'random_baseline_js_to_global', 'random_baseline_same_dominant_label_ratio']

    real_avgs = {}
    rand_avgs = {}
    for m in metrics:
        vals = [float(r[m]) for r in topn_diag if r.get(m)]
        if vals:
            real_avgs[m] = (np.mean(vals), np.std(vals), len(vals))
    for m in rand_metrics:
        vals = [float(r[m]) for r in topn_diag if r.get(m)]
        if vals:
            rand_avgs[m] = (np.mean(vals), np.std(vals), len(vals))

    f.write("| Metric | FIDSUS Top-N | Random Baseline | Direction |\n")
    f.write("|--------|-------------|-----------------|----------|\n")

    comparisons = [
        ('mean_neighbor_entropy', 'random_baseline_mean_entropy', 'Entropy'),
        ('mean_neighbor_normalized_entropy', 'random_baseline_mean_entropy', 'Norm Entropy'),
        ('mean_neighbor_js_to_global', 'random_baseline_js_to_global', 'JS to Global'),
    ]
    for real_key, rand_key, label in comparisons:
        if real_key in real_avgs and rand_key in rand_avgs:
            rv = real_avgs[real_key][0]
            rv_std = real_avgs[real_key][1]
            rnd = rand_avgs[rand_key][0]
            rnd_std = rand_avgs[rand_key][1]
            direction = "higher" if rv > rnd else "lower"
            f.write(f"| {label} | {rv:.4f} ± {rv_std:.4f} | {rnd:.4f} ± {rnd_std:.4f} | {direction} |\n")

    if 'same_dominant_label_ratio' in real_avgs and 'random_baseline_same_dominant_label_ratio' in rand_avgs:
        rv = real_avgs['same_dominant_label_ratio'][0]
        rnd = rand_avgs['random_baseline_same_dominant_label_ratio'][0]
        direction = "higher" if rv > rnd else "lower"
        f.write(f"| Same Dominant Label Ratio | {rv:.4f} | {rnd:.4f} | {direction} |\n")

    if 'neighbor_class_coverage' in real_avgs and 'random_baseline_class_coverage' in rand_avgs:
        rv = real_avgs['neighbor_class_coverage'][0]
        rnd = rand_avgs['random_baseline_class_coverage'][0]
        direction = "higher" if rv > rnd else "lower"
        f.write(f"| Class Coverage | {rv:.4f} | {rnd:.4f} | {direction} |\n")

    if 'same_dominant_family_ratio' in real_avgs:
        f.write(f"\n| Same Dominant Family Ratio (Top-N) | {real_avgs['same_dominant_family_ratio'][0]:.4f} |\n")


def _write_section4(f, affinity_raw, profiles):
    """Affinity correlations."""
    # Build client profile lookup
    prof_map = {}
    for p in profiles:
        cid = int(p['client_id'])
        prof_map[cid] = p

    # Aggregate affinity data
    client_affinity_pairs = defaultdict(list)
    for row in affinity_raw:
        cid = int(row['client_id'])
        nid = int(row['neighbor_id'])
        new_aff = float(row.get('new_affinity', 0))
        client_affinity_pairs[(cid, nid)].append(new_aff)

    # Compute average affinity and look up profile similarity
    pairs = []
    for (cid, nid), affs in client_affinity_pairs.items():
        avg_aff = np.mean(affs)
        if cid in prof_map and nid in prof_map:
            cp = prof_map[cid]
            np_ = prof_map[nid]
            # Label distribution similarity
            try:
                cd = json.loads(cp.get('label_distribution_json', '{}'))
                nd = json.loads(np_.get('label_distribution_json', '{}'))
                all_c = sorted(set(list(cd.keys()) + list(nd.keys())), key=int)
                cvec = [float(cd.get(c, 0)) for c in all_c]
                nvec = [float(nd.get(c, 0)) for c in all_c]
                csum = sum(cvec)
                nsum = sum(nvec)
                if csum > 0 and nsum > 0:
                    cvec = [v/csum for v in cvec]
                    nvec = [v/nsum for v in nvec]
                    js = _js_div(cvec, nvec)
                    label_sim = max(0, 1 - math.sqrt(max(js, 0)))
                else:
                    label_sim = 0
            except:
                label_sim = 0

            same_dom = 1 if cp.get('dominant_label') == np_.get('dominant_label') else 0
            same_fam = 1 if cp.get('dominant_family') == np_.get('dominant_family') else 0
            pairs.append({
                'affinity': avg_aff,
                'label_sim': label_sim,
                'entropy_c': float(cp.get('label_entropy', 0)),
                'entropy_n': float(np_.get('label_entropy', 0)),
                'js_c': float(cp.get('js_to_global_distribution', 0)),
                'js_n': float(np_.get('js_to_global_distribution', 0)),
                'same_dom': same_dom,
                'same_fam': same_fam,
            })

    if not pairs:
        f.write("No affinity-pair data with profile overlap.\n")
        return

    f.write("Correlation of average affinity with:\n\n")
    f.write("| Variable | Pearson r |\n")
    f.write("|----------|----------|\n")

    for var in ['label_sim', 'same_dom', 'same_fam', 'entropy_c', 'entropy_n', 'js_c', 'js_n']:
        x = [p['affinity'] for p in pairs]
        y = [p[var] for p in pairs]
        if len(set(y)) < 2:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if np.isnan(r):
            r = 0
        f.write(f"| {var} | {r:.4f} |\n")


def _js_div(p, q):
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    kl1 = sum(pi * math.log(max(pi, 1e-12) / max(mi, 1e-12)) for pi, mi in zip(p, m))
    kl2 = sum(qi * math.log(max(qi, 1e-12) / max(mi, 1e-12)) for qi, mi in zip(q, m))
    return (kl1 + kl2) / 2


def _write_section5(f, fine_metrics, family_metrics):
    """Fine-grained vs family-level results."""
    last_fine = fine_metrics[-1]
    last_fam = family_metrics[-1] if family_metrics else {}

    f.write("### Final Round Results\n\n")
    f.write("| Metric | Fine-Grained | Family-Level |\n")
    f.write("|--------|-------------|-------------|\n")
    f.write(f"| Accuracy | {last_fine.get('accuracy', 'N/A')} | {last_fam.get('accuracy', 'N/A')} |\n")
    f.write(f"| Macro F1 | {last_fine.get('macro_f1', 'N/A')} | {last_fam.get('macro_f1', 'N/A')} |\n")
    f.write(f"| Weighted F1 | {last_fine.get('weighted_f1', 'N/A')} | {last_fam.get('weighted_f1', 'N/A')} |\n")


def _write_section6(f, gap):
    """Family-Fine gap."""
    last_gap = gap[-1]
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| family_fine_accuracy_gap | {last_gap.get('family_fine_accuracy_gap', 'N/A')} |\n")
    f.write(f"| family_fine_macroF1_gap | {last_gap.get('family_fine_macroF1_gap', 'N/A')} |\n")

    # Interpretation
    try:
        acc_gap = float(last_gap.get('family_fine_accuracy_gap', 0))
        f1_gap = float(last_gap.get('family_fine_macroF1_gap', 0))
        if acc_gap > 0.01 and f1_gap > 0.01:
            f.write("\nFamily-level metrics are noticeably higher than fine-grained metrics. "
                    "This suggests that the model performs better at attack family recognition "
                    "than at distinguishing between similar attacks within the same family.\n")
        elif acc_gap < -0.01:
            f.write("\nFamily-level metrics are lower than fine-grained (unexpected). "
                    "This may indicate mapping issues or label class numbering mismatch.\n")
        else:
            f.write("\nThe gap between family-level and fine-grained metrics is small.\n")
    except:
        pass


def _write_section7(f, intra):
    """Intra-family confusion ratio."""
    total_confusion = sum(int(r['confusion_count']) for r in intra)
    intra_confusion = sum(int(r['confusion_count']) for r in intra if r['is_intra_family'] == '1')

    intra_ratio = intra_confusion / max(total_confusion, 1)

    f.write(f"- Total confusion events: {total_confusion}\n")
    f.write(f"- Intra-family confusion events: {intra_confusion}\n")
    f.write(f"- Intra-family confusion ratio: {intra_ratio:.4f}\n\n")

    if intra_ratio > 0.5:
        f.write("The majority of errors are intra-family, suggesting the model struggles "
                "to distinguish attacks within the same family.\n")
    elif intra_ratio > 0.3:
        f.write("A significant portion of errors are intra-family.\n")
    else:
        f.write("Intra-family confusion is present but not dominant.\n")


def _write_section8(f, intra, sim_match):
    """Top confused attack pairs."""
    sorted_intra = sorted(intra, key=lambda r: -int(r['confusion_count']))

    f.write("| Rank | True → Pred | Intra-Family? | Confusion Count | Rate |\n")
    f.write("|------|------------|---------------|-----------------|------|\n")
    for r in sorted_intra[:15]:
        tf = f"{r['true_label']} → {r['pred_label']}"
        is_intra = "Yes" if r['is_intra_family'] == '1' else "No"
        f.write(f"| {r['pair_rank']} | {tf} | {is_intra} | {r['confusion_count']} | {r['confusion_rate']} |\n")

    # Check specific pairs mentioned in research hypothesis
    f.write("\n### Key Attack Pairs (from research hypothesis)\n\n")
    key_pairs = [
        # NSL-KDD
        ("3", "4", "U2R ↔ R2L (NSL-KDD, both Privilege/Access)"),
        ("4", "3", "R2L ↔ U2R (NSL-KDD, both Privilege/Access)"),
        # UNSW
        ("1", "2", "Backdoor ↔ Analysis (UNSW, both Backdoor/Analysis)"),
        ("2", "1", "Analysis ↔ Backdoor (UNSW, both Backdoor/Analysis)"),
        ("1", "7", "Backdoor ↔ DoS (UNSW)"),
        ("4", "8", "Shellcode ↔ Worms (UNSW, both Shellcode/Worms)"),
        ("8", "4", "Worms ↔ Shellcode (UNSW, both Shellcode/Worms)"),
    ]
    for pair in key_pairs:
        found = None
        for r in sorted_intra:
            if r['true_label'] == pair[0] and r['pred_label'] == pair[1]:
                found = r
                break
        if found:
            f.write(f"- **{pair[2]}**: Count={found['confusion_count']}, "
                    f"intra_family={found['is_intra_family']}\n")
        else:
            f.write(f"- **{pair[2]}**: Not found in top confusion pairs (may have zero confusion)\n")


def _write_section10(f, topn_diag, intra, gap, affinity_raw, profiles):
    """Conclusion: does the evidence support the hypothesis?"""
    f.write("### Hypothesis\n\n")
    f.write("> FIDSUS's affinity/top-n mechanism mainly captures client similarity "
            "without explicit consideration of class balance and global representativeness. "
            "This mechanism may enhance attack-family-level recognition but is insufficient "
            "for solving intra-family fine-grained attack differentiation.\n\n")

    f.write("### Evidence Assessment\n\n")

    # Check evidence from top-n analysis
    evidence_support = 0
    evidence_against = 0
    findings = []

    # Check 1: Top-n label similarity vs random
    if topn_diag:
        real_sim = None
        rand_sim = None
        if 'same_dominant_label_ratio' in topn_diag[0]:
            real_vals = [float(r['same_dominant_label_ratio']) for r in topn_diag if r.get('same_dominant_label_ratio')]
            rand_vals = [float(r['random_baseline_same_dominant_label_ratio']) for r in topn_diag if r.get('random_baseline_same_dominant_label_ratio')]
            if real_vals and rand_vals:
                real_sim = np.mean(real_vals)
                rand_sim = np.mean(rand_vals)
                if real_sim > rand_sim * 1.05:
                    findings.append(f"[SUPPORT] Top-n neighbors have higher same-dominant-label ratio than random ({real_sim:.4f} vs {rand_sim:.4f})")
                    evidence_support += 1
                else:
                    findings.append(f"[AGAINST] Top-n same-dominant-label ratio not higher than random ({real_sim:.4f} vs {rand_sim:.4f})")
                    evidence_against += 1

        if 'mean_neighbor_entropy' in topn_diag[0] and 'random_baseline_mean_entropy' in topn_diag[0]:
            real_ent = np.mean([float(r['mean_neighbor_entropy']) for r in topn_diag if r.get('mean_neighbor_entropy')])
            rand_ent = np.mean([float(r['random_baseline_mean_entropy']) for r in topn_diag if r.get('random_baseline_mean_entropy')])
            if real_ent < rand_ent * 0.98:
                findings.append(f"[SUPPORT] Top-n neighbors have lower entropy than random ({real_ent:.4f} vs {rand_ent:.4f}), suggesting less diverse selection")
                evidence_support += 1
            else:
                findings.append(f"[AGAINST] Top-n entropy not lower than random ({real_ent:.4f} vs {rand_ent:.4f})")
                evidence_against += 1

    # Check 2: Family-fine gap
    if gap:
        last_gap = gap[-1]
        try:
            acc_gap = float(last_gap.get('family_fine_accuracy_gap', 0))
            if acc_gap > 0.02:
                findings.append(f"[SUPPORT] Family accuracy {acc_gap:.4f} higher than fine-grained, confirming family-level advantage")
                evidence_support += 1
            else:
                findings.append(f"[AGAINST] Family-fine accuracy gap is small ({acc_gap:.4f})")
                evidence_against += 1
        except:
            pass

    # Check 3: Intra-family confusion
    if intra:
        total = sum(int(r['confusion_count']) for r in intra)
        intra_c = sum(int(r['confusion_count']) for r in intra if r['is_intra_family'] == '1')
        ratio = intra_c / max(total, 1)
        if ratio > 0.3:
            findings.append(f"[SUPPORT] Intra-family confusion ratio is {ratio:.2%}, indicating family-internal confusion")
            evidence_support += 1
        else:
            findings.append(f"[AGAINST] Intra-family confusion ratio is low ({ratio:.2%})")
            evidence_against += 1

    for finding in findings:
        f.write(f"- {finding}\n")

    f.write(f"\n### Verdict\n\n")
    f.write(f"Supporting evidence: {evidence_support} | Contrary evidence: {evidence_against}\n\n")

    if evidence_support > evidence_against:
        f.write("**The evidence SUPPORTS the hypothesis.** FIDSUS's affinity/top-n mechanism "
                "demonstrates a tendency toward similarity-based client selection without "
                "explicit class-balance awareness, and the resulting model shows stronger "
                "family-level than fine-grained performance.\n")
    elif evidence_support == evidence_against and evidence_support > 0:
        f.write("**The evidence is MIXED.** Some metrics support the hypothesis while others "
                "do not. More data or different hyperparameters may clarify the picture.\n")
    elif evidence_against > evidence_support:
        f.write("**The evidence DOES NOT SUPPORT the hypothesis in this run.** "
                "FIDSUS's affinity mechanism may not behave as predicted in real training, "
                "or the effect may require different experimental conditions to manifest.\n")
    else:
        f.write("**INCONCLUSIVE** - Insufficient data to assess the hypothesis. "
                "Run full experiments and verify all logs are generated correctly.\n")


if __name__ == '__main__':
    main()
