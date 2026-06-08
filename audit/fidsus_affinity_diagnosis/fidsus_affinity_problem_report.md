# FIDSUS Affinity Matrix Diagnosis Report

## 1. Code Review Conclusions

### Key Files

| File | Line | Role |
|------|------|------|
| `system/flcore/servers/FIDSUS.py:17` | `__init__` | `self.P = torch.diag(torch.ones(self.num_clients))` — affinity matrix init |
| `system/flcore/servers/FIDSUS.py:59` | `send_models` | `torch.topk(self.P[client.id], M_).indices` — top-n selection |
| `system/flcore/servers/FIDSUS.py:97` | `receive_models` | `self.P[client.id] += client.weight_vector` — affinity update |
| `system/flcore/clients/clientFIDSUS.py:99-110` | `weight_cal` | `w = (L_old - L_received) / (||param_diff|| + 1e-5)` — weight computation |
| `system/flcore/clients/clientFIDSUS.py:134-141` | `aggregate_parameters` | Weighted sum of received feature extractors |
| `system/flcore/clients/clientFIDSUS.py:269-282` | `aggregation` | MMD-weighted prototype aggregation |

### Signals Actually Used

- ✅ Validation loss improvement (L_old − L_received)
- ✅ Parameter distance (||param_n − param_i||)
- ✅ Prototype MMD distance (for prototype aggregation)
- ✅ Cumulative affinity weights (P accumulates over rounds)

### Signals NOT Used

- ❌ Client label distribution p_i^c
- ❌ Client label entropy H_i
- ❌ Class balance / imbalance ratio
- ❌ Minority class coverage
- ❌ Attack family coverage
- ❌ Per-class recall / per-family recall
- ❌ Global label distribution representativeness
- ❌ JS/KL divergence to global distribution

### Verdict

**The code explicitly confirms: FIDSUS affinity/top-n selection does NOT use any
class-balance, entropy, or family-coverage information.** It is purely a
model-similarity mechanism based on validation loss, parameter distance, and
prototype MMD.

## 2. Affinity/Top-N Mechanism: Actual Signals

```
weight_i = (L_old − L_received_i) / (||param_old − param_i|| + 1e-5)
self.P[client_k][client_i] += weight_i
top_n_clients = argtopk(self.P[client_k], M)
```

The weight is high when:
1. Client i's model performs well on client k's validation data
2. Client i's parameters are close to client k's old parameters

This is model-similarity-based, NOT class-balance-based.

## 3. Explicit Consideration of Class Balance & Attack-Family Coverage?

**NO.** The following are absent from the codebase:
- No reference to `label_distribution` in FIDSUS-related code
- No `entropy`, `balance`, or `representativeness` computation
- No `family` or `group` concept in training loop
- No minority-class-aware selection

## 4. Dataset Label Distribution & Attack Similarity Summary

### NSL-KDD (5 classes, ~148K samples)

| Class | Train | Test | Total | % |
|-------|-------|------|-------|---|
| DoS | 45,927 | 7,460 | 53,387 | 35.9% |
| Normal | 67,343 | 9,711 | 77,054 | 51.9% |
| Probe | 11,656 | 2,421 | 14,077 | 9.5% |
| R2L | 995 | 2,885 | 3,880 | 2.6% |
| U2R | 52 | 67 | 119 | 0.1% |

**Attack similarity (Pearson r):**
- R2L–U2R: 0.612 (highest)
- DoS–Probe: 0.490

### UNSW-NB15 (10 classes, ~257K samples)

| Class | % | Class | % |
|-------|---|-------|---|
| Normal | 36.1% | Generic | 22.8% |
| Exploits | 17.3% | Fuzzers | 9.4% |
| DoS | 6.3% | Reconnaissance | 5.4% |
| Analysis | 1.0% | Backdoor | 0.9% |
| Shellcode | 0.6% | Worms | 0.1% |

**High-similarity pairs:** Recon–Shellcode (0.960), Analysis–Backdoor (0.945), DoS–Backdoor (0.913)


## 5. Affinity vs Label Similarity / Entropy / Family Similarity Correlations

| Metric | Avg Pearson r | Interpretation |
|--------|--------------|----------------|
| Affinity vs Label Distribution Similarity | +1.000 | **Strong positive** — FIDSUS selects clients with similar label distributions |
| Affinity vs Client Entropy | +0.100 | **Near zero** — FIDSUS does NOT prefer high-entropy (balanced) clients |
| Affinity vs JS to Global Distribution | -0.239 | **Negative/weak** — FIDSUS does NOT prefer globally representative clients |
| Affinity vs Same Dominant Class | +0.797 | **Strong positive** — FIDSUS clusters clients with the same dominant class |

**Interpretation**: The affinity matrix is strongly correlated with label distribution
similarity and same-class preference, but has essentially no correlation with client
label entropy or global representativeness. This confirms that:

1. **FIDSUS captures "similarity" effectively** — clients with similar label mixes get
   high affinity scores.
2. **FIDSUS does NOT capture "balance" or "diversity"** — there is no preference for
   clients with diverse class coverage or balanced distributions.
3. **This is the exact problem described in the hypothesis**: "similarity-based selection
   ≠ class-balanced selection."


### NSLKDD
- Avg client entropy: 0.5335
- Avg client observed classes: 3.7
- Avg client JS to global: 0.5246

### UNSW
- Avg client entropy: 0.9347
- Avg client observed classes: 6.8
- Avg client JS to global: 0.5813


## 6. Family-Level vs Fine-Grained Evaluation

See `family_fine_gap.csv` and per-dataset/per-method output directories for
detailed metrics including fine-grained confusion matrices, family-level
confusion matrices, and per-class recall values.

## 7. Intra-Family Confusion Analysis

See `intra_family_confusion_pairs.csv` in each per-method directory for
the top confused pairs within each attack family. See
`confusion_pair_similarity_match.csv` for whether each confused pair is
intra-family.

## 8. FIDSUS vs Baseline Comparison

See `method_comparison_family_fine_gap.csv` and
`method_comparison_intra_family_confusion.csv` for:
- Family-fine accuracy/macro-F1 gaps per method
- Intra-family confusion ratios per method

## 9. Overall Judgment

Based on code review AND experiments A/B:

**Verdict: HYPOTHESIS SUPPORTED**

The evidence is strong:

1. **Code review**: FIDSUS does not use label entropy, class balance, family coverage,
   or global representativeness. Confirmed by reading all FIDSUS source files.

2. **Experiment A**: FIDSUS top-n selection:
   - Has significantly LOWER mean entropy than random-n selection
   - Has LOWER class coverage than random-n selection
   - Has MUCH HIGHER same-dominant-class ratio (0.87-0.92 vs 0.20-0.30)
   - Shows that FIDSUS preferentially selects clients with similar label distributions

3. **Experiment B**: Correlations confirm:
   - Affinity ↔ Label Similarity: strong positive (r≈1.0)
   - Affinity ↔ Client Entropy: near zero (r≈0.08-0.12)
   - Affinity ↔ JS to Global: negative (r≈−0.24)
   - Affinity ↔ Same Dominant Class: strong positive (r≈0.76-0.83)

4. **Experiment C**: The controlled simulation shows that FIDSUS has a positive
   family-fine gap (family accuracy > fine-grained accuracy), indicating that
   the mechanism does learn attack-family-level patterns. However, this comes at
   the cost of fine-grained discrimination within families.

5. **Experiment D**: Intra-family confusion ratio is higher in FIDSUS compared to
   FedAvg/FedProx, confirming that FIDSUS's similarity clustering tends to group
   similar attack types together — which helps family recognition but hurts
   fine-grained separation.

**Final conclusion: FIDSUS's affinity matrix does implement attack-family-level
client clustering, but this mechanism does not address class balance, minority
class coverage, or fine-grained discrimination within attack families.**
This is exactly the problem that HiCS-FL's motivation identifies: "similarity
selection is not equal to class-balanced selection."

## 10. Next Steps & Recommendations

1. **Heterogeneity-aware affinity**: Add label distribution JS divergence as a
   regularization term in affinity computation.
2. **Entropy-aware top-n selection**: Require at least K neighbors to have
   above-median label entropy.
3. **Attack-family-aware sampling**: Ensure each round's selected group covers
   all attack families (or at least configurable minimum).
4. **Contrastive/prototype learning**: For intra-family separation — use
   contrastive loss within families to separate similar attacks.
5. **Hierarchical IDS evaluation**: Always report both family-level and
   fine-grained metrics when evaluating on non-IID IDS data.

---
Generated: 2026-06-08T05:46:56
