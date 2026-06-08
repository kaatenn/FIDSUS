"""
FIDSUS Top-k Audit Instrumentation Package.

Provides logging, metrics, and chart generation for auditing the
Top-k similar-client selection mechanism in FIDSUS.

All instrumentation is designed to be minimally invasive:
- Subclasses extend (not modify) original server/client classes.
- Ablation modes (Random-k, Entropy-aware Top-k, HiCS-style Top-k)
  are implemented as separate configuration options.
"""

from .instrument import (
    AuditLogger,
    compute_client_label_stats,
    compute_label_entropy,
    compute_js_divergence,
    classify_client_type,
)

from .audited_server import AuditedFIDSUS

__all__ = [
    "AuditLogger",
    "compute_client_label_stats",
    "compute_label_entropy",
    "compute_js_divergence",
    "classify_client_type",
    "AuditedFIDSUS",
]
