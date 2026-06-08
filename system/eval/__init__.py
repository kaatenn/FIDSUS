"""Family-level evaluation module for FIDSUS."""

from eval.family_eval import (
    DATASET_CONFIGS,
    EvalResult,
    FullEvalReport,
    IntraFamilyConfusion,
    NSLKDD_FAMILY_MAP,
    NSLKDD_LABEL_MAP,
    UAV_NIDD_FAMILY_MAP,
    UAV_NIDD_LABEL_MAP,
    UNSW_FAMILY_MAP,
    UNSW_LABEL_MAP,
    compute_eval_result,
    compute_intra_family_analysis,
    generate_summary_json,
    generate_summary_text,
    label_id_to_name,
    label_name_to_family,
    run_full_evaluation,
    save_report,
)
