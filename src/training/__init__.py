"""Training Data Production Module (v2.1)

Extracts training examples from corrections, exports in various formats,
and integrates with fine-tuning pipelines.
"""

from .extractor import (
    TrainingExample,
    extract_training_example,
    label_example,
    score_example,
    get_training_queue
)

from .exporter import (
    export_jsonl,
    export_parquet,
    export_hf_dataset,
    ExportFormat
)

from .feedback_loop import (
    FeedbackLoop,
    queue_for_finetuning,
    get_pending_finetuning,
    emit_finetune_receipt
)

__all__ = [
    "TrainingExample",
    "extract_training_example",
    "label_example",
    "score_example",
    "get_training_queue",
    "export_jsonl",
    "export_parquet",
    "export_hf_dataset",
    "ExportFormat",
    "FeedbackLoop",
    "queue_for_finetuning",
    "get_pending_finetuning",
    "emit_finetune_receipt"
]
