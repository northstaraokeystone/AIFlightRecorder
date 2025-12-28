"""Training Data Exporter (v2.1)

Exports training examples in various formats for fine-tuning.
"""

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

from .extractor import TrainingExample
from ..core import emit_receipt, dual_hash


class ExportFormat(Enum):
    """Supported export formats."""
    JSONL = "jsonl"
    PARQUET = "parquet"
    HF_DATASET = "hf_dataset"
    CSV = "csv"


def export_jsonl(examples: List[TrainingExample],
                 output_path: str,
                 include_features: bool = True) -> dict:
    """Export training examples as JSONL.

    Args:
        examples: Training examples to export
        output_path: Output file path
        include_features: Whether to include extracted features

    Returns:
        Export result dict
    """
    exported = 0
    total_bytes = 0

    with open(output_path, 'w') as f:
        for example in examples:
            record = {
                "example_id": example.example_id,
                "decision_id": example.decision_id,
                "original": example.original_decision,
                "corrected": example.corrected_decision,
                "reason_code": example.reason_code,
                "label": example.label,
                "quality_score": example.quality_score,
                "created_at": example.created_at
            }

            if include_features:
                record["features"] = example.features

            if example.context:
                record["context"] = example.context

            line = json.dumps(record, sort_keys=True) + "\n"
            f.write(line)
            exported += 1
            total_bytes += len(line.encode('utf-8'))

    # Emit receipt
    receipt = emit_export_receipt(
        format_type="jsonl",
        output_path=output_path,
        example_count=exported,
        total_bytes=total_bytes
    )

    return {
        "format": "jsonl",
        "output_path": output_path,
        "examples_exported": exported,
        "total_bytes": total_bytes,
        "receipt": receipt
    }


def export_parquet(examples: List[TrainingExample],
                   output_path: str,
                   include_features: bool = True) -> dict:
    """Export training examples as Parquet.

    Requires pyarrow to be installed.

    Args:
        examples: Training examples to export
        output_path: Output file path
        include_features: Whether to include extracted features

    Returns:
        Export result dict
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        # Fall back to JSONL if pyarrow not available
        jsonl_path = output_path.replace('.parquet', '.jsonl')
        return export_jsonl(examples, jsonl_path, include_features)

    # Build table
    records = []
    for example in examples:
        record = {
            "example_id": example.example_id,
            "decision_id": example.decision_id,
            "original": json.dumps(example.original_decision),
            "corrected": json.dumps(example.corrected_decision),
            "reason_code": example.reason_code,
            "label": example.label,
            "quality_score": example.quality_score,
            "created_at": example.created_at
        }

        if include_features:
            record["features"] = json.dumps(example.features)

        records.append(record)

    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)

    # Get file size
    total_bytes = os.path.getsize(output_path)

    receipt = emit_export_receipt(
        format_type="parquet",
        output_path=output_path,
        example_count=len(examples),
        total_bytes=total_bytes
    )

    return {
        "format": "parquet",
        "output_path": output_path,
        "examples_exported": len(examples),
        "total_bytes": total_bytes,
        "receipt": receipt
    }


def export_hf_dataset(examples: List[TrainingExample],
                      output_dir: str,
                      dataset_name: str = "flight_recorder_corrections") -> dict:
    """Export as Hugging Face dataset format.

    Args:
        examples: Training examples
        output_dir: Output directory
        dataset_name: Dataset name

    Returns:
        Export result dict
    """
    try:
        from datasets import Dataset, Features, Value, ClassLabel
    except ImportError:
        # Fall back to JSONL if datasets not available
        jsonl_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
        return export_jsonl(examples, jsonl_path)

    # Build records
    records = {
        "example_id": [],
        "decision_id": [],
        "original_action": [],
        "corrected_action": [],
        "original_confidence": [],
        "corrected_confidence": [],
        "reason_code": [],
        "label": [],
        "quality_score": []
    }

    for example in examples:
        records["example_id"].append(example.example_id)
        records["decision_id"].append(example.decision_id)

        orig_action = example.original_decision.get("action", {}).get("type", "")
        corr_action = example.corrected_decision.get("action", {}).get("type", "")
        records["original_action"].append(orig_action)
        records["corrected_action"].append(corr_action)

        records["original_confidence"].append(
            example.original_decision.get("confidence", 0.5)
        )
        records["corrected_confidence"].append(
            example.corrected_decision.get("confidence", 0.5)
        )

        records["reason_code"].append(example.reason_code)
        records["label"].append(example.label)
        records["quality_score"].append(example.quality_score)

    dataset = Dataset.from_dict(records)

    # Save
    output_path = os.path.join(output_dir, dataset_name)
    dataset.save_to_disk(output_path)

    receipt = emit_export_receipt(
        format_type="hf_dataset",
        output_path=output_path,
        example_count=len(examples),
        total_bytes=0  # Unknown for HF datasets
    )

    return {
        "format": "hf_dataset",
        "output_path": output_path,
        "examples_exported": len(examples),
        "receipt": receipt
    }


def export_csv(examples: List[TrainingExample],
               output_path: str) -> dict:
    """Export training examples as CSV.

    Args:
        examples: Training examples
        output_path: Output path

    Returns:
        Export result dict
    """
    import csv

    fieldnames = [
        "example_id", "decision_id", "reason_code", "label",
        "quality_score", "original_action", "corrected_action",
        "original_confidence", "corrected_confidence", "created_at"
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for example in examples:
            writer.writerow({
                "example_id": example.example_id,
                "decision_id": example.decision_id,
                "reason_code": example.reason_code,
                "label": example.label,
                "quality_score": example.quality_score,
                "original_action": example.original_decision.get("action", {}).get("type", ""),
                "corrected_action": example.corrected_decision.get("action", {}).get("type", ""),
                "original_confidence": example.original_decision.get("confidence", 0.5),
                "corrected_confidence": example.corrected_decision.get("confidence", 0.5),
                "created_at": example.created_at
            })

    total_bytes = os.path.getsize(output_path)

    receipt = emit_export_receipt(
        format_type="csv",
        output_path=output_path,
        example_count=len(examples),
        total_bytes=total_bytes
    )

    return {
        "format": "csv",
        "output_path": output_path,
        "examples_exported": len(examples),
        "total_bytes": total_bytes,
        "receipt": receipt
    }


def emit_export_receipt(format_type: str, output_path: str,
                        example_count: int, total_bytes: int) -> dict:
    """Emit training data export receipt.

    Args:
        format_type: Export format
        output_path: Output path
        example_count: Number of examples
        total_bytes: Total bytes written

    Returns:
        Receipt dict
    """
    return emit_receipt("training_export", {
        "format": format_type,
        "output_path": output_path,
        "example_count": example_count,
        "total_bytes": total_bytes,
        "exported_at": datetime.now(timezone.utc).isoformat()
    }, silent=True)


def auto_export(examples: List[TrainingExample],
                output_dir: str,
                formats: Optional[List[ExportFormat]] = None) -> List[dict]:
    """Export to multiple formats automatically.

    Args:
        examples: Training examples
        output_dir: Output directory
        formats: Formats to export (None for all)

    Returns:
        List of export results
    """
    if formats is None:
        formats = [ExportFormat.JSONL]  # Default to JSONL

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for fmt in formats:
        if fmt == ExportFormat.JSONL:
            path = os.path.join(output_dir, f"training_{timestamp}.jsonl")
            results.append(export_jsonl(examples, path))

        elif fmt == ExportFormat.PARQUET:
            path = os.path.join(output_dir, f"training_{timestamp}.parquet")
            results.append(export_parquet(examples, path))

        elif fmt == ExportFormat.HF_DATASET:
            results.append(export_hf_dataset(examples, output_dir))

        elif fmt == ExportFormat.CSV:
            path = os.path.join(output_dir, f"training_{timestamp}.csv")
            results.append(export_csv(examples, path))

    return results
