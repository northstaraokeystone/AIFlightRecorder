"""Tests for training data production (v2.1)."""

import os
import pytest
import tempfile
import time
from pathlib import Path

from src.training import (
    extract_training_example,
    label_example,
    score_example,
    export_jsonl,
    queue_for_finetuning,
    get_pending_finetuning
)
from src.training.extractor import TrainingExample, TrainingExampleBuffer
from src.training.exporter import export_parquet, auto_export, export_csv
from src.training.feedback_loop import FinetuneJob, FeedbackLoop, emit_finetune_receipt


class TestTrainingExtractor:
    """Tests for training example extraction."""

    def test_extract_from_intervention(self):
        """Should extract example from intervention."""
        intervention = {
            "decision_id": "d1",
            "original_decision": {"action": {"type": "CONTINUE"}, "confidence": 0.6},
            "correction": {"action": {"type": "AVOID"}, "confidence": 0.95},
            "reason_code": "MODEL_ERROR"
        }

        example = extract_training_example(intervention)

        assert example is not None
        assert example.decision_id == "d1"
        assert example.original_output is not None
        assert example.corrected_output is not None

    def test_extract_preserves_context(self):
        """Extraction should preserve context."""
        intervention = {
            "decision_id": "d1",
            "original_decision": {"action": {"type": "CONTINUE"}},
            "correction": {"action": {"type": "AVOID"}},
            "context": {"altitude": 1000, "speed": 50}
        }

        example = extract_training_example(intervention)

        assert example.context == intervention["context"]

    def test_label_example(self):
        """Should label example correctly."""
        example = TrainingExample(
            example_id="ex1",
            decision_id="d1",
            input_context={"altitude": 1000},
            original_output={"action": {"type": "CONTINUE"}},
            corrected_output={"action": {"type": "AVOID"}},
            context={}
        )

        labeled = label_example(example, "high_confidence_correction")

        assert labeled.label == "high_confidence_correction"

    def test_score_example(self):
        """Should score example for training priority."""
        example = TrainingExample(
            example_id="ex1",
            decision_id="d1",
            input_context={},
            original_output={"confidence": 0.8},
            corrected_output={"confidence": 0.95},
            context={}
        )

        scored = score_example(example)

        assert scored.score is not None
        assert 0.0 <= scored.score <= 1.0


class TestTrainingExampleBuffer:
    """Tests for example buffer."""

    def test_buffer_add(self):
        """Buffer should store examples."""
        buffer = TrainingExampleBuffer(max_size=100)
        example = TrainingExample(
            example_id="ex1",
            decision_id="d1",
            input_context={},
            original_output={},
            corrected_output={},
            context={}
        )

        buffer.add(example)

        assert len(buffer) == 1

    def test_buffer_max_size(self):
        """Buffer should respect max size."""
        buffer = TrainingExampleBuffer(max_size=5)

        for i in range(10):
            example = TrainingExample(
                example_id=f"ex{i}",
                decision_id=f"d{i}",
                input_context={},
                original_output={},
                corrected_output={},
                context={}
            )
            buffer.add(example)

        assert len(buffer) <= 5

    def test_buffer_get_batch(self):
        """Buffer should return batches."""
        buffer = TrainingExampleBuffer(max_size=100)

        for i in range(20):
            example = TrainingExample(
                example_id=f"ex{i}",
                decision_id=f"d{i}",
                input_context={},
                original_output={},
                corrected_output={},
                context={}
            )
            buffer.add(example)

        batch = buffer.get_batch(size=10)
        assert len(batch) == 10


class TestExporter:
    """Tests for training data export."""

    def test_export_jsonl(self, tmp_path):
        """Should export to JSONL format."""
        examples = [
            TrainingExample(
                example_id="ex1",
                decision_id="d1",
                input_context={"test": True},
                original_output={"action": "continue"},
                corrected_output={"action": "avoid"},
                context={}
            )
        ]

        output_path = tmp_path / "output.jsonl"
        result = export_jsonl(examples, str(output_path))

        assert result["success"] is True
        assert output_path.exists()

        # Verify JSONL content
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 1

    def test_export_csv(self, tmp_path):
        """Should export to CSV format."""
        examples = [
            TrainingExample(
                example_id="ex1",
                decision_id="d1",
                input_context={"test": True},
                original_output={"action": "continue"},
                corrected_output={"action": "avoid"},
                context={}
            )
        ]

        output_path = tmp_path / "output.csv"
        result = export_csv(examples, str(output_path))

        assert result["success"] is True
        assert output_path.exists()

    def test_auto_export_selects_format(self, tmp_path):
        """auto_export should select format based on extension."""
        examples = [
            TrainingExample(
                example_id="ex1",
                decision_id="d1",
                input_context={},
                original_output={},
                corrected_output={},
                context={}
            )
        ]

        jsonl_path = tmp_path / "output.jsonl"
        result = auto_export(examples, str(jsonl_path))
        assert result["format"] == "jsonl"


class TestFeedbackLoop:
    """Tests for feedback loop queue."""

    def test_queue_for_finetuning(self):
        """Should queue job for finetuning."""
        examples = [
            TrainingExample(
                example_id="ex1",
                decision_id="d1",
                input_context={},
                original_output={},
                corrected_output={},
                context={}
            )
        ]

        job_id = queue_for_finetuning(examples, priority="high")

        assert job_id is not None

    def test_get_pending_finetuning(self):
        """Should get pending jobs."""
        # Queue some jobs
        for i in range(3):
            examples = [
                TrainingExample(
                    example_id=f"ex{i}",
                    decision_id=f"d{i}",
                    input_context={},
                    original_output={},
                    corrected_output={},
                    context={}
                )
            ]
            queue_for_finetuning(examples, priority="normal")

        pending = get_pending_finetuning()

        assert isinstance(pending, list)
        # May have jobs from previous tests too

    def test_feedback_loop_class(self):
        """FeedbackLoop class should work."""
        loop = FeedbackLoop()

        examples = [
            TrainingExample(
                example_id="ex1",
                decision_id="d1",
                input_context={},
                original_output={},
                corrected_output={},
                context={}
            )
        ]

        job_id = loop.queue_job(examples, priority="high")
        assert job_id is not None

        pending = loop.get_pending()
        assert isinstance(pending, list)


class TestFinetuneJob:
    """Tests for FinetuneJob dataclass."""

    def test_job_creation(self):
        """Job should be created with defaults."""
        job = FinetuneJob(
            job_id="j1",
            examples=[],
            priority="normal",
            created_at="2024-01-01T00:00:00Z"
        )

        assert job.status == "pending"
        assert job.started_at is None

    def test_job_priority(self):
        """Job should have priority."""
        high_job = FinetuneJob(
            job_id="j1",
            examples=[],
            priority="high",
            created_at="2024-01-01T00:00:00Z"
        )
        low_job = FinetuneJob(
            job_id="j2",
            examples=[],
            priority="low",
            created_at="2024-01-01T00:00:00Z"
        )

        assert high_job.priority == "high"
        assert low_job.priority == "low"


class TestFinetuneReceipt:
    """Tests for finetune receipt emission."""

    def test_emit_finetune_receipt(self):
        """Finetune receipt should have required fields."""
        job = FinetuneJob(
            job_id="j1",
            examples=[],
            priority="high",
            created_at="2024-01-01T00:00:00Z",
            status="pending"
        )

        receipt = emit_finetune_receipt(job)

        assert receipt["receipt_type"] == "finetune"
        assert receipt["job_id"] == "j1"
        assert receipt["priority"] == "high"


class TestPerformance:
    """Performance tests for training modules."""

    def test_extraction_latency(self):
        """Extraction should be fast."""
        intervention = {
            "decision_id": "d1",
            "original_decision": {"action": {"type": "CONTINUE"}},
            "correction": {"action": {"type": "AVOID"}},
            "reason_code": "MODEL_ERROR"
        }

        start = time.perf_counter()
        for _ in range(100):
            extract_training_example(intervention)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 5, f"Extraction latency {elapsed_ms}ms exceeds 5ms"

    def test_export_latency(self, tmp_path):
        """Export should be reasonably fast."""
        examples = [
            TrainingExample(
                example_id=f"ex{i}",
                decision_id=f"d{i}",
                input_context={"data": f"test{i}"},
                original_output={"action": "continue"},
                corrected_output={"action": "avoid"},
                context={}
            )
            for i in range(100)
        ]

        output_path = tmp_path / "output.jsonl"

        start = time.perf_counter()
        export_jsonl(examples, str(output_path))
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Export latency {elapsed_ms}ms exceeds 100ms for 100 examples"
