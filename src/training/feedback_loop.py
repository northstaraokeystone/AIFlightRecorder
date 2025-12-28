"""Feedback Loop - Fine-tuning Integration (v2.1)

Manages the feedback loop from interventions to model fine-tuning.
Note: Does not execute fine-tuning (queue only in v2.2).
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .extractor import TrainingExample, get_training_queue, filter_training_queue
from .exporter import export_jsonl, ExportFormat
from ..core import emit_receipt


@dataclass
class FinetuneJob:
    """A fine-tuning job in the queue."""
    job_id: str
    created_at: str
    example_count: int
    example_ids: List[str]
    status: str  # "queued", "approved", "declined", "submitted"
    export_path: Optional[str] = None
    submitted_at: Optional[str] = None
    notes: Optional[str] = None


# Fine-tuning queue
_finetune_queue: List[FinetuneJob] = []


class FeedbackLoop:
    """Manages the intervention -> training -> fine-tuning loop.

    Workflow:
    1. Interventions create training examples
    2. Examples are scored and labeled
    3. High-quality examples are queued for fine-tuning
    4. Humans approve fine-tuning batches
    5. Approved batches are submitted (external process)
    """

    def __init__(self, min_quality: float = 0.6,
                 batch_size: int = 100):
        """Initialize feedback loop.

        Args:
            min_quality: Minimum quality score for fine-tuning
            batch_size: Batch size for fine-tuning jobs
        """
        self._min_quality = min_quality
        self._batch_size = batch_size
        self._jobs: List[FinetuneJob] = []

    def create_job(self, examples: Optional[List[TrainingExample]] = None,
                   notes: Optional[str] = None) -> FinetuneJob:
        """Create a fine-tuning job from examples.

        Args:
            examples: Examples to include (None = auto from queue)
            notes: Optional job notes

        Returns:
            Created FinetuneJob
        """
        if examples is None:
            examples = filter_training_queue(
                min_quality=self._min_quality,
                labels=["correct"]
            )

        if not examples:
            raise ValueError("No qualifying examples for fine-tuning")

        job = FinetuneJob(
            job_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            example_count=len(examples),
            example_ids=[e.example_id for e in examples],
            status="queued",
            notes=notes
        )

        self._jobs.append(job)
        _finetune_queue.append(job)

        # Emit receipt
        emit_finetune_receipt(job, "created")

        return job

    def approve_job(self, job_id: str, approver_id: str = "unknown") -> bool:
        """Approve a fine-tuning job.

        Args:
            job_id: Job to approve
            approver_id: Who approved

        Returns:
            True if approved, False if not found
        """
        for job in self._jobs:
            if job.job_id == job_id and job.status == "queued":
                job.status = "approved"
                emit_finetune_receipt(job, "approved", approver_id)
                return True
        return False

    def decline_job(self, job_id: str, reason: str = "") -> bool:
        """Decline a fine-tuning job.

        Args:
            job_id: Job to decline
            reason: Reason for declining

        Returns:
            True if declined, False if not found
        """
        for job in self._jobs:
            if job.job_id == job_id and job.status == "queued":
                job.status = "declined"
                job.notes = reason
                emit_finetune_receipt(job, "declined")
                return True
        return False

    def export_job(self, job_id: str, output_dir: str) -> Optional[str]:
        """Export job data for fine-tuning.

        Args:
            job_id: Job to export
            output_dir: Output directory

        Returns:
            Export path or None
        """
        job = None
        for j in self._jobs:
            if j.job_id == job_id:
                job = j
                break

        if not job or job.status not in ("approved", "queued"):
            return None

        # Get examples from queue
        all_examples = get_training_queue()
        job_examples = [
            e for e in all_examples
            if e.example_id in set(job.example_ids)
        ]

        if not job_examples:
            return None

        # Export
        output_path = f"{output_dir}/finetune_{job.job_id}.jsonl"
        export_jsonl(job_examples, output_path)

        job.export_path = output_path
        return output_path

    def submit_job(self, job_id: str) -> bool:
        """Mark job as submitted (external process will handle).

        Args:
            job_id: Job to submit

        Returns:
            True if submitted, False if not found/not approved
        """
        for job in self._jobs:
            if job.job_id == job_id and job.status == "approved":
                job.status = "submitted"
                job.submitted_at = datetime.now(timezone.utc).isoformat()
                emit_finetune_receipt(job, "submitted")
                return True
        return False

    def get_queued_jobs(self) -> List[FinetuneJob]:
        """Get all queued jobs."""
        return [j for j in self._jobs if j.status == "queued"]

    def get_approved_jobs(self) -> List[FinetuneJob]:
        """Get all approved jobs."""
        return [j for j in self._jobs if j.status == "approved"]

    def get_job(self, job_id: str) -> Optional[FinetuneJob]:
        """Get job by ID."""
        for j in self._jobs:
            if j.job_id == job_id:
                return j
        return None

    def get_stats(self) -> dict:
        """Get feedback loop statistics."""
        status_counts = {}
        for j in self._jobs:
            status_counts[j.status] = status_counts.get(j.status, 0) + 1

        return {
            "total_jobs": len(self._jobs),
            "by_status": status_counts,
            "total_examples_queued": sum(j.example_count for j in self._jobs),
            "min_quality_threshold": self._min_quality,
            "batch_size": self._batch_size
        }


# =============================================================================
# MODULE-LEVEL INTERFACE
# =============================================================================

_feedback_loop: Optional[FeedbackLoop] = None


def get_feedback_loop() -> FeedbackLoop:
    """Get the global feedback loop."""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
    return _feedback_loop


def queue_for_finetuning(examples: Optional[List[TrainingExample]] = None,
                         notes: Optional[str] = None) -> FinetuneJob:
    """Queue examples for fine-tuning.

    Args:
        examples: Examples to queue (None = auto)
        notes: Job notes

    Returns:
        Created job
    """
    loop = get_feedback_loop()
    return loop.create_job(examples, notes)


def get_pending_finetuning() -> List[FinetuneJob]:
    """Get all pending fine-tuning jobs.

    Returns:
        List of queued and approved jobs
    """
    loop = get_feedback_loop()
    return loop.get_queued_jobs() + loop.get_approved_jobs()


def emit_finetune_receipt(job: FinetuneJob, action: str,
                          actor: Optional[str] = None) -> dict:
    """Emit fine-tuning receipt.

    Args:
        job: FinetuneJob
        action: Action taken
        actor: Who took action

    Returns:
        Receipt dict
    """
    data = {
        "job_id": job.job_id,
        "action": action,
        "status": job.status,
        "example_count": job.example_count,
        "created_at": job.created_at
    }

    if actor:
        data["actor"] = actor

    if job.export_path:
        data["export_path"] = job.export_path

    if job.submitted_at:
        data["submitted_at"] = job.submitted_at

    if job.notes:
        data["notes"] = job.notes

    return emit_receipt("finetune", data, silent=True)


def emit_rollback_receipt(model_version: str, rollback_to: str,
                          reason: str) -> dict:
    """Emit model rollback receipt.

    Args:
        model_version: Current model version
        rollback_to: Version to rollback to
        reason: Reason for rollback

    Returns:
        Receipt dict
    """
    return emit_receipt("rollback", {
        "from_version": model_version,
        "to_version": rollback_to,
        "reason": reason,
        "rolled_back_at": datetime.now(timezone.utc).isoformat()
    }, silent=True)
