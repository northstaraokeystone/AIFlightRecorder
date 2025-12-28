"""Knowledge Module - CRAG Fallback (v2.2)"""

from .crag import (
    assess_knowledge_sufficiency,
    fallback_to_external,
    fuse_internal_external,
    emit_crag_receipt,
    CRAGResult,
    ExternalResult,
    FusedResult
)

__all__ = [
    "assess_knowledge_sufficiency",
    "fallback_to_external",
    "fuse_internal_external",
    "emit_crag_receipt",
    "CRAGResult",
    "ExternalResult",
    "FusedResult"
]
