"""Evaluate character arcs in stories produced by narrative generators.

This package deliberately leaves ArcANE's existing role-playing evaluation
pipeline unchanged. It reuses the same broad idea of phase-level evidence and
trajectory-level judgment, but scores arcs that are intrinsic to one generated
story rather than responses against an ArcANE probe reference.
"""

from .models import DIMENSIONS

__all__ = ["DIMENSIONS"]
