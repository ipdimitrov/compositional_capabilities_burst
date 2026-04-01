"""Simplified burst experiment: 3 independent phases run from a notebook.

Usage:
    from simple import pretrain, finetune, forget, report, interp
"""

from simple import interp as interp
from simple import report as report
from simple.data import make_data as make_data
from simple.finetune import finetune as finetune
from simple.forget import forget as forget
from simple.pretrain import pretrain as pretrain

__all__ = ["finetune", "forget", "interp", "make_data", "pretrain", "report"]
