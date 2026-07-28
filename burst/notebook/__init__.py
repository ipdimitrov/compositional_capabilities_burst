"""Notebook-friendly API for the burst experiment.

Provides the same 3-phase pipeline (pretrain, finetune, forget) as a simple
function-call interface, built on top of burst/ internals.

Usage:
    from burst.notebook import make_data, pretrain, finetune, forget
    from burst.notebook import interp, report
"""

from burst.notebook import interp as interp
from burst.notebook import report as report
from burst.notebook.data import make_data as make_data
from burst.notebook.finetune import finetune as finetune
from burst.notebook.forget import forget as forget
from burst.notebook.pretrain import pretrain as pretrain

__all__ = ["finetune", "forget", "interp", "make_data", "pretrain", "report"]
