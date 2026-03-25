"""Simplified burst experiment: 3 independent phases run from a notebook.

Usage:
    from burst.simple import pretrain, finetune, forget, report
"""
from burst.simple.data import make_data
from burst.simple.pretrain import pretrain
from burst.simple.finetune import finetune
from burst.simple.forget import forget
from burst.simple import report
