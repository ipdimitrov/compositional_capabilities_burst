"""Simplified burst experiment: 3 independent phases run from a notebook.

Usage:
    from simple import pretrain, finetune, forget, report
"""
from simple.data import make_data
from simple.pretrain import pretrain
from simple.finetune import finetune
from simple.forget import forget
from simple import report
