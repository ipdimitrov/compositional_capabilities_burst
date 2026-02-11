from dataclasses import dataclass, field


@dataclass
class NetConfig:
    compile: bool = False
    vocab_size: int = 512
    context_size: int = 50
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 96
    dropout: float = 0.0
    bias: bool = False
    mlp: bool = True


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 200
    min_lr: float = 6e-6


@dataclass
class BurstExperimentConfig:
    seed: int = 42
    n_alphabets: int = 10
    seq_len: int = 6
    depth: int = 5
    n_functions: int = 3
    n_train_compositions: int = 50
    ndocuments: int = 5000
    neval_documents: int = 500

    total_steps: int = 3000
    batch_size: int = 32
    p_target: float = 0.10
    K_bursts: list[int] = field(default_factory=lambda: [2, 5, 10])
    undo_steps: int = 2000
    relearn_steps: int = 500
    p_relearn: float = 0.10
    eval_every: int = 50
    n_target: int = 5

    net: NetConfig = field(default_factory=NetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def summary_str(self) -> str:
        return (
            f"seed={self.seed}  |  net: {self.net.n_layer}L/{self.net.n_embd}d/{self.net.n_head}H"
            f"  |  train={self.total_steps} undo={self.undo_steps} relearn={self.relearn_steps}"
            f"  |  batch={self.batch_size}  p={self.p_target}  n_target={self.n_target}"
            f"  |  compositions={self.n_train_compositions}"
        )


@dataclass
class Idea3Config(BurstExperimentConfig):
    n_target_tasks: int = 4
    p_per_task: float = 0.10
