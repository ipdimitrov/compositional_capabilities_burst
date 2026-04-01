import torch
from omegaconf import DictConfig
from torch import nn


class AutoLstm(nn.Module):
    """Autoregressive LSTM language model."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize embedding, LSTM layers, and output projection."""
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lstm = nn.LSTM(
            config.n_embd, config.n_embd, num_layers=config.n_layer, batch_first=True, bias=True
        )
        self.fc = nn.Linear(config.n_embd, config.vocab_size)
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self.use_hidden = False

        self.apply(self._init_weights)

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with normal or orthogonal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param.data, gain=1.0)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute logits for the input token indices."""
        x_embd = self.wte(inp)

        if self.use_hidden:
            lstm_out, (hidden, cell) = self.lstm(x_embd, self.hidden)
            self.hidden = (hidden, cell)
        else:
            lstm_out, (hidden, cell) = self.lstm(x_embd, None)

        return self.fc(lstm_out)

