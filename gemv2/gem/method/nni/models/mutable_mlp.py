"""
Mutable MLP for NNI weight-sharing NAS.

TODO: Implement weight-sharing MLP using nni.retiarii when NNI is installed.

Placeholder outline:

    import nni.retiarii.nn.pytorch as nn
    from nni.retiarii import model_wrapper

    @model_wrapper
    class MutableMLP(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.fc1 = nn.LayerChoice([
                nn.Linear(input_dim, 64),
                nn.Linear(input_dim, 128),
                nn.Linear(input_dim, 256),
                nn.Linear(input_dim, 512),
            ], label="hidden_size_0")
            ...

        def forward(self, x):
            ...

Until NNI weight-sharing support is added, use NNITuner with random sampling
which evaluates full standalone models per trial.
"""

# Intentionally empty -- placeholder for future NNI Retiarii integration.
