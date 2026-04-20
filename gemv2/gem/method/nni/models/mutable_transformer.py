"""
Mutable FT-Transformer for NNI weight-sharing NAS.

TODO: Implement weight-sharing transformer using nni.retiarii when NNI is installed.

Placeholder outline:

    import nni.retiarii.nn.pytorch as nn
    from nni.retiarii import model_wrapper

    @model_wrapper
    class MutableFTTransformer(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.d_token = nn.ValueChoice([32, 64, 96, 128], label="d_token")
            self.n_layers = nn.ValueChoice([2, 3, 4, 5, 6], label="n_layers")
            ...

        def forward(self, x):
            ...

Until NNI weight-sharing support is added, use NNITuner with random sampling
which evaluates full standalone FTTransformer models per trial.
"""

# Intentionally empty -- placeholder for future NNI Retiarii integration.
