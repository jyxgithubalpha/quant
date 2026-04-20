from dataclasses import dataclass, field
import os


@dataclass
class DataConfig:
    fac_path: str = "/project/model_share/share_1/factor_data/fac20250212/fac20250212.fea"
    label_path: str = "/project/model_share/share_1/label_data/label1.fea"
    liquid_path: str = "/project/model_share/share_1/label_data/can_trade_amt1.fea"


@dataclass
class SplitConfig:
    valid_quarters: int = 2
    gap_days: int = 10


@dataclass
class TrainConfig:
    max_epochs: int = 20
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-2
    early_stop_patience: int = 8


@dataclass
class EvalConfig:
    money: float = 1.5e9
    top_k: int = 500


@dataclass
class ModelConfig:
    f_alpha: int = 900
    f_style: int = 40
    f_meta: int = 2
    hist_len: int = 20

    d_style: int = 32
    d_alpha: int = 64
    d_tmp: int = 32
    d_model: int = 96
    d_edge: int = 8

    temporal_encoder: str = "tcn"
    topk_sim: int = 20
    topk_dyn: int = 20
    topk_latent: int = 20
    n_industries: int = 8

    n_prop_layers: int = 2
    n_heads: int = 4
    composer: str = "semiring"

    use_prior: bool = True
    use_sim: bool = True
    use_dynamic: bool = True
    use_latent: bool = True

    w_rank: float = 1.0
    w_ic: float = 0.5
    w_reg: float = 1e-3

    two_stage: bool = True
    pretrain_epochs: int = 3
    dropout: float = 0.2


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    results_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"))
    quarters: list[tuple[int, int]] = field(default_factory=lambda: [
        (2023, 1), (2023, 2), (2023, 3), (2023, 4),
        (2024, 1), (2024, 2), (2024, 3), (2024, 4),
    ])
    ablations: dict = field(default_factory=lambda: {
        "baseline": {},
        "no_latent": {"use_latent": False},
        "no_prior": {"use_prior": False},
        "no_dynamic": {"use_dynamic": False},
        "no_sim": {"use_sim": False},
        "sum_only": {"composer": "sum"},
        "max_only": {"composer": "max"},
        "agr_only": {"composer": "agr"},
        "plain_attn": {"composer": "attn"},
        "single_layer": {"n_prop_layers": 1},
    })
