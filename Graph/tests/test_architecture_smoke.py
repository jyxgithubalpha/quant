from domain.config import ExperimentConfig


def test_experiment_config_defaults():
    cfg = ExperimentConfig()
    assert cfg.train.max_epochs > 0
    assert cfg.eval.money > 0
    assert "baseline" in cfg.ablations
