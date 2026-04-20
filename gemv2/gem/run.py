"""
Entry point for gem experiments.
"""

import logging
import os
from pathlib import Path

# Ensure matplotlib (and other tools that use tempfile) can find writable
# directories even in sandboxed environments where /tmp is not accessible.
_project_tmp = Path(__file__).parent.parent / ".tmp_cache"
_project_tmp.mkdir(parents=True, exist_ok=True)

if "MPLCONFIGDIR" not in os.environ:
    _mpl_cfg = _project_tmp / "matplotlib"
    _mpl_cfg.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cfg)

# Also redirect TMPDIR so that any library using tempfile finds a writable dir.
if not os.environ.get("TMPDIR") or not Path(os.environ["TMPDIR"]).exists():
    os.environ["TMPDIR"] = str(_project_tmp)
    import tempfile as _tempfile
    _tempfile.tempdir = str(_project_tmp)

import hydra
from omegaconf import DictConfig

from .bootstrap import build_runtime
from .experiment import ExperimentManager

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    runtime = build_runtime(cfg)

    manager = ExperimentManager(
        split_generator=runtime.split_generator,
        data_module=runtime.data_module,
        experiment_config=runtime.experiment_config,
        train_config=runtime.train_config,
        method_config=runtime.method_config,
    )

    log.info("Starting experiment: %s", runtime.experiment_config.name)
    results = manager.run()

    return results


if __name__ == "__main__":
    main()
