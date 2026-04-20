"""
Tests for ray_trainer.py — RayXGBRankModel, ensure_ray_initialized, _compute_groups.

Run with:
    cd /home/user165/workspace/quant/xgboostv2
    pytest tests/test_ray_trainer.py -v
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# ─── Ensure xgboostv2 dir is on sys.path ──────────────────────────────────────
_XGBV2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _XGBV2_DIR not in sys.path:
    sys.path.insert(0, _XGBV2_DIR)

# Import the module and specific names
import ray_trainer
from ray_trainer import RayXGBRankModel, ensure_ray_initialized, _compute_groups


# ─── _compute_groups ──────────────────────────────────────────────────────────

class TestComputeGroups:
    def test_single_date(self):
        dates = np.array([20230101, 20230101, 20230101])
        groups = _compute_groups(dates)
        assert groups.tolist() == [3]
        assert groups.dtype == np.int32

    def test_multiple_dates(self):
        dates = np.array([20230101, 20230101, 20230102, 20230102, 20230102])
        groups = _compute_groups(dates)
        assert groups.tolist() == [2, 3]
        assert groups.dtype == np.int32

    def test_sorted_unique_dates(self):
        dates = np.array([1, 2, 3])
        groups = _compute_groups(dates)
        assert groups.tolist() == [1, 1, 1]

    def test_empty_returns_empty(self):
        dates = np.array([])
        groups = _compute_groups(dates)
        assert len(groups) == 0

    def test_counts_sum_equals_length(self):
        dates = np.array([10, 10, 20, 30, 30, 30])
        groups = _compute_groups(dates)
        assert int(groups.sum()) == len(dates)


# ─── ensure_ray_initialized ───────────────────────────────────────────────────
# Use patch.object(ray_trainer, 'ray') to reliably patch the module-level
# reference in ray_trainer's namespace, avoiding sys.modules lookup issues.

class TestEnsureRayInitialized:
    def test_calls_ray_init_when_not_initialized(self):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = False
        with patch.object(ray_trainer, "ray", mock_ray):
            ensure_ray_initialized()
        mock_ray.init.assert_called_once_with()

    def test_skips_ray_init_when_already_initialized(self):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True
        with patch.object(ray_trainer, "ray", mock_ray):
            ensure_ray_initialized()
        mock_ray.init.assert_not_called()

    def test_idempotent_double_call(self):
        """Second call (ray already up) must not call init again."""
        call_count = {"n": 0}

        def fake_is_initialized():
            call_count["n"] += 1
            return call_count["n"] > 1  # first: False → init; subsequent: True → skip

        mock_ray = MagicMock()
        mock_ray.is_initialized.side_effect = fake_is_initialized
        with patch.object(ray_trainer, "ray", mock_ray):
            ensure_ray_initialized()
            ensure_ray_initialized()
        assert mock_ray.init.call_count == 1


# ─── RayXGBRankModel.__init__ ─────────────────────────────────────────────────

class TestRayXGBRankModelInit:
    def test_params_merged_with_cuda_defaults(self):
        model = RayXGBRankModel({"eta": 0.05})
        assert model.params["eta"] == 0.05
        assert model.params["device"] == "cuda"
        assert model.params["tree_method"] == "hist"

    def test_initial_state_is_none(self):
        model = RayXGBRankModel({"eta": 0.1})
        assert model.model is None
        assert model.best_iteration is None
        assert model._model_bytes is None

    def test_accepts_legacy_kwargs(self):
        # num_actors, cpus_per_actor, gpus_per_actor kept for compat
        model = RayXGBRankModel(
            {"eta": 0.1}, num_actors=4, cpus_per_actor=2, gpus_per_actor=1
        )
        assert model.params["eta"] == 0.1


# ─── RayXGBRankModel.train ────────────────────────────────────────────────────

class TestRayXGBRankModelTrain:
    def _make_mock_ray(self):
        """Create a mock ray module that simulates ray.get(_ray_xgb_train.remote(...))."""
        # Simulate model bytes + best_iteration + n_imp returned by _ray_xgb_train
        fake_model_bytes = b'{"learner": {}}'  # minimal JSON
        mock_ray = MagicMock()
        mock_ray.get.return_value = (fake_model_bytes, 77, 5)
        return mock_ray, fake_model_bytes

    def test_train_sets_model_and_best_iteration(self):
        mock_ray, _ = self._make_mock_ray()
        rng = np.random.default_rng(0)
        X = rng.random((60, 5)).astype(np.float32)
        y = rng.random(60).astype(np.float32)
        dates = [20230101] * 30 + [20230102] * 30

        mock_booster = MagicMock()
        with patch.object(ray_trainer, "ray", mock_ray), \
             patch("xgboost.Booster", return_value=mock_booster):
            model = RayXGBRankModel({"eta": 0.1})
            model.train(X, y, dates, X, y, dates)

        assert model.model is mock_booster
        assert model.best_iteration == 77

    def test_train_calls_ray_get_once(self):
        mock_ray, _ = self._make_mock_ray()
        rng = np.random.default_rng(1)
        X = rng.random((30, 4)).astype(np.float32)
        y = rng.random(30).astype(np.float32)
        dates = [20230101] * 30

        mock_booster = MagicMock()
        with patch.object(ray_trainer, "ray", mock_ray), \
             patch("xgboost.Booster", return_value=mock_booster):
            model = RayXGBRankModel({"eta": 0.1})
            model.train(X, y, dates, X, y, dates)

        mock_ray.get.assert_called_once()


# ─── RayXGBRankModel.predict ─────────────────────────────────────────────────

class TestRayXGBRankModelPredict:
    def _trained_model(self):
        mock_booster = MagicMock(name="Booster")
        mock_booster.best_iteration = 42
        mock_booster.get_score.return_value = {}
        mock_booster.predict.return_value = np.array([0.1, 0.2, 0.3])
        model = RayXGBRankModel({"eta": 0.1})
        model.model = mock_booster
        # No _model_bytes → predict falls back to local booster.predict
        model._model_bytes = None
        return model

    def test_predict_returns_numpy_array_without_dates_codes(self):
        model = self._trained_model()
        X = np.random.rand(3, 5).astype(np.float32)
        result = model.predict(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_predict_returns_dataframe_with_dates_codes(self):
        import polars as pl
        model = self._trained_model()
        X = np.random.rand(3, 5).astype(np.float32)
        dates = [20230101, 20230101, 20230102]
        codes = ["S0001", "S0002", "S0003"]
        result = model.predict(X, dates=dates, codes=codes)

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["date", "Code", "score"]
        assert len(result) == 3

    def test_predict_uses_ray_remote_when_model_bytes_available(self):
        mock_preds = np.array([0.5, 0.6])
        mock_ray = MagicMock()
        mock_ray.get.return_value = mock_preds

        model = RayXGBRankModel({"eta": 0.1})
        model.model = MagicMock()
        model._model_bytes = b"fake_bytes"
        X = np.random.rand(2, 3).astype(np.float32)

        with patch.object(ray_trainer, "ray", mock_ray):
            result = model.predict(X)

        mock_ray.get.assert_called_once()
        np.testing.assert_array_equal(result, mock_preds)


# ─── RayXGBRankModel.save / load ─────────────────────────────────────────────

class TestRayXGBRankModelSaveLoad:
    def test_save_calls_save_model(self):
        mock_booster = MagicMock()
        model = RayXGBRankModel({"eta": 0.1})
        model.model = mock_booster
        model.save("/tmp/test_model.json")
        mock_booster.save_model.assert_called_once_with("/tmp/test_model.json")

    def test_save_noop_when_no_model(self):
        model = RayXGBRankModel({"eta": 0.1})
        # Should not raise even without a trained model
        model.save("/tmp/should_not_exist.json")

    def test_load_round_trip(self):
        mock_loaded = MagicMock()
        mock_loaded.save_raw.return_value = b"model_bytes"
        with patch("xgboost.Booster", return_value=mock_loaded):
            model = RayXGBRankModel({"eta": 0.1})
            model.load("/tmp/test_model.json")

        mock_loaded.load_model.assert_called_once_with("/tmp/test_model.json")
        assert model.model is mock_loaded
        assert model._model_bytes == b"model_bytes"

    def test_get_feature_importance(self):
        mock_booster = MagicMock()
        mock_booster.get_score.return_value = {"f0": 1.5, "f1": 0.3}
        model = RayXGBRankModel({"eta": 0.1})
        model.model = mock_booster
        imp = model.get_feature_importance()
        assert isinstance(imp, dict)
        assert "f0" in imp
