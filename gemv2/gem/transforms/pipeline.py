"""
TransformPipeline -- chains multiple BaseTransform instances.
"""

from typing import List, Optional, Tuple

import numpy as np

from ..core.data import ProcessedBundle, SplitBundle, SplitView
from ..core.training import TransformContext
from .base import BaseTransform, extract_date_keys


class TransformPipeline:
    """
    Data processing pipeline.

    Manages a sequence of transforms and provides a single entry-point
    ``fit_transform_views`` that fits on the training split and transforms
    all three splits in one call.
    """

    def __init__(self, transforms: Optional[List[BaseTransform]] = None) -> None:
        self.transforms: List[BaseTransform] = transforms or []
        self._context: TransformContext = {}

    # -- context ----------------------------------------------------------

    def set_context(self, ctx: TransformContext) -> "TransformPipeline":
        """Set context and propagate to every transform."""
        self._context = ctx
        for t in self.transforms:
            t.set_context(ctx)
        return self

    def add_transform(self, transform: BaseTransform) -> "TransformPipeline":
        """Append a transform to the pipeline."""
        transform.set_context(self._context)
        self.transforms.append(transform)
        return self

    # -- low-level array API ----------------------------------------------

    def fit(
        self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None,
    ) -> "TransformPipeline":
        """Fit all transforms sequentially.  Each sees the output of the previous."""
        X_cur, y_cur = X.copy(), y.copy()
        for t in self.transforms:
            t.fit(X_cur, y_cur, keys)
            X_cur, y_cur = t.transform(X_cur, y_cur, keys)
        return self

    def transform(
        self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all (already fitted) transforms."""
        X_cur, y_cur = X.copy(), y.copy()
        for t in self.transforms:
            X_cur, y_cur = t.transform(X_cur, y_cur, keys)
        return X_cur, y_cur

    def inverse_transform(
        self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse transform (reverse order)."""
        X_cur, y_cur = X.copy(), y.copy()
        for t in reversed(self.transforms):
            X_cur, y_cur = t.inverse_transform(X_cur, y_cur, keys)
        return X_cur, y_cur

    # -- feature name tracking --------------------------------------------

    def get_output_feature_names(self, input_names: List[str]) -> List[str]:
        """Track feature names through the entire pipeline."""
        names = input_names
        for t in self.transforms:
            names = t.get_output_feature_names(names)
        return names

    # -- high-level views API (called by ModelPipeline) -------------------

    def fit_transform_views(
        self,
        bundle: SplitBundle,
        context: Optional[TransformContext] = None,
    ) -> ProcessedBundle:
        """
        Fit on train split, then transform all three splits.

        This is the main entry-point called by ``ModelPipeline._ensure_processed``.
        """
        if context is not None:
            self.set_context(context)

        # Fit on training data
        train_keys = extract_date_keys(bundle.train.keys)
        self.fit(bundle.train.X, bundle.train.y, train_keys)

        # Derive output feature names
        output_names = self.get_output_feature_names(bundle.train.feature_names)

        # Transform all splits
        def _transform_view(view: SplitView) -> SplitView:
            k = extract_date_keys(view.keys)
            X_out, y_out = self.transform(view.X, view.y, k)
            return SplitView(
                indices=view.indices,
                X=X_out,
                y=y_out,
                keys=view.keys,
                feature_names=output_names,
                label_names=view.label_names,
                extra=view.extra,
                group=view.group,
            )

        return ProcessedBundle(
            train=_transform_view(bundle.train),
            val=_transform_view(bundle.val),
            test=_transform_view(bundle.test),
            split_spec=bundle.split_spec,
        )
