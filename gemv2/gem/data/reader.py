"""
Data readers.
"""


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pyarrow as pa
import polars as pl

from ..core.data import SourceSpec


class DataReader(ABC):
    @abstractmethod
    def read(
        self,
        source_spec: SourceSpec,
        date_start: int,
        date_end: int,
        filters: Optional[Any] = None,
    ) -> pl.DataFrame:
        pass

    def read_sources(
        self,
        source_spec_dict: Dict[str, SourceSpec],
        date_start: int,
        date_end: int,
    ) -> Dict[str, pl.DataFrame]:
        return {
            source_name: self.read(source_spec, date_start, date_end)
            for source_name, source_spec in source_spec_dict.items()
        }


class FeatherReader(DataReader):
    @staticmethod
    def _standardize_columns(
        df: pl.DataFrame,
        date_col: str,
        code_col: str,
        rename_map: Optional[Dict[str, str]] = None,
    ) -> pl.DataFrame:
        """Rename source-specific column names to canonical 'date' / 'code' names,
        then apply any extra ``rename_map`` entries."""
        renames: Dict[str, str] = {}
        if date_col != "date" and date_col in df.columns:
            renames[date_col] = "date"
        if code_col != "code" and code_col in df.columns:
            renames[code_col] = "code"
        if rename_map:
            renames.update(rename_map)
        if renames:
            df = df.rename(renames)
        return df

    def _filter_date_range(self, df: pl.DataFrame, date_start: int, date_end: int) -> pl.DataFrame:
        if "date" not in df.columns:
            return df
        date_dtype = df["date"].dtype
        if date_dtype == pl.Utf8 or date_dtype == pl.String:
            date_expr = pl.col("date").cast(pl.Int32)
        else:
            date_expr = pl.col("date")
        return df.filter((date_expr >= date_start) & (date_expr <= date_end))

    @staticmethod
    def _resolve_value_col(source_spec: SourceSpec) -> str:
        if source_spec.value_col:
            return source_spec.value_col
        return f"{source_spec.name}_value"

    def _empty_frame_for_source(self, source_spec: SourceSpec) -> pl.DataFrame:
        schema: Dict[str, pl.DataType] = {
            "date": pl.Int32,
            "code": pl.Utf8,
        }
        value_col = source_spec.value_col
        if value_col:
            schema[value_col] = pl.Float32
        return pl.DataFrame(schema=schema)

    def _process_dataframe(
        self,
        df: pl.DataFrame,
        source_spec: SourceSpec,
        date_start: int,
        date_end: int,
    ) -> pl.DataFrame:
        layout = (source_spec.layout or "tabular").lower()
        pivot = (source_spec.pivot or "").lower()
        date_col = source_spec.date_col or "date"
        code_col = source_spec.code_col or "code"
        value_col = self._resolve_value_col(source_spec)
        index_col = source_spec.index_col or "index"

        if layout == "wide":
            if pivot == "from_index":
                if index_col not in df.columns:
                    raise ValueError(
                        f"Source '{source_spec.name}' expects index_col '{index_col}' for pivot 'from_index'."
                    )
                df = df.unpivot(
                    index=[index_col],
                    variable_name=code_col,
                    value_name=value_col,
                )
                if index_col != date_col:
                    df = df.rename({index_col: date_col})
            elif pivot == "from_date":
                if date_col not in df.columns:
                    raise ValueError(
                        f"Source '{source_spec.name}' expects date_col '{date_col}' for pivot 'from_date'."
                    )
                df = df.unpivot(
                    index=[date_col],
                    variable_name=code_col,
                    value_name=value_col,
                )
            else:
                raise ValueError(
                    f"Source '{source_spec.name}' has layout='wide' but unsupported pivot='{source_spec.pivot}'. "
                    "Expected one of: from_index, from_date."
                )
        elif layout not in {"tabular", "long"}:
            raise ValueError(
                f"Unsupported layout '{source_spec.layout}' for source '{source_spec.name}'."
            )

        df = self._standardize_columns(
            df,
            date_col=date_col,
            code_col=code_col,
            rename_map=source_spec.rename_map,
        )
        return self._filter_date_range(df, date_start, date_end)

    def read(
        self,
        source_spec: SourceSpec,
        date_start: int,
        date_end: int,
        filters: Optional[Any] = None,
    ) -> pl.DataFrame:
        path = Path(source_spec.path)
        feather_reader = pa.ipc.open_file(str(path))
        batches = []

        for batch_idx in range(feather_reader.num_record_batches):
            batch = feather_reader.get_record_batch(batch_idx)
            batch_df = pl.from_arrow(batch)
            processed_batch = self._process_dataframe(batch_df, source_spec, date_start, date_end)
            batches.append(processed_batch)

        if not batches:
            return self._empty_frame_for_source(source_spec)

        return pl.concat(batches, how="vertical_relaxed")
