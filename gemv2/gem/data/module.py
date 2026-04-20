"""
DataModule coordinates reading, preprocessing and assembly.
"""


from typing import Dict

from ..core.data import GlobalDataset, SourceSpec
from .assembler import DatasetAssembler
from .preprocessors import (
    CrossSourcePipeline,
    SourcePipeline,
)
from .reader import DataReader


class DataModule:
    def __init__(
        self,
        sourcespec_dict: Dict[str, SourceSpec],
        data_reader: DataReader,
        single_source_pipeline_dict: Dict[str, SourcePipeline],
        cross_source_pipeline: CrossSourcePipeline,
        dataset_assembler: DatasetAssembler,
    ):
        self.sourcespec_dict = sourcespec_dict
        self.data_reader = data_reader
        self.single_source_pipeline_dict = single_source_pipeline_dict or {}
        self.cross_source_pipeline = cross_source_pipeline
        self.dataset_assembler = dataset_assembler
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not self.sourcespec_dict:
            raise ValueError("sourcespec_dict must not be empty.")

        supported_formats = {"feather"}
        for source_key, source_spec in self.sourcespec_dict.items():
            if source_spec.name != source_key:
                raise ValueError(
                    "SourceSpec name mismatch: "
                    f"dict key='{source_key}' but SourceSpec.name='{source_spec.name}'."
                )
            if source_spec.format not in supported_formats:
                raise ValueError(
                    f"Unsupported source format '{source_spec.format}' for source "
                    f"'{source_key}'. Supported formats: {sorted(supported_formats)}."
                )

        unknown_pipeline_sources = sorted(
            set(self.single_source_pipeline_dict.keys())
            - set(self.sourcespec_dict.keys())
        )
        if unknown_pipeline_sources:
            raise ValueError(
                "single_source_pipeline_dict contains unknown "
                f"sources: {unknown_pipeline_sources}"
            )

    def _read_sources(self, date_start: int, date_end: int) -> Dict[str, object]:
        return self.data_reader.read_sources(self.sourcespec_dict, date_start, date_end)

    def _apply_single_source_pipelines(self, source_dict: Dict) -> Dict:
        for source_name, pipeline in self.single_source_pipeline_dict.items():
            if source_name not in source_dict:
                continue
            source_dict[source_name] = pipeline.fit_transform(source_dict[source_name])
        return source_dict

    def _apply_cross_source_pipeline(self, source_dict: Dict) -> Dict:
        if self.cross_source_pipeline is None:
            return source_dict
        return self.cross_source_pipeline.fit_transform(source_dict)

    def build_dataset(self, date_start: int, date_end: int) -> GlobalDataset:
        source_dict = self._read_sources(date_start, date_end)
        source_dict = self._apply_single_source_pipelines(source_dict)
        source_dict = self._apply_cross_source_pipeline(source_dict)
        return self.dataset_assembler.assemble(source_dict)
