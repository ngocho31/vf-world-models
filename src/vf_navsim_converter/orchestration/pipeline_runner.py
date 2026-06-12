"""Starter pipeline runner for VF to NAVSIM conversion."""

from dataclasses import dataclass
from typing import Optional, Sequence

from ..config import ConversionConfig
from ..constants import VF_REQUIRED_CAMERAS, VF_REQUIRED_TIMESERIES_STREAMS, VF_STREAM_TO_BUNDLE_FIELD
from ..contracts.models_canonical import CanonicalFrame
from ..contracts.models_navsim import NavsimFrameRecord, SceneMapSyncContext
from ..contracts.models_raw import RawSensorBundle
from ..readers.interfaces import CalibrationReader, CameraReader, TimeSeriesReader
from ..readers.discovery import DiscoveryResult, InputDiscovery
from ..readers.discovery import print_discovery_report, save_discovery_manifest
from ..readers.vf_readers import VFSourceReader
from ..readers.vf_readers import save_raw_sensor_bundle_manifest
from ..transforms.interfaces import CanonicalFrameBuilder, NavsimAssembler, SceneMapSyncResolver, TimestampAligner
from ..transforms.timestamp_alignment import TokenIntersectionTimestampAligner
from ..transforms.timestamp_alignment import save_timestamp_alignment_manifest
from ..transforms.canonical_frames import DefaultCanonicalFrameBuilder, save_canonical_manifest
from ..transforms.scene_map_sync import DefaultSceneMapSyncResolver, save_scene_map_sync_context
from ..transforms.navsim_assembly import DefaultNavsimAssembler, save_navsim_assembly_manifest
from ..writers.interfaces import NavsimLogsWriter, SensorBlobsWriter
from ..writers import DefaultSensorBlobsWriter
from ..writers import DefaultNavsimLogsWriter, save_stage7_artifact_manifest, save_conversion_report
from ..validators.interfaces import AssetValidator, SchemaValidator, TemporalValidator, ValidationResult


@dataclass
class PipelineComponents:
    discovery: Optional[InputDiscovery] = None
    camera_reader: Optional[CameraReader] = None
    timeseries_reader: Optional[TimeSeriesReader] = None
    calibration_reader: Optional[CalibrationReader] = None
    timestamp_aligner: Optional[TimestampAligner] = None
    canonical_builder: Optional[CanonicalFrameBuilder] = None
    scene_map_sync_resolver: Optional[SceneMapSyncResolver] = None
    navsim_assembler: Optional[NavsimAssembler] = None
    sensor_blobs_writer: Optional[SensorBlobsWriter] = None
    navsim_logs_writer: Optional[NavsimLogsWriter] = None
    schema_validator: Optional[SchemaValidator] = None
    temporal_validator: Optional[TemporalValidator] = None
    asset_validator: Optional[AssetValidator] = None


class PipelineRunner:
    """Coordinates conversion stages using dependency-injected interfaces."""

    def __init__(self, config: ConversionConfig) -> None:
        self._config = config
        self._components = PipelineComponents(
            discovery=InputDiscovery(config.pipeline.input_root),
            camera_reader=VFSourceReader(config.pipeline.input_root),
            timeseries_reader=VFSourceReader(config.pipeline.input_root),
            calibration_reader=VFSourceReader(config.pipeline.input_root),
            timestamp_aligner=TokenIntersectionTimestampAligner(
                max_sensor_nav_drift_ms=config.alignment.max_sensor_nav_drift_ms,
                frame_stride=config.alignment.frame_stride,
                max_frames=config.alignment.max_frames,
            ),
            canonical_builder=DefaultCanonicalFrameBuilder(
                reference_lidar_sensor=config.lidar.reference_sensor,
            ),
            scene_map_sync_resolver=DefaultSceneMapSyncResolver(
                output_root=config.pipeline.output_root,
                default_map_location=config.pipeline.map_location,
                map_sync_min_overlap=config.pipeline.map_sync_min_overlap,
            ),
            navsim_assembler=DefaultNavsimAssembler(
                vehicle_name=config.pipeline.vehicle_name,
                map_location=config.pipeline.map_location,
            ),
            sensor_blobs_writer=DefaultSensorBlobsWriter(
                input_root=config.pipeline.input_root,
                link_mode=config.io.link_mode,
                lidar_policy=config.lidar.policy,
            ),
            navsim_logs_writer=DefaultNavsimLogsWriter(),
        )

    def run(self) -> Sequence[ValidationResult]:
        """Run the conversion pipeline and return validation results."""

        # Stage 1: Discovery
        discovery_result = self.run_stage1_discovery()

        # Stage 2: Read raw signals into a bundle
        raw_bundle = self.run_stage2_read_raw_signals(discovery_result=discovery_result)

        # Stage 3: Align timestamps
        tokens = self.run_stage3_align_timestamps(raw_bundle)

        # Stage 4: Build canonical frames
        canonical_batch = self.run_stage4_build_canonical_frames(raw_bundle, tokens)

        # Stage 4.5: Resolve scene-map synchronization context
        scene_map_context = self.run_stage45_resolve_scene_map_sync(raw_bundle)

        # Stage 5: Assemble NAVSIM frames
        scene_artifact = self.run_stage5_assemble_navsim_frames(canonical_batch.frames, scene_map_context)

        # Stage 6: Materialize Assets (camera & lidar)
        if self._components.sensor_blobs_writer is not None:
            self._components.sensor_blobs_writer.materialize_camera_assets(scene_artifact.frames, self._config.pipeline.output_root)

        # Stage 7: Write Artifacts
        self.run_stage7_write_artifacts(scene_artifact.frames)

        # Placeholder for next stages (write logs, validation, etc.)
        return []

    def run_stage1_discovery(self) -> DiscoveryResult:
        """Run Stage 1 discovery through orchestration layer."""
        result = self._components.discovery.discover()

        # Print and save discovery results for visibility and debugging
        print_discovery_report(result)  # type: ignore[union-attr]
        path = self._config.pipeline.output_root / ".conversion_metadata" / "stage1_discovery_manifest.json"
        save_discovery_manifest(result, path)

        return result

    def run_stage2_read_raw_signals(
        self,
        discovery_result: DiscoveryResult,
    ) -> RawSensorBundle:
        """Run Stage 2 and return a populated RawSensorBundle."""
        if not discovery_result.is_valid:
            raise ValueError(
                "Stage 1 discovery is invalid; cannot run Stage 2. Errors: %s"
                % "; ".join(discovery_result.errors)
            )

        if self._components.camera_reader is None:
            raise ValueError("camera_reader is not configured")
        if self._components.timeseries_reader is None:
            raise ValueError("timeseries_reader is not configured")
        if self._components.calibration_reader is None:
            raise ValueError("calibration_reader is not configured")

        missing_cameras = [
            cam_name for cam_name in VF_REQUIRED_CAMERAS if cam_name not in discovery_result.cameras
        ]
        if missing_cameras:
            raise ValueError("Required cameras missing from discovery: %s" % ", ".join(missing_cameras))

        missing_streams = [
            stream_name
            for stream_name in VF_REQUIRED_TIMESERIES_STREAMS
            if stream_name not in discovery_result.timeseries_streams
        ]
        if missing_streams:
            raise ValueError("Required timeseries streams missing from discovery: %s" % ", ".join(missing_streams))

        camera_frames_by_sensor = {}
        for sensor_name in VF_REQUIRED_CAMERAS:
            frames = list(self._components.camera_reader.list_camera_frames(sensor_name))
            if not frames:
                raise ValueError("No frames found for required camera: %s" % sensor_name)
            camera_frames_by_sensor[sensor_name] = frames

        lidar_frames = []
        hdmap_assets = []
        if hasattr(self._components.camera_reader, "list_lidar_frames"):
            lidar_frames = list(self._components.camera_reader.list_lidar_frames())  # type: ignore[attr-defined]
        if hasattr(self._components.camera_reader, "list_hdmap_assets"):
            hdmap_assets = list(self._components.camera_reader.list_hdmap_assets())  # type: ignore[attr-defined]

        time_series_data = {field_name: [] for field_name in VF_STREAM_TO_BUNDLE_FIELD.values()}
        for stream_name in discovery_result.timeseries_streams:
            field_name = VF_STREAM_TO_BUNDLE_FIELD.get(stream_name)
            if field_name is None:
                continue
            rows = list(self._components.timeseries_reader.read_csv_stream(stream_name))
            time_series_data[field_name] = rows

        calibration_data = self._components.calibration_reader.read_calibration()

        bundle = RawSensorBundle(
            camera_frames_by_sensor=camera_frames_by_sensor,
            lidar_frames=lidar_frames,
            hdmap_assets=hdmap_assets,
            nav_rows=time_series_data.get("nav_rows", []),
            imu_rows=time_series_data.get("imu_rows", []),
            vehicle_info_rows=time_series_data.get("vehicle_info_rows", []),
            steer_rows=time_series_data.get("steer_rows", []),
            calibration=calibration_data,
        )

        path = self._config.pipeline.output_root / ".conversion_metadata" / "stage2_raw_sensor_bundle.json"
        save_raw_sensor_bundle_manifest(bundle, path)
        return bundle

    def run_stage3_align_timestamps(self, raw_bundle: RawSensorBundle) -> Sequence[str]:
        """Run Stage 3 timestamp alignment and return synchronized tokens."""
        if self._components.timestamp_aligner is None:
            raise ValueError("timestamp_aligner is not configured")

        aligned_tokens = list(self._components.timestamp_aligner.align(raw_bundle))
        if not aligned_tokens:
            raise ValueError("Stage 3 produced zero aligned tokens")

        path = self._config.pipeline.output_root / ".conversion_metadata" / "stage3_alignment_manifest.json"
        save_timestamp_alignment_manifest(aligned_tokens, path)
        return aligned_tokens

    def run_stage4_build_canonical_frames(
        self,
        raw_bundle: RawSensorBundle,
        tokens: Sequence[str],
    ):
        """Run Stage 4 canonical frame construction and return the canonical batch."""
        if self._components.canonical_builder is None:
            raise ValueError("canonical_builder is not configured")

        canonical_batch = self._components.canonical_builder.build_with_cache(raw_bundle, tokens)
        path = self._config.pipeline.output_root / ".conversion_metadata" / "stage4_canonical_manifest.json"
        save_canonical_manifest(canonical_batch, path)
        return canonical_batch

    def run_stage45_resolve_scene_map_sync(self, raw_bundle: RawSensorBundle) -> SceneMapSyncContext:
        """Run Stage 4.5 scene-map synchronization context resolution and return the context for use in Stage 5 assembly."""
        if self._components.scene_map_sync_resolver is None:
            raise ValueError("scene_map_sync_resolver is not configured")

        context = self._components.scene_map_sync_resolver.resolve(raw_bundle)
        path = self._config.pipeline.output_root / ".conversion_metadata" / "stage45_scene_map_sync_context.json"
        save_scene_map_sync_context(context, path)
        return context

    def run_stage5_assemble_navsim_frames(
        self,
        canonical_frames: Sequence[CanonicalFrame],
        scene_map_context: SceneMapSyncContext | None = None,
    ):
        """Run Stage 5 assembly and return NAVSIM scene artifact."""
        if self._components.navsim_assembler is None:
            raise ValueError("navsim_assembler is not configured")

        scene_artifact = self._components.navsim_assembler.assemble(
            canonical_frames,
            scene_map_context=scene_map_context,
        )
        path = self._config.pipeline.output_root / ".conversion_metadata" / "stage5_navsim_manifest.json"
        save_navsim_assembly_manifest(scene_artifact, path)
        return scene_artifact

    def run_stage7_write_artifacts(self, frames: Sequence[NavsimFrameRecord]) -> None:
        """Run Stage 7 artifact writing and emit manifest/report files."""
        if self._components.navsim_logs_writer is None:
            raise ValueError("navsim_logs_writer is not configured")

        output_root = self._config.pipeline.output_root
        log_name = frames[0].log_name if frames else "navsim_scene"
        pkl_path = output_root / "navsim_logs" / "trainval" / f"{log_name}.pkl"
        self._components.navsim_logs_writer.write_frames(frames, pkl_path)

        metadata_dir = output_root / ".conversion_metadata"
        save_stage7_artifact_manifest(frames, pkl_path, metadata_dir / "stage7_artifact_manifest.json")
        save_conversion_report(frames, pkl_path, metadata_dir / "conversion_report.json")
