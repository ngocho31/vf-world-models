"""CLI entrypoint for VF to NAVSIM conversion."""

import argparse
import logging
from pathlib import Path

from .config import AlignmentConfig, ConversionConfig, IOConfig, LidarConfig, PipelineConfig, ValidationConfig
from .config import HdMapConversionConfig
from .map_conversion.runner import HdMapConversionRunner
from .orchestration.pipeline_runner import PipelineRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VF to NAVSIM conversion CLI")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory of VF dataset")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for NAVSIM output")
    parser.add_argument("--vehicle-name", type=str, default="veh-01", help="Vehicle identifier")
    parser.add_argument("--map-location", type=str, default="vn-hdmap-demo", help="VN map location")
    parser.add_argument(
        "--map-version",
        type=str,
        default=None,
        help="Map version folder; default is current date (YYYY-MM-DD)",
    )
    return parser


def build_config(args: argparse.Namespace) -> ConversionConfig:
    return ConversionConfig(
        pipeline=PipelineConfig(
            input_root=args.input_root,
            output_root=args.output_root,
            vehicle_name=args.vehicle_name,
            map_location=args.map_location,
            map_version=args.map_version,
        ),
        alignment=AlignmentConfig(),
        lidar=LidarConfig(),
        io=IOConfig(),
        validation=ValidationConfig(),
    )


def build_hdmap_config(args: argparse.Namespace) -> HdMapConversionConfig:
    return HdMapConversionConfig(
        input_root=args.input_root / args.map_location,
        output_root=args.output_root,
        map_location=args.map_location,
        map_version=args.map_version,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = build_config(args)
    hdmap_config = build_hdmap_config(args)

    logger.info("=" * 70)
    logger.info("VF to NAVSIM Conversion Pipeline")
    logger.info("=" * 70)

    # Validate input root exists
    if not config.pipeline.input_root.exists():
        logger.error(f"Input root does not exist: {config.pipeline.input_root}")
        return 1

    # Prepare manifest directory
    manifest_dir = config.pipeline.output_root / ".conversion_metadata"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting HD map conversion...")
    map_runner = HdMapConversionRunner(hdmap_config)

    # Run HD map conversion first to ensure map output is available for log conversion and synchronization metadata
    try:
        map_summary = map_runner.run()
    except Exception as exc:
        logger.error(f"HD map conversion failed: {exc}")
        return 1

    # Log the map conversion summary
    logger.info(f"HD map conversion completed. Summary:")
    logger.info(f"  Map Location: {map_summary.map_location}")
    logger.info(f"  Map Version: {map_summary.map_version}")
    logger.info(f"  Way Count: {map_summary.way_count}")
    logger.info(f"  Boundary Count: {map_summary.boundary_count}")
    logger.info(f"  Baseline Path Count: {map_summary.baseline_path_count}")
    logger.info(f"  Lane Polygon Count: {map_summary.lane_polygon_count}")
    logger.info(f"  Crosswalk Count: {map_summary.crosswalk_count}")
    logger.info(f"  Carpark Count: {map_summary.carpark_count}")

    logger.info("Starting VF to NAVSIM conversion...")
    runner = PipelineRunner(config=config)

    # Run the full pipeline orchestration
    try:
        validation_results = runner.run()
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        return 1

    # Print validation results summary
    logger.info("Pipeline completed. Validation results:")
    for result in validation_results:
        logger.info(f"  {type(result).__name__}: {getattr(result, 'status', result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
