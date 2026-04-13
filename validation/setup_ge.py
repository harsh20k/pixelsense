"""One-time setup for Great Expectations suites + checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import great_expectations as gx
from great_expectations.core import ExpectationConfiguration

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

GX_ROOT = Path(__file__).parent / "gx"
DOMAINS = ["pixel_art", "photorealistic"]


def ensure_runtime_datasource(context: gx.DataContext) -> None:
    """Create GE runtime datasource used by validate_batch.py."""
    context.add_or_update_datasource(
        name="pixelsense_runtime",
        class_name="Datasource",
        execution_engine={"class_name": "PandasExecutionEngine"},
        data_connectors={
            "runtime_connector": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["domain", "split"],
            }
        },
    )


def build_suite(context: gx.DataContext, suite_name: str) -> None:
    suite = context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    expectations: list[ExpectationConfiguration] = [
        ExpectationConfiguration(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": 1},
        ),
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={"column_list": ["pixel_value", "height", "width", "channels"]},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "pixel_value",
                "min_value": -1.0,
                "max_value": 1.0,
                "mostly": 1.0,
            },
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "pixel_value"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "height", "value_set": [256]},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "width", "value_set": [256]},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "channels", "value_set": [3]},
        ),
    ]

    for exp in expectations:
        suite.add_expectation(exp)

    context.save_expectation_suite(suite)
    log.info("Saved suite: %s", suite_name)


def build_checkpoint(context: gx.DataContext) -> None:
    context.add_or_update_checkpoint(
        name="pixelsense_checkpoint",
        config_version=1.0,
        class_name="SimpleCheckpoint",
        validations=[],
    )
    log.info("Saved checkpoint: pixelsense_checkpoint")


def main() -> None:
    context = gx.get_context(context_root_dir=str(GX_ROOT))
    ensure_runtime_datasource(context)

    for domain in DOMAINS:
        build_suite(context, suite_name=f"{domain}_suite")

    build_checkpoint(context)
    log.info("GE setup complete. Context at: %s", GX_ROOT)


if __name__ == "__main__":
    main()
