#!/usr/bin/env python
"""
CLI Functionality to generate reports and summaries of data sources.

To generate model output, see ./run_model.py
"""
import pathlib
import json
import logging
import click
from libs.datasets import JHUDataset
from libs.datasets import dataset_export
from libs.datasets import data_version

_logger = logging.getLogger(__name__)


@click.group('data')
def main():
    """Generate reports from raw data."""
    pass


@main.command("latest")
@click.option(
    "--output",
    "-o",
    help="Output directory",
    type=pathlib.Path,
    default=pathlib.Path("results/case_summaries"),
)
@data_version.with_git_version_click_option
def run_latest(version: data_version.DataVersion, output: pathlib.Path):
    """Get latest case values from JHU dataset."""
    output.mkdir(exist_ok=True)
    timeseries = JHUDataset.local().timeseries()
    state_summaries = dataset_export.latest_case_summaries_by_state(timeseries)

    for state_summary in state_summaries:
        state = state_summary.state
        output_file = output / f"{state}.summary.json"
        _logger.info(f"Writing latest data for {state} to {output_file}")
        output_file.write_text(state_summary.json(indent=2))

    version.write_file("case_summary", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
