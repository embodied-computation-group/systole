# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import argparse
import os
from typing import List, Optional, Union

import pandas as pd
import pkg_resources
from bokeh.embed import components
from bokeh.resources import INLINE
from jinja2 import Template

from systole import __version__ as version
from systole.reports.group_level import (
    frequency_domain_group_level,
    nonlinear_domain_group_level,
    time_domain_group_level,
)
from systole.reports.utils import create_reports


def wrapper(
    bids_folder: str,
    participants_id: Union[str, List],
    tasks: Union[str, List],
    sessions: Union[str, List[str]] = "session1",
    result_folder: Optional[str] = None,
    template_file=pkg_resources.resource_filename(__name__, "./group_level.html"),
    overwrite=True,
):
    """Create group-level interactive report. Will preprocesses subject level data and
    create reports if requested.

    Parameters
    ----------
    bids_folder : str
        The path to the BIDS structured folder containing the raw data.
    participants_id : str | list
        String or list of strings defining the participant(s) ID that should be
        processed.
    tasks : str | list
        The task(s) that will be analyzed. Should match with a task in the BIDS folder.
    sessions : str | list
        The session reference that should be analyzed. Should match a session number in
        the BIDS folder. Defaults to `"session1"`.
    result_folder : str
        The result folder.
    template_file : str
        The template file for group-level report.
    overwrite : bool
        Create new individual reports if `True` (default).

    """

    if isinstance(sessions, str):
        sessions = [sessions]
    elif isinstance(sessions, list):
        pass
    else:
        raise ValueError("Invalid sessions parameter.")

    if isinstance(tasks, str):
        tasks = [tasks]
    elif isinstance(tasks, list):
        pass
    else:
        raise ValueError("Invalid tasks parameter.")

    # Result folder
    if result_folder is None:
        derivatives = bids_folder + "/derivatives/systole/"
    else:
        derivatives = result_folder
    if not os.path.exists(derivatives):
        os.makedirs(derivatives)

    # Load HTML template
    with open(template_file, "r", encoding="utf-8") as f:
        html_template = f.read()

    for session in sessions:

        for task in tasks:

            # Output file name
            html_filename = (
                f"{result_folder}/group_level_ses-{session}_task-{task}.html"
            )

            # Create individual reports
            if overwrite:
                create_reports(
                    tasks=tasks,
                    sessions=session,
                    result_folder=derivatives,
                    participants_id=participants_id,
                    bids_folder=bids_folder,
                )

            # Gather individual metrics
            summary_df = pd.DataFrame([])
            for sub in participants_id:
                summary_file = f"{result_folder}/{sub}/{session}/{sub}_ses-{session}_task-{task}_features.tsv"
                if os.path.isfile(summary_file):
                    summary_df = summary_df.append(
                        pd.read_csv(summary_file, sep="\t"),
                        ignore_index=True,
                    )

            time_domain = time_domain_group_level(summary_df)
            frequency_domain = frequency_domain_group_level(summary_df)
            nonlinear_domain = nonlinear_domain_group_level(summary_df)

            # Embed plots in a dictionary
            plots = dict(
                time_domain=time_domain,
                frequency_domain=frequency_domain,
                nonlinear_domain=nonlinear_domain,
            )

            # Create script and div variables that will be passed to the template
            script, div = components(plots)

            # Load HTML template in memory
            template = Template(html_template)
            resources = INLINE.render()

            # Generate HTML txt variables
            show_ecg, show_ppg, show_respiration = True, True, True
            html = template.render(
                resources=resources,
                script=script,
                div=div,
                systole_version=version,
                show_ecg=show_ecg,
                show_ppg=show_ppg,
                show_respiration=show_respiration,
            )

            # Save the HTML file locally
            with open(html_filename, mode="w", encoding="utf-8") as f:
                f.write(html)
            print(f"Group-level report saved as {html_filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--bids_folder", action="store", help="Provides entry BIDS folder."
    )
    parser.add_argument(
        "-p", "--participants_id", action="store", help="Provides participants IDs."
    )
    parser.add_argument("-t", "--tasks", action="store", help="Provides tasks.")
    parser.add_argument(
        "-o", "--result_folder", action="store", help="Provides the result folder."
    )
    args = parser.parse_args()

    wrapper(
        tasks=args.tasks,
        result_folder=args.result_folder,
        participants_id=args.participants_id,
        bids_folder=args.bids_folder,
    )
