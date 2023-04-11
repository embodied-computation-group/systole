# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import argparse
from os import PathLike
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import pkg_resources  # type:ignore
from bokeh.embed import components
from bokeh.resources import INLINE
from jinja2 import Template
from joblib import Parallel, delayed

from systole import __version__ as version
from systole.reports.group_level import (
    artefacts_group_level,
    frequency_domain_group_level,
    nonlinear_domain_group_level,
    time_domain_group_level,
)
from systole.reports.utils import create_reports


def wrapper(
    bids_folder: Union[str, PathLike],
    participants_id: Union[str, List],
    patterns: Union[str, List],
    sessions: Union[str, List[str]] = "ses-session1",
    modality: str = "beh",
    result_folder: Optional[Union[str, PathLike]] = None,
    template_file=pkg_resources.resource_filename(__name__, "./group_level.html"),
    overwrite=False,
    n_jobs: int = 1,
    html_report: bool = False,
):
    """Preprocesses subject level data and create reports (both  subject-level and
    group-level).

    Parameters
    ----------
    bids_folder :
        The path to the BIDS structured folder containing the raw data.
    participants_id :
        String or list of strings defining the participant(s) ID that should be
        processed.
    patterns :
        The string pattern(s) that will be looked for in the subject files. This pattern
        can help to identify a given pattern (e.g. `"pattern-mypattern"`) or more complexe
        recording IDs. This string should match with a file name pattern in the subject
        folder. If it is matching with more than one file, the operatio is aborted.
    sessions :
        The session reference that should be analyzed. Should match a session number in
        the BIDS folder. Defaults to `"session1"`.
    modality :
        The type of data (e.g. `"beh"`, `"func"`...) where the physiological recording
        is stored. Defaults to `"beh"`.
    result_folder :
        The result folder.
    template_file :
        The template file for group-level report.
    overwrite :
        Create new individual reports if `True` (default).
    n_jobs :
        Number of processes to run in parallel.
    html_report :
        If `True`, save an html report. This file embeds the signal for
        interactive visualization and can therefore be large, it is recommended to
        generate HTML reports for review or problematic recordings only.

    Raises
    ------
    ValueError:
        If invalid session, pattern or participant parameters are provided.

    """

    if isinstance(sessions, str):
        sessions = [sessions]
    elif isinstance(sessions, list):
        pass
    else:
        raise ValueError("Invalid sessions parameter.")

    if isinstance(participants_id, str):
        participants_id = [participants_id]
    elif isinstance(participants_id, list):
        pass
    else:
        raise ValueError("Invalid participants_id parameter.")

    if isinstance(patterns, str):
        patterns = [patterns]
    elif isinstance(patterns, list):
        pass
    else:
        raise ValueError("Invalid patterns parameter.")

    # Result folder
    if isinstance(bids_folder, str):
        bids_folder = Path(bids_folder)
    if isinstance(result_folder, str):
        result_folder = Path(result_folder)
    if result_folder is None:
        result_folder = Path(bids_folder, "derivatives", "systole")
    if not result_folder.exists():  # type: ignore
        result_folder.mkdir(parents=True)  # type: ignore

    # Load HTML template
    with open(template_file, "r", encoding="utf-8") as f:
        html_template = f.read()

    ######################
    # Individual reports #
    ######################
    if overwrite is True:
        for session in sessions:
            for pattern in patterns:
                print(f"Preprocessing... Session: {session} - Modality: {modality}.")
                Parallel(n_jobs=n_jobs)(
                    delayed(create_reports)(
                        participant_id=participant,
                        bids_folder=bids_folder,
                        result_folder=result_folder,
                        pattern=pattern,
                        session=session,
                        modality=modality,
                        html_report=html_report,
                    )
                    for participant in participants_id
                )

    #######################
    # Group level reports #
    #######################
    print("Create group-level report.")
    for session in sessions:
        for pattern in patterns:
            # Extract the file name pattern used to save the individual reports
            # Loop through all participants and try to find the corresponding file
            for participant in participants_id:
                physio_files = list(
                    Path(bids_folder, participant, session, modality).glob(
                        f"**/*{pattern}*_physio.tsv.gz"
                    )
                )
                if len(physio_files) > 0:
                    break

            if len(physio_files) == 0:
                print(
                    (
                        f"No individual reports found for session: {session}"
                        f" with pattern: {pattern}"
                    )
                )
                continue

            physio_file = physio_files[0]
            file_name = physio_file.name[:-14]
            file_name = f"ses-{file_name.split('ses-', 1)[1]}"

            # Output file name
            html_filename = Path(
                result_folder,
                f"group_level_{file_name}.html",
            )

            # Output file name
            df_filename = Path(
                result_folder,
                f"group_level_{file_name}.tsv",
            )

            # Gather individual metrics
            summary_df = pd.DataFrame([])
            for participant in participants_id:
                summary_file = Path(
                    result_folder,
                    participant,
                    session,
                    modality,
                    f"{participant}_{file_name}_features.tsv",
                )

                if summary_file.exists():
                    summary_df = pd.concat(
                        [summary_df, pd.read_csv(summary_file, sep="\t")],
                        ignore_index=True,
                    )

            time_domain = time_domain_group_level(summary_df)
            frequency_domain = frequency_domain_group_level(summary_df)
            nonlinear_domain = nonlinear_domain_group_level(summary_df)
            artefacts = artefacts_group_level(summary_df)

            # Embed plots in a dictionary
            plots = dict(
                time_domain=time_domain,
                frequency_domain=frequency_domain,
                nonlinear_domain=nonlinear_domain,
                artefacts=artefacts,
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

            # Save the group-level reports HTML file locally
            with open(html_filename, mode="w", encoding="utf-8") as f:
                f.write(html)
            print(f"Group-level HTML report saved as {html_filename}.")

            # Save the group-level reports HTML file locally
            summary_df.to_csv(df_filename, sep="\t", index=False)
            print(f"Group-level data frame summary report saved as {df_filename}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--bids_folder", action="store", help="Provides entry BIDS folder."
    )
    parser.add_argument(
        "-p",
        "--participants_id",
        action="store",
        help="Provides participants IDs.",
        nargs="+",
    )
    parser.add_argument("-t", "--patterns", action="store", help="Provides patterns.")
    parser.add_argument(
        "-r", "--html_reports", action="store", help="Save HTML reports is True."
    )
    parser.add_argument(
        "-o", "--result_folder", action="store", help="Provides the result folder."
    )
    parser.add_argument(
        "-n", "--n_jobs", action="store", help="Number of processes to run."
    )
    parser.add_argument(
        "-d", "--modality", action="store", help="Data type (eg. 'beh')."
    )
    parser.add_argument(
        "-w",
        "--overwrite",
        action="store",
        help="If `True`, overwrite preexisting files.",
    )
    args = parser.parse_args()

    # If no participants ID provided, load the participants' IDs from the BIDS directory
    if args.participants_id is None:
        args.participants_id = pd.read_csv(
            f"{args.bids_folder}/participants.tsv", sep="\t"
        )["participant_id"].to_list()

    # Define and create result folder automatically
    if args.result_folder is None:
        args.result_folder = Path(args.bids_folder, "derivatives", "systole/")
    if isinstance(args.result_folder, str):
        args.result_folder = Path(args.result_folder)
    if not args.result_folder.exists():
        args.result_folder.mkdir(parents=True)

    if args.overwrite is None:
        args.overwrite = False
    elif args.overwrite == "True":
        args.overwrite = True
    elif args.overwrite == "False":
        args.overwrite = False

    wrapper(
        patterns=args.patterns,
        result_folder=args.result_folder,
        participants_id=args.participants_id,
        modality=args.modality,
        bids_folder=args.bids_folder,
        n_jobs=int(args.n_jobs),
        overwrite=args.overwrite,
        html_report=args.html_reports,
    )
