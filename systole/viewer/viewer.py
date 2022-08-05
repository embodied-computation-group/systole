# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import functools
import json
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import date2num
from matplotlib.widgets import SpanSelector

from systole.detection import ppg_peaks
from systole.plots import plot_raw


class Viewer:
    """Interactive plots based on Matplotlib for visualization and edition of peaks
    detection in physiological signal.

    Parameters
    ----------
    figsize : tuple
        The size of the interactive Matplotlib figure for peaks edition. Defaults to
        `(15, 7)`.

    Notes
    -----
    This module is largely ispired by the peakdet toolbox
    (https://github.com/physiopy/peakdet).

    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (15, 7),
        input_folder: Union[str, PathLike] = "",
        output_folder: Union[str, PathLike] = "",
    ) -> None:

        self.figsize = figsize

        ##################
        # Create widgets #
        ##################

        self.bids_path = widgets.Textarea(
            value=input_folder,
            placeholder="Type something",
            description="BIDS folders:",
            disabled=False,
            layout=widgets.Layout(width="200px"),
        )
        self.session_ = widgets.Textarea(
            value="ses-session1",
            placeholder="Type something",
            description="Session:",
            disabled=False,
            layout=widgets.Layout(width="200px"),
        )
        self.modality_ = widgets.Textarea(
            value="beh",
            placeholder="Type something",
            description="Modality:",
            disabled=False,
            layout=widgets.Layout(width="200px"),
        )
        self.pattern_ = widgets.Textarea(
            value="task-",
            placeholder="Type something",
            description="Pattern:",
            disabled=False,
            layout=widgets.Layout(width="200px"),
        )
        self.signal_type_ = widgets.Dropdown(
            options=["PPG", "ECG", "RESP"],
            value="PPG",
            description="Signal:",
            layout=widgets.Layout(width="200px"),
        )
        self.save_button_ = widgets.ToggleButton(
            value=False,
            description="Save modifications",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Description",
            icon="check",
            layout=widgets.Layout(width="200px"),
        )
        self.output_folder_ = widgets.Textarea(
            value=output_folder,
            placeholder="Type something",
            description="Output folder:",
            disabled=False,
            layout=widgets.Layout(width="200px"),
        )

        # Update the participant list from the BIDS parameters
        try:
            # Find the list of all participant from the BIDS logs
            self.participants_list = (
                pd.read_csv(Path(self.bids_path.value, "participants.tsv"), sep="\t")
                .participant_id.sort_values()
                .to_list()
            )
            # Filter participants that have no physio recording
            self.participants_list = [
                part
                for part in self.participants_list
                if any(
                    Path(
                        self.bids_path.value,
                        part,
                        self.session_.value,
                        self.modality_.value,
                    ).glob(f"*{self.pattern_.value}*_physio.tsv.gz")
                )
            ]
        except FileNotFoundError:
            self.participants_list = ["sub-"]
        self.participants_ = widgets.Dropdown(
            options=self.participants_list,
            value=self.participants_list[0],
            description="Participant ID",
            layout=widgets.Layout(width="200px"),
        )

        # Keep updated if dropdown menus are used
        self.bids_path.observe(self.update_list, names="value")
        self.session_.observe(self.update_list, names="value")
        self.modality_.observe(self.update_list, names="value")
        self.pattern_.observe(self.update_list, names="value")

        self.participants_.observe(self.plot_signal, names="value")

        self.save_button_.observe(self.save, names="value")

        # Show the navigator and main plot
        self.output = widgets.Output()
        self.box = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.bids_path,
                        self.session_,
                        self.participants_,
                        self.modality_,
                        self.pattern_,
                    ]
                ),
                widgets.HBox([self.save_button_, self.output_folder_]),
            ]
        )

    def update_list(self, change):
        """Updating the input files when the dropdown menues are used."""
        self.participants_list = (
            pd.read_csv(Path(self.bids_path.value, "participants.tsv"), sep="\t")
            .participant_id.sort_values()
            .to_list()
        )
        self.participants_list = [
            part
            for part in self.participants_list
            if any(
                Path(
                    self.bids_path.value,
                    self.participants_.value,
                    self.session_.value,
                    self.modality_.value,
                ).glob(f"*{self.pattern_.value}*_physio.tsv.gz")
            )
        ]

        self.participants_.option = self.participants_list

    def plot_signal(self, change):

        self.output.clear_output()
        with self.output:

            # Start the interactive editor for peaks correction
            self.editor = Editor(
                bids_folder=self.bids_path.value,
                participant_id=self.participants_.value,
                patterns=self.pattern_.value,
                sessions=self.session_.value,
                modality=self.modality_.value,
                figsize=self.figsize,
            )
            plt.show()

    def save(self, change):
        """Save the JSON file logging the peaks correction into the BIDS derivatives
        folder."""
        self.editor.save(output_folder=self.output_folder_.value)


class Editor:
    """Class for creating a plot of signal and instantaneous frequency for manually
    editing peaks vectors.

    Parameters
    ----------
    physio_file : str | PathLike
        Path to the physiological recording.
    json_file : str | PathLike
        Path to the JSON metadata of the physiological recording.

    """

    def __init__(
        self,
        bids_folder: Optional[Union[str, PathLike]],
        participant_id: Optional[Union[str, List]],
        pattern: Optional[Union[str, List]],
        session: Optional[Union[str, List[str]]] = "ses-session1",
        modality: Optional[Union[str, PathLike]] = "beh",
        physio_file: Union[str, PathLike] = "",
        json_file: Union[str, PathLike] = "",
        figsize: Tuple[int, int] = (15, 7),
    ) -> None:

        if bids_folder is not None:
            self.bids_folder = bids_folder
        if participant_id is not None:
            self.participant_id = participant_id
        if pattern is not None:
            self.pattern = pattern
        if session is not None:
            self.session = session
        if modality is not None:
            self.modality = modality

        if physio_file:
            self.physio_file = Path(physio_file)
            self.json_file = Path(json_file)
        else:
            self.physio_file = list(
                Path(
                    self.bids_folder,
                    str(self.participant_id),
                    str(self.session),
                    self.modality,
                ).glob(f"*{self.pattern}*_physio.tsv.gz")
            )[0]
            self.json_file = list(
                Path(
                    self.bids_folder,
                    str(self.participant_id),
                    str(self.session),
                    self.modality,
                ).glob(f"*{self.pattern}*.json")
            )[0]

        print(f"Loading {self.physio_file}")

        # Opening JSON file and extract metadata
        f = open(self.json_file)
        json_data = json.load(f)

        self.sfreq = json_data["SamplingFrequency"]
        self.input_columns_names = json_data["Columns"]

        try:
            self.recording_start_time = json_data["StartTime"]
            self.recording_end_time = json_data["EndTime"]
        except KeyError:
            self.recording_start_time, self.recording_end_time = None, None

        f.close()

        self.input_signal = pd.read_csv(
            self.physio_file,
            sep="\t",
            compression="gzip",
            names=self.input_columns_names,
        )["cardiac"]

        # Peaks detection on the input signal
        self.signal, self.peaks = ppg_peaks(signal=self.input_signal, sfreq=self.sfreq)
        self.initial_peaks = self.peaks.copy()

        # Create a time vector from signal length and convert it to Matplotlib ax values
        self.time = pd.to_datetime(
            np.arange(0, len(self.signal)), unit="ms", origin="unix"
        )
        self.x_vec = date2num(self.time)

        # Create the main plot_raw instance
        self.fig, self.ax = plt.subplots(nrows=2, figsize=figsize, sharex=True)
        plot_raw(
            signal=self.signal,
            peaks=self.peaks,
            modality="ppg",
            backend="matplotlib",
            show_heart_rate=True,
            show_artefacts=True,
            sfreq=1000,
            ax=[self.ax[0], self.ax[1]],
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # two selectors for rejection (left mouse) and deletion (right mouse)
        self.delete = functools.partial(self.on_remove)
        self.span1 = SpanSelector(
            self.ax[0],
            self.delete,
            "horizontal",
            button=1,
            props=dict(facecolor="red", alpha=0.2),
            useblit=True,
        )
        self.add = functools.partial(self.on_add)
        self.span2 = SpanSelector(
            self.ax[0],
            self.add,
            "horizontal",
            button=3,
            props=dict(facecolor="green", alpha=0.2),
            useblit=True,
        )

    def on_remove(self, xmin, xmax):
        """Removes specified peaks by either rejection / deletion"""
        # Get the interval in sample idexes
        tmin, tmax = np.searchsorted(self.x_vec, (xmin, xmax))
        self.peaks[tmin:tmax] = False
        self.plot_signals()

    def on_add(self, xmin, xmax):
        """Add a new peak on the maximum signal value from the selected range."""
        # Get the interval in sample idexes
        tmin, tmax = np.searchsorted(self.x_vec, (xmin, xmax))
        self.peaks[tmin + np.argmax(self.signal[tmin:tmax])] = True
        self.plot_signals()

    def on_key(self, event):
        """Undoes last span select or quits peak editor"""
        # accept both control or Mac command key as selector
        self.event = event.key
        if event.key in ["ctrl+q", "super+d"]:
            self.quit()
        elif event.key in ["left"]:
            xlo, xhi = self.ax[0].get_xlim()
            step = xhi - xlo
            self.ax[0].set_xlim(xlo - step, xhi - step)
            self.fig.canvas.draw()
        elif event.key in ["right"]:
            xlo, xhi = self.ax[0].get_xlim()
            step = xhi - xlo
            self.ax[0].set_xlim(xlo + step, xhi + step)
            self.fig.canvas.draw()

    def plot_signals(self, plot=True):
        """Clears axes and plots data / peaks / troughs."""

        # Clear axes and redraw, retaining x-/y-axis zooms
        xlim, ylim = self.ax[0].get_xlim(), self.ax[0].get_ylim()
        xlim2, ylim2 = self.ax[1].get_xlim(), self.ax[1].get_ylim()
        self.ax[0].clear()
        self.ax[1].clear()
        plot_raw(
            signal=self.signal,
            peaks=self.peaks,
            modality="ppg",
            backend="matplotlib",
            show_heart_rate=True,
            show_artefacts=True,
            sfreq=1000,
            ax=[self.ax[0], self.ax[1]],
        )
        self.ax[0].set(xlim=xlim, ylim=ylim)
        self.ax[1].set(xlim=xlim2, ylim=ylim2)

        # Show span selectors
        # two selectors for rejection (left mouse) and deletion (right mouse)
        self.delete = functools.partial(self.on_remove)
        self.span1 = SpanSelector(
            self.ax[0],
            self.delete,
            "horizontal",
            button=1,
            props=dict(facecolor="red", alpha=0.2),
            useblit=True,
        )
        self.add = functools.partial(self.on_add)
        self.span2 = SpanSelector(
            self.ax[0],
            self.add,
            "horizontal",
            button=3,
            props=dict(facecolor="green", alpha=0.2),
            useblit=True,
        )

        self.fig.canvas.draw()

    def quit(self):
        """Quits editor"""
        plt.close(self.fig)

    def save(self, output_folder: Union[str, PathLike] = ""):
        """Save the corrected peaks in the derivatives folder."""

        if not output_folder:
            output_folder = self.bids_folder

        # Path to the corrected signal and JSON files
        self.corrected_json_file = Path(
            output_folder,
            "systole",
            "corrected",
            str(self.participant_id),
            str(self.session),
            self.modality,
            f"sub-{self.participant_id}_{self.session}_{self.pattern}_corrected.json",
        )

        if not self.corrected_json_file.parent.exists():
            self.corrected_json_file.parent.mkdir(parents=True)

        add_idx = np.where(self.peaks & ~self.initial_peaks)[0].tolist()
        remove_idx = np.where(~self.peaks & self.initial_peaks)[0].tolist()

        # Create the JSON metadata and save it in the corrected derivative folder
        data = {
            "ppg": {
                "add_idx": add_idx,
                "remove_idx": remove_idx,
                "bads": {"start": None, "end": None},
            },
        }

        with open(self.corrected_json_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
