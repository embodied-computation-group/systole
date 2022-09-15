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

from systole.detection import ecg_peaks, ppg_peaks, rsp_peaks
from systole.plots import plot_raw
from systole.utils import ecg_strings, norm_bad_segments, ppg_strings, resp_strings


class Viewer:
    """Interactive plots based on Matplotlib for visualization and edition of peaks
    detection in physiological signal.

    Parameters
    ----------
    figsize : tuple
        The size of the interactive Matplotlib figure for peaks edition. Defaults to
        `(15, 7)`.
    input_folder : str | PathLike
        Path to the input BIDS folder.
    output_folder : str | PathLike
        Path to the output folder. This is where the JSON files containing peaks
        correction logs will be saved. If an empty strimg is provided (default), the
        results will be saved in `BIDS/derivative/systole/corrected/`.
    session : str | PathLike
        The BIDS sub-session where the pysio files are stored. Defaults to
        `"ses-session1"`.
    modality : str | PathLike
        The BIDS sub-modality where the pysio files are stored (e.g. `"func"` or
        `"beh"`).
    pattern : str | PathLike
        The string pattern that the pysio files should contain. This allows to refine
        the selection of possible physio files, in case the folders contains many
        `_physio-gz.tsv`.
    signal_type : str | PathLike
       The type of signal that are being analyzed. Can be `"PPG"`, `"ECG"` or `"RESP"`.
       Defaults to `"PPG"`.
    participant_id : str | None
        The participant ID as registered in the BIDS folder.

    Notes
    -----
    This module was largely inspired by the peakdet toolbox
    (https://github.com/physiopy/peakdet).

    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (15, 7),
        input_folder: Union[str, PathLike] = "",
        output_folder: Union[str, PathLike] = "",
        session: Union[str, PathLike] = "ses-session1",
        modality: Union[str, PathLike] = "beh",
        pattern: Union[str, PathLike] = "task-",
        signal_type: Union[str, PathLike] = "PPG",
        participant_id: Optional[str] = None,
    ) -> None:

        self.figsize = figsize

        ##################
        # Create widgets #
        ##################

        self.bids_path = widgets.Textarea(
            value=input_folder,
            placeholder="Type something",
            description="Input:",
            disabled=False,
            layout=widgets.Layout(width="250px"),
        )
        self.session_ = widgets.Textarea(
            value=session,
            placeholder="Type something",
            description="Session:",
            disabled=False,
            layout=widgets.Layout(width="250px"),
        )
        self.modality_ = widgets.Textarea(
            value=modality,
            placeholder="Type something",
            description="Modality:",
            disabled=False,
            layout=widgets.Layout(width="250px"),
        )
        self.pattern_ = widgets.Textarea(
            value=pattern,
            placeholder="Type something",
            description="Pattern:",
            disabled=False,
            layout=widgets.Layout(width="250px"),
        )
        self.signal_type_ = widgets.Dropdown(
            options=["PPG", "ECG", "RESP"],
            value=signal_type,
            description="Signal:",
            layout=widgets.Layout(width="250px"),
        )
        self.save_button_ = widgets.Button(
            description="Save modifications",
            disabled=False,
            button_style="",
            tooltip="Description",
            icon="save",
            layout=widgets.Layout(width="250px"),
        )
        self.output_folder_ = widgets.Textarea(
            value=output_folder,
            placeholder="Type something",
            description="Output:",
            disabled=False,
            layout=widgets.Layout(width="250px"),
        )
        self.edition_ = widgets.ToggleButtons(
            options=["Correction", "Rejection"], diabled=False
        )
        self.rejection_ = widgets.Checkbox(
            value=True, descrition="Valid recording", disabled=False, indent=True
        )

        # Update the participant list from the BIDS parameters
        try:
            # Get the list of all participant from the folders
            self.participants_list = [
                f.stem for f in list(Path(self.bids_path.value).glob("sub-*/"))
            ]
            if self.participants_list:
                self.participants_list.sort()

            # Filter participants that have no physio recording
            filter_participants_list = [
                part
                for part in self.participants_list
                if any(
                    Path(
                        self.bids_path.value,
                        part,
                        self.session_.value,
                        self.modality_.value,
                    ).glob(f"*{self.pattern_.value}*.tsv.gz")
                )
            ]
            if len(filter_participants_list) == 0:
                print(
                    "No file is matching the given paterns.\n"
                    f"... Input: {self.bids_path.value}\n"
                    f"... Session: {self.session_.value}\n"
                    f"... Modality: {self.modality_.value}\n"
                    f"... Pattern: {self.pattern_.value}"
                )
                self.participants_list = ["sub-"]
            else:
                self.participants_list = filter_participants_list
        except FileNotFoundError:
            print("Directory not found.")
            self.participants_list = ["sub-"]

        self.participant_id = self.participants_list[0]

        self.participants_ = widgets.Dropdown(
            options=self.participants_list,
            value=self.participant_id,
            description="Participant ID",
            layout=widgets.Layout(width="200px"),
        )

        # Keep updated if dropdown menus are used
        self.bids_path.observe(self.update_list, names="value")
        self.session_.observe(self.update_list, names="value")
        self.modality_.observe(self.update_list, names="value")
        self.pattern_.observe(self.update_list, names="value")
        self.signal_type_.observe(self.plot_signal, names="value")
        self.participants_.observe(self.plot_signal, names="value")
        self.save_button_.observe(self.save, names="value")

        # Show the navigator and main plot
        self.io_box = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.bids_path,
                        self.session_,
                        self.participants_,
                        self.modality_,
                    ]
                ),
                widgets.HBox([self.pattern_, self.signal_type_, self.output_folder_]),
            ]
        )

        self.commands_box = widgets.HBox(
            [self.edition_, self.rejection_, self.save_button_]
        )

        self.output = widgets.Output()

        # Plot the first pysio file if any
        self.plot_signal(change=None)

    def update_list(self, change):
        """Updating the list of participants when the text boxes are used."""
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
                ).glob(f"*{self.pattern_.value}*.tsv.gz")
            )
        ]

        self.participants_.option = self.participants_list

    def plot_signal(self, change):

        self.output.clear_output()
        with self.output:

            # Start the interactive editor for peaks correction
            self.editor = Editor(
                input_folder=self.bids_path.value,
                participant_id=self.participants_.value,
                pattern=self.pattern_.value,
                session=self.session_.value,
                modality=self.modality_.value,
                figsize=self.figsize,
                viewer=self,
            )
            plt.show()

    def save(self, change):
        """Save the JSON file logging the peaks correction into the BIDS derivatives
        folder."""
        # Save the new JSON file
        self.editor.save(output_folder=self.output_folder_.value)
        self.output.clear_output()
        # Printing saving message
        print(
            f"Saving participant {self.participants_.value} in {self.output_folder_.value} completed."
        )


class Editor:
    """Class for visualization and manual edition of peaks vectors.

    Parameters
    ----------
    input_folder : str | PathLike
        Path to the input BIDS folder.
    participant_id : str | list
        The participant ID, following BIDS standards.
    pattern : str | PathLike
        The string pattern that the pysio files should contain. This allows to refine
        the selection of possible physio files, in case the folders contains many
        `_physio-gz.tsv`.
    session : str | PathLike
        The BIDS sub-session where the pysio files are stored.
    modality : str | PathLike
        The BIDS sub-modality where the pysio files are stored (e.g. `"func"` or
        `"beh"`).
    physio_file : str | PathLike
        Path to the physiological recording.
    json_file : str | PathLike
        Path to the JSON metadata of the physiological recording.
    figsize : tuple
        The size of the interactive Matplotlib figure for peaks edition. Defaults to
        `(15, 7)`.
    viewer : :py:class`systole.viewer.Viewer` instance | None
        The viewer instance from which the editor is called.

    Attributes
    ----------
    initial_peaks : np.ndarray
        The peaks vector as detected using the default peaks detection algorithm.
    peaks : np.ndarray
        The corrected peaks vector after manual insertion/deletion.
    physio_file : PathLike | None
        Path to the physiological recording.
    json_file : PathLike | None
        Path to the sidecar JSON file.

    """

    def __init__(
        self,
        input_folder: Optional[Union[str, PathLike]],
        participant_id: Optional[Union[str, List]],
        pattern: Optional[Union[str, List]],
        session: Optional[Union[str, List[str]]] = "ses-session1",
        modality: Optional[Union[str, PathLike]] = "beh",
        physio_file: Union[str, PathLike] = "",
        json_file: Union[str, PathLike] = "",
        figsize: Tuple[int, int] = (15, 7),
        viewer: Optional[Viewer] = None,
    ) -> None:

        if input_folder is not None:
            self.input_folder = input_folder
        if participant_id is not None:
            self.participant_id = participant_id
        if pattern is not None:
            self.pattern = pattern
        if session is not None:
            self.session = session
        if modality is not None:
            self.modality = modality
        if viewer is not None:
            self.viewer = viewer

        self.figsize = figsize
        self.bad_segments: List[int] = []

        # Load the physio files and store parameters, then load the signal from the
        # physio file and perform peaks detection
        self = self.load_file(
            physio_file=physio_file, json_file=json_file
        ).load_signal()

        # If a signal is available, call the main plotting method
        if self.signal is not None:

            # Create a time vector from signal length and convert it to Matplotlib ax values
            self.time = pd.to_datetime(
                np.arange(0, len(self.signal)), unit="ms", origin="unix"
            )
            self.x_vec = date2num(self.time)

            # Create the main plot_raw instance
            self.fig, self.ax = plt.subplots(nrows=2, figsize=self.figsize, sharex=True)

            if self.bad_segments:
                bad_segments = [
                    (self.bad_segments[i], self.bad_segments[i + 1])
                    for i in range(0, len(self.bad_segments), 2)
                ]
            else:
                bad_segments = None

            plot_raw(
                signal=self.signal,
                peaks=self.peaks,
                modality=self.viewer.signal_type_.value.lower(),
                backend="matplotlib",
                show_heart_rate=True,
                show_artefacts=True,
                bad_segments=bad_segments,
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
        """Removes specified peaks by either rejection / deletion, or mark bad
        segments."""

        # Get the interval in sample idexes
        if self.viewer.edition_.value == "Correction":
            tmin, tmax = np.searchsorted(self.x_vec, (xmin, xmax))
            self.peaks[tmin:tmax] = False
            self.plot_signals()

        elif self.viewer.edition_.value == "Rejection":
            tmin, tmax = np.searchsorted(self.x_vec, (xmin, xmax))
            self.bad_segments.append(int(tmin))
            self.bad_segments.append(int(tmax))

            # Makes it a list of tuple
            bad_segments = [
                (self.bad_segments[i], self.bad_segments[i + 1])
                for i in range(0, len(self.bad_segments), 2)
            ]

            # Merge overlapping segments if any
            bad_segments = norm_bad_segments(bad_segments)
            self.bad_segments = list(np.array(bad_segments).flatten())
            print(self.bad_segments)
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

    def plot_signals(self):
        """Clears axes and plots data / peaks / troughs."""

        if self.signal is not None:

            # Clear axes and redraw, retaining x-/y-axis zooms
            xlim, ylim = self.ax[0].get_xlim(), self.ax[0].get_ylim()
            xlim2, ylim2 = self.ax[1].get_xlim(), self.ax[1].get_ylim()
            self.ax[0].clear()
            self.ax[1].clear()

            # Convert bad segments into list of tuple
            if self.bad_segments:
                bad_segments = [
                    (self.bad_segments[i], self.bad_segments[i + 1])
                    for i in range(0, len(self.bad_segments), 2)
                ]
            else:
                bad_segments = None

            plot_raw(
                signal=self.signal,
                peaks=self.peaks,
                modality=self.viewer.signal_type_.value.lower(),
                backend="matplotlib",
                show_heart_rate=True,
                show_artefacts=True,
                bad_segments=bad_segments,
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

            # Customize the plot a bit
            for ax in self.ax:
                ax.spines.right.set_visible(False)
                ax.spines.top.set_visible(False)
                ax.tick_params(
                    direction="in",
                    width=1.5,
                    which="major",
                    size=8,
                )
                ax.tick_params(direction="in", width=1, which="minor", size=4)
                ax.grid(which="major", alpha=0.5, linewidth=0.5)
            self.fig.set_tight_layout()
            plt.margins(x=0, y=0)
            plt.minorticks_on()
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.1, top=0.1)

            self.fig.canvas.draw()

            return self

    def quit(self):
        """Quits editor"""
        plt.close(self.fig)

    def save(self, output_folder: Union[str, PathLike] = ""):
        """Save the corrected peaks in the derivatives folder."""

        if not output_folder:
            output_folder = self.input_folder

        # Path to the corrected signal and JSON files
        self.corrected_json_file = Path(
            output_folder,
            "systole",
            "corrected",
            str(self.participant_id),
            str(self.session),
            self.modality,
            f"{self.participant_id}_{self.session}_{self.pattern}_corrected.json",
        )

        if not self.corrected_json_file.parent.exists():
            self.corrected_json_file.parent.mkdir(parents=True)

        if self.corrected_json_file.exists():
            # Load the existing corrected JSON data
            f = open(self.corrected_json_file)
            metadata = json.load(f)
            f.close()
        else:
            metadata = {}

        # Create the JSON metadata and save it in the corrected derivative folder
        corrected_info = {
            "valid": self.viewer.rejection_.value,
            "corrected_peaks": np.where(self.peaks)[0].tolist(),
            "bad_segments": [int(x) for x in self.bad_segments],
        }
        metadata[self.viewer.signal_type_.value] = corrected_info

        with open(self.corrected_json_file, "w") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def load_signal(self):
        """Find the signal in the physiological recording and perform peaks detection."""
        self.data = None
        self.peaks = None
        self.signal = None
        self.input_signal = None
        self.initial_peaks = None
        if self.physio_file is None:
            return self

        # Load the physiological signal from the BIDS folder
        self.data = pd.read_csv(
            self.physio_file,
            sep="\t",
            compression="gzip",
            names=self.input_columns_names,
        )
        self.data.columns = self.data.columns.str.lower()

        # Path to the corrected JSON files (if the signal has already been checked)
        self.corrected_json_file = Path(
            self.viewer.output_folder_.value,
            "derivatives",
            "systole",
            "corrected",
            str(self.participant_id),
            str(self.session),
            self.modality,
            f"{self.physio_file.stem[:-11]}_corrected.json",
        )

        if self.viewer.signal_type_.value == "ECG":
            ecg_col = [col for col in self.data.columns if col in ecg_strings]
            ecg_col = ecg_col[0] if len(ecg_col) > 0 else None

            self.input_signal = self.data[ecg_col].to_numpy()

            # Peaks detection on the input signal
            self.signal, self.peaks = ecg_peaks(
                signal=self.input_signal, sfreq=self.sfreq
            )
            self.initial_peaks = self.peaks.copy()
            print(f"Loading electrocardiogram - sfreq={self.sfreq} Hz.")

        elif self.viewer.signal_type_.value == "PPG":
            ppg_col = [col for col in self.data.columns if col in ppg_strings]
            ppg_col = ppg_col[0] if len(ppg_col) > 0 else None

            self.input_signal = self.data[ppg_col].to_numpy()

            # Peaks detection on the input signal
            self.signal, self.peaks = ppg_peaks(
                signal=self.input_signal, sfreq=self.sfreq
            )
            self.initial_peaks = self.peaks.copy()
            print(f"Loading photoplethysmogram - sfreq={self.sfreq} Hz.")

        elif self.viewer.signal_type_.value == "RESP":
            res_col = [col for col in self.data.columns if col in resp_strings]
            res_col = res_col[0] if len(res_col) > 0 else None

            self.input_signal = self.data[res_col].to_numpy()

            # Peaks detection on the input signal
            self.signal, (self.peaks, _) = rsp_peaks(
                signal=self.input_signal, sfreq=self.sfreq
            )
            self.initial_peaks = self.peaks.copy()
            print(f"Loading respiratory signal - sfreq={self.sfreq} Hz.")

        # Load peaks, bad segments and reject signal from the JSON logs
        if self.corrected_json_file.exists():

            # Opening JSON file and extract metadata
            f = open(self.corrected_json_file)
            json_data = json.load(f)

            self.bad_segments = json_data[self.viewer.signal_type_.value.lower()][
                "bad_segments"
            ]

            # If corrected peaks already exist, load here and replace the revious ones
            self.peaks = np.zeros(len(self.signal), dtype=bool)
            self.peaks[
                np.array(
                    json_data[self.viewer.signal_type_.value.lower()]["corrected_peaks"]
                )
            ] = True
            f.close()

        # If the signal is invalid, set it to None
        if np.isnan(self.input_signal).all():
            print("Empty signal, settings everything to None.")
            self.signal = None
            self.input_signal = None
            self.peaks = None
            self.initial_peaks = None

        return self

    def load_file(
        self, physio_file: Union[str, PathLike], json_file: Union[str, PathLike]
    ):
        """Load the physio files."""

        self.recording_start_time = None
        self.recording_end_time = None
        self.sfreq = None
        self.input_columns_names = None
        self.json_file = None
        self.physio_file = None

        # If a path to a physio file is provided, otherwise search in the BIDS folder
        if physio_file:
            self.physio_file = Path(physio_file)
            self.json_file = Path(json_file)
        else:
            physio_files = list(
                Path(
                    self.input_folder,
                    str(self.participant_id),
                    str(self.session),
                    self.modality,
                ).glob(f"*{self.pattern}*.tsv.gz")
            )
            if len(physio_files) == 0:
                self.physio_file, self.json_file = None, None
            elif len(physio_files) > 1:
                self.physio_file, self.json_file = None, None
                print(
                    "More than one recording match the provided string pattern."
                    "Use a more explicit/longer string pattern to find your recording."
                )
            else:
                self.physio_file = physio_files[0]
                json_files = list(
                    Path(
                        self.input_folder,
                        str(self.participant_id),
                        str(self.session),
                        self.modality,
                    ).glob(f"*{self.pattern}*.json")
                )
                if len(json_files) == 0:
                    self.physio_file, self.json_file = None, None
                elif len(json_files) > 1:
                    self.physio_file, self.json_file = None, None
                    print(
                        "More than one JSON file match the provided string pattern. "
                        "Use a more explicit/longer string pattern to find your recording."
                    )
                else:
                    self.json_file = json_files[0]

        if self.json_file is not None:

            # Opening JSON file and extract metadata
            f = open(self.json_file)
            json_data = json.load(f)

            self.sfreq = json_data["SamplingFrequency"]
            self.input_columns_names = json_data["Columns"]

            try:
                self.recording_start_time = json_data["StartTime"]
                self.recording_end_time = json_data["EndTime"]
            except KeyError:
                pass

            f.close()

        if physio_file is None:
            print("No physiological recording found for this participant.")
        elif json_file is None:
            print("No JSON file found for this participant.")
        else:
            print(f"Loading {self.physio_file}")

        return self
