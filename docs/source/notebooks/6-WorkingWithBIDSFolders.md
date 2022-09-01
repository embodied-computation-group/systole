---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

```{code-cell} ipython3
:tags: [hide-input]

%%capture
import sys
if 'google.colab' in sys.modules:
    ! pip install systole
```

# Working with BIDS folders
Starting in version `0.2.3`, Systole provides tools to interact efficiently with large datasets containing physiological recordings. Most of the functionalities interface with folders structured following the [BIDS standards](https://bids-specification.readthedocs.io/en/stable/) and this is the format we recommend using if you are following this tutorial.

Under BIDS standards, physiological recordings, sometimes associated with behavioural tasks or neural recordings, are stored with a filename ending with `*_physio.tsv.gz` and are always accompanied with sidecar a `*_physio.json` file containing metadata like the recording modality or the sampling frequency. Accessing both the times series and its accompanying metadata will help Systole automate the preprocessing by finding the correct parameters for peaks detection and reports.

Once you have organized your folder, you should have a structure resembling this one:

```
└─ BIDS/
   ├─ sub-0001/
   │  └─ ses-session1/
   │     └─ beh/
   │        ├─ sub-0001_ses_session1_task-mytask_physio.tsv.gz
   │        └─ sub-0001_ses_session1_task-mytask_physio.json
   │
   ├─ sub-0002/
   ├─ sub-0003/
   └─ ... 
```

Here, we have physiological recordings associated with a behavioural task for `n` participants in the folder.

+++

## Signal preprocessing and creation of subject and group-level reports

+++

The first step will be to extract peaks from the physiological signal and generate individual reports for each participant, so we can visualize the signal quality and peaks detection.

### Preprocessing the physiological recording from one participant

The py:func:`systole.reports` sub-module contains tools to directly interact with BIDS formatted folders, preprocess and save individual reports in a BIDS consistent way. Those functionalities are built on the top of the py:func:`systole.reports.subject_level_report` function. This function will simply take a signal as input and will save as output the preprocessed signal with peaks detection (`_physio.tsv.gz` with the `_physio.json`), an `.html` reports adapted to the kind of signal that was provided, and a `features.tsv` file containing heart rate or respiratory rate variability features.

For example, running the following code:


```python
from systole import import_dataset1
from systole.reports import subject_level_report

ecg = import_dataset1(modalities=["ECG"]).ecg.to_numpy()

subject_level_report(
    participant_id="participant_test",
    pattern="task_test",
    result_folder="./",
    session="session_test",
    ecg=ecg,
    ecg_sfreq=1000,
)
```

+++

will save these four new files in the file folder.
1. The `.html` file is a standalone document that can be visualized in the browser.
2. The `features.tsv` contains heart rate and/or respiration rate variability metrics.
3. The `_physio.tsv.gz` and the `_physio.json` files contain the preprocessed signal with new columns `peaks` for peaks detection.

+++

### Preprocessing the entire BIDS folder

The previous function call can be automated for each participant and each file of a given BIDS folder and to extract the physiological features using the information provided in the `json` metadata automatically. This can be done using the py:func:`systole.reports.wrapper` function, or directly from the command line. For example, the following command:

```bash
systole --bids_folder="/mnt/scratch/BIDS/" \
        --patterns="task-mytask" \
        --modality="beh" \
        --n_jobs=10 \
        --overwrite=True
```

will preprocess the data for all participants with a physiological recording in the session `ses-session1` (default), for the behavioural modality (`beh`) and the task `mytask`. We set `n_jobs=10`, meaning that we will run 40 processes in parallel, and `overwrite=True` to overwrite previous data with the same ID in the derivative folder.

+++

```{note}
When setting `overwrite=True`, only the preprocessed derivatives can be overwritten, but not the edited files located in `BIDS/systole/derivatives/corrected/*`. This means that it is possible to re-run the preprocessing event after working on the manual artefacts edition (see below).
```

+++

Once the preprocessing is completed, the structure of the BIDS repository should now include a new `systole` folder in the derivatives:

```
└─ BIDS/
   ├─ derivatives/
   │  └─ systole/
   │     └─ sub-0001/
   │         └─ ses-session1/
   │            └─ beh/
   │               ├─ sub-0001_ses_session1_task-mytask_features.tsv
   │               ├─ sub-0001_ses_session1_task-mytask_report.html
   │               ├─ sub-0001_ses_session1_task-mytask_physio.tsv.gz
   │               └─ sub-0001_ses_session1_task-mytask_physio.json
   ├─ sub-0001/
   │  └─ ses-session1/
   │     └─ beh/
   │        ├─ sub-0001_ses_session1_task-mytask_physio.tsv.gz
   │        └─ sub-0001_ses_session1_task-mytask_physio.json
   │
   ├─ sub-0002/
   ├─ sub-0003/
   └─ ... 
```

+++

(viewer)=
## Manual edition of peaks vector and labelling bad segments using the Viewer

While we hope that the peaks detection function used by [Systole](https://embodied-computation-group.github.io/systole/#) is sufficiently robust to extract peak vectors without errors for most of the uses cases, you might still encounter noisy or invalid recording that you will want to manually inspect and sometimes edit.

The py:func:`systole.viewer` sub-module contains handy functions to interactively inspect and edit the preprocessed folders and save the modified information accordingly.

```{code-cell} ipython3
from IPython.display import display
from systole.viewer import Viewer

%matplotlib ipympl
```

```{code-cell} ipython3
view = Viewer()
```

```{code-cell} ipython3
display(view.box, view.output);
```

* Left mouse button : remove all the peaks in the selected interval.
* Right mouse button : add one new peaks where the signal local maximum is found.

+++

After peaks correction, a new `corrected` subfolder will be appended to the systole derivatives:

```
└─ BIDS/
   ├─ derivatives/
   │  └─ systole/
   │     ├─ corrected/
   │        └─ sub-0001/
   │           └─ ses-session1/
   │              └─ beh/
   │                 └─ sub-0001_ses_session1_task-mytask_physio.json
   │     └─ sub-0001/
   │         └─ ses-session1/
   │            └─ beh/
   │               ├─ sub-0001_ses_session1_task-mytask_features.tsv
   │               ├─ sub-0001_ses_session1_task-mytask_report.html
   │               ├─ sub-0001_ses_session1_task-mytask_physio.tsv.gz
   │               └─ sub-0001_ses_session1_task-mytask_physio.json
   ├─ sub-0001/
   │  └─ ses-session1/
   │     └─ beh/
   │        ├─ sub-0001_ses_session1_task-mytask_physio.tsv.gz
   │        └─ sub-0001_ses_session1_task-mytask_physio.json
   │
   ├─ sub-0002/
   ├─ sub-0003/
   └─ ... 
```

The logs of artefacts correction will be locted in the new `_physio.json` file and contains all information about bad segments labelling, peaks deletion and peaks insertion.

* `add_idx` logs the position of new peaks, in sample idex.
* `remove_idx` logs the position of removed peaks, in sample idex.
* The `bads` field logs the starting and ending points of bad segments in the recoring.


```python
{
    "ppg": {
        "add_idx": [
            632675
        ],
        "remove_idx": [
            630427,
        ],
        "bads": {
            "start": null,
            "end": null
        }
    }
}
```
