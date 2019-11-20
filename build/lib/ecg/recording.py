# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import time
from ecg.detection import oxi_peaks
from ecg.plotting import plot_oximeter, plot_events, plot_hr


class Oximeter():
    """Recording PPG signal with Nonin pulse oximeter.

    Parameters
    ----------
    serial : pySerial object
        The `serial` instance interfacing with the USB port.
    sfreq : int
        The sampling frequency of the recording. Defautl is 75 Hz.
    add_channels : int
        If int, will create as many additionnal channels. If None, no
        additional channels created.

    Attributes
    ----------
    instant_rr : list
        Time serie of instantaneous heartrate.
    n_channels : int | None
        Number of additional channels.
    recording : list
        Time serie of PPG signal.
    sfreq : int
        Sampling frequnecy. Default value is 75 Hz.
    threshold : list
        The threshold used to detect beat peaks. Will use the average +
        standars deviation.
    times : list
        Time vector (in seconds).
    diff : list
        Records the differential of the PPG signal. Used to detect heartbeat
        peaks.
    peaks : list
        List of 0 and 1. 1 index detected peaks.
    channels : list | dict
        Additional channels to record. Will continuously record `n_channels`
        additional channels in parallel of `recording` with default `0` as
        defalut value.
    serial : PySerial instance
        PySerial object indexing the USB port to read.

    Examples
    --------
    First, you will need to define a `serial` instance, indexing the USB port
    where the Nonin Pulse Oximeter is plugged.

    >>> import serial
    >>> ser = serial.Serial('COM4')

    This instance is then used to create an `Oximeter` instance that will be
    used for the recording.

    >>> from ecgrecording import Oximeter
    >>> oximeter = Oximeter(serial=ser, sfreq=75)

    Use the `setup()` method to initialize the recording. This will find the
    start byte to ensure that all the forthcoming data is in Synch. You should
    not wait more than ~10s between the setup and the recording, otherwise the
    buffer will start to overwrite the data.

    >>> oximeter.setup()

    2 methods are availlable to record PPG signal:

        * The `read()` method will continuously record for certain amount of
        time (specified by the `duration` parameter, in seconds). This is the
        easiest and most robust method, but it is not possible to run
        instructions in the meantime.

        >>> oximeter.read(duration=10)

        * The `readInWaiting()` method will read all the availlable bytes (up
        to 10 seconds of recording). When inserted into a while loop, it allows
        to record PPG signal together with other scripts.

        >>> import time
        >>> tstart = time.time()
        >>> while time.time() - tstart < 10:
        >>>     oximeter.readInWaiting()
        >>>     # Insert code here

    The recorded signal can latter be inspected using the `plot()` method.

    >>> oximeter.plot()
    """
    def __init__(self, serial, sfreq=75, add_channels=None):

        self.serial = serial
        self.lag = 0
        self.sfreq = sfreq
        self.dist = int(self.sfreq * 0.2)

        # Initialize recording with empty lists
        self.instant_rr = []
        self.recording = []
        self.times = []
        self.n_channels = add_channels
        self.threshold = []
        self.diff = []
        self.peaks = []
        if add_channels is not None:
            self.channels = {}
            for i in range(add_channels):
                self.channels['Channel_' + str(i)] = []
        else:
            self.channels = None

    def add_paquet(self, paquet, window=1):
        """Read a portion of data.

        Parameters
        ----------
        paquet : int
            The data to record. Should be an integer between 0 and 240.
        window : int or float
            Length of the window used to compute threshold (seconds). Default
            is `1`.

        Returns
        -------
        Oximeter instance.

        Notes
        -----
        Will automatically calculate the differential, threshold and increment
        additional channles with 0 if provided.
        """

        # Store new data
        self.recording.append(paquet)
        self.peaks.append(0)

        # Add 0 to the additional channels
        if self.channels is not None:
            for ch in self.channels:
                self.channels[ch].append(0)

        # Update times vector
        if not self.times:
            self.times = [0]
        else:
            self.times.append(len(self.times)/self.sfreq)

        # Update threshold
        window = int(window * self.sfreq)
        self.threshold.append((np.mean(self.recording[-window:]) +
                               np.std(self.recording[-window:])))

        # Store new differential if not exist
        if not self.diff:
            self.diff = [0]
        else:
            self.diff.append(self.recording[-1] - self.recording[-2])

            # Is it a threshold crossing value?
            if paquet > self.threshold[-1]:

                # Is the new differential zero or crossing zero?
                if (self.diff[-1] <= 0) & (self.diff[-2] > 0):

                    # Is it far enough from the previous peak (0.2 s)?
                    if not any(self.peaks[-15:]):
                        self.peaks[-1] = 1

        # Update instantaneous heart rate
        if sum(self.peaks) >= 2:
            self.instant_rr.append(
                (np.diff(np.where(self.peaks)[0])[-1]/self.sfreq)*1000)
        else:
            self.instant_rr.append(float('nan'))

        return self

    def check(self, paquet):
        """Check if the provided paquet is correcg

        Parameters
        ----------
        paquet : list
            The paquet to inspecg
        """
        check = False
        if len(paquet) >= 5:
            if paquet[0] == 1:
                if (paquet[1] >= 0) & (paquet[1] <= 255):
                    if (paquet[2] >= 0) & (paquet[2] <= 255):
                        if paquet[3] <= 127:
                            if paquet[4] == sum(paquet[:4]) % 256:
                                check = True

        return check

    def find_peaks(self):
        """Find peaks in recorded signal.

        Returns
        -------
        Oximeter instance. The peaks occurences are stored in the `peaks`
        attribute.
        """

        self.peaks = oxi_peaks(self.recording)

        # R-R intervals (in miliseconds)
        self.rr = (np.diff(self.peaks)/self.sfreq) * 1000

        # Beats per minutes
        self.bpm = 60000/self.rr

        return self

    def plot_events(self, ax=None):
        """Visualize the distribution of events stored in additional channels.

        Return
        ------
        fig, ax : Matplotlib instances.
            The figure and axe instances.
        """
        ax = plot_events(self, ax=ax)

        return ax

    def plot_hr(self, ax=None):
        """Plot heartrate extracted from PPG recording.

        Return
        ------
        fig, ax : Matplotlib instances.
            The figure and axe instances.
        """
        ax = plot_hr(self, ax=ax)

        return ax

    def plot_recording(self, ax=None):
        """Plot recorded signal.

        Return
        ------
        fig, ax : Matplotlib instances.
            The figure and axe instances.
        """
        ax = plot_oximeter(self, ax=ax)

        return ax

    def read(self, duration):
        """Find start byte.

        Parameters
        ----------
        duration : int or float
            Length of the desired recording time.
        """
        tstart = time.time()
        while time.time() - tstart < duration:
            if self.serial.inWaiting() >= 5:
                # Store Oxi level
                paquet = list(self.serial.read(5))
                if self.check(paquet):
                    self.add_paquet(paquet[2])
                else:
                    self.setup()
        return self

    def readInWaiting(self, stop=False):
        """Read in wainting oxi data.

        Parameters
        ----------
        stop : boolean, defalut to `False`
            Whether the recording should continue if an error is detected.
        """
        # Read oxi
        while self.serial.inWaiting() >= 5:
            # Store Oxi level
            paquet = list(self.serial.read(5))
            if self.check(paquet):
                self.add_paquet(paquet[2])
            else:
                if stop:
                    raise ValueError('Synch error')
                else:
                    print('Synch error')
                    while True:
                        self.serial.reset_input_buffer()
                        paquet = list(self.serial.read(5))
                        if self.check(paquet=paquet):
                            break

    def setup(self, read_duration=1):
        """Find start byte and read a portion of signal.

        Parameters
        ----------
        read_duration : int
            Length of signal to record after setup. Default is set to 1 second.

        Notes
        -----
        .. warning:: Will remove previously recorded data.
        """
        # Reset recording instance
        self.__init__(serial=self.serial, add_channels=self.n_channels)
        while True:
            self.serial.reset_input_buffer()
            paquet = list(self.serial.read(5))
            if self.check(paquet=paquet):
                break
        self.read(duration=read_duration)
        return self

    def waitBeat(self):
        """Read Oximeter until a heartbeat is detected.
        """
        while True:
            if self.serial.inWaiting() >= 5:
                # Store Oxi level
                paquet = list(self.serial.read(5))
                if self.check(paquet):
                    self.add_paquet(paquet[2])
                    if any(self.peaks[-2:]):  # Peak found
                        break
                else:
                    print('Synch error')
        return self
