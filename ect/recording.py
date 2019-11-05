# Function for online analysis of ECG and Oximetric data

import numpy as np
import time
from cardioception.detection import oxi_peaks


class Oximeter():
    """Recording with Nonin pulse oximeter.

    Parameters
    ----------
    serial : pySerial object
        Where to read the oximeter recording.
    sfreq : int
        The sampling frequency of the recording.

    Attributes
    ----------

    Examples
    --------
    Record 10s.
    >>> import serial
    >>> from cardioception.recording import Oximeter

    >>> ser = serial.Serial('COM4',
    >>>                     baudrate=9600,
    >>>                     timeout=1/75,
    >>>                     stopbits=1,
    >>>                     parity=serial.PARITY_NONE)

    >>> oximeter = Oximeter(serial=ser, sfreq=75)
    >>> oximeter.setup()
    >>> oximeter.read(duration=10)
    """
    def __init__(self, serial=None, sfreq=75):

        self.instant_rr = []
        self.lag = 0
        self.recording = []
        self.sfreq = sfreq
        self.dist = int(self.sfreq * 0.2)
        self.triggers = []
        self.times = []
        self.diff = []
        self.peaks = []
        self.stim = []
        if serial is not None:
            self.serial = serial

    def add_paquet(self, paquet, window=1):
        """Read a portion of data and return a trigger when the main peak is
        found in the oxi data.

        Parameters
        ----------
        paquet : int
            The data to be read.
        window : int or float
            Length of the window used to compute threshold (seconds). Default
            is `1`.

        Returns
        -------
        Oximeter instance.

        Notes
        -----
        If no differential, will use available recording to create one.
        """

        # Store new data
        self.recording.append(paquet)
        self.stim.append(0)
        if not self.times:
            self.times = [0]
        else:
            self.times.append(len(self.times)/self.sfreq)

        # Update threshold
        window = int(window * self.sfreq)
        self.thr = (np.mean(self.recording[-window:]) +
                    np.std(self.recording[-window:]))

        # Store new differential if not exist
        if not self.diff:
            self.diff = np.diff(self.recording).tolist()
            self.triggers = [0] * len(self.recording)
        else:
            self.diff.append(self.recording[-1] - self.recording[-2])

        # Is it a threshold crossing value?
        if paquet > self.thr:

            # Is the new differential zero or crossing zero?
            if ((self.diff[-1] == 0) |
               ((self.diff[-1] > 0) != (self.diff[-2] > 0))):

                # Was the previous differential positive?
                if self.diff[-2] > 0:

                    # Is it far enough from previous peak (0.2 s)?
                    if self.lag > self.dist:
                        self.triggers.append(1)
                        self.lag = -1

        # If event was detected
        if self.lag >= 0:
            self.triggers.append(0)
        self.lag += 1

        # Update instantaneous heart rate
        if sum(self.triggers) > 2:
            self.instant_rr.append(
                (np.diff(np.where(self.triggers)[0])[-1]/self.sfreq)*1000)
        else:
            self.instant_rr.append(0)

        return self

    def check(self, paquet):
        """Check if the provided paquet is correct.

        Parameters
        ----------
        paquet : list
            The paquet to verify.
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
                    self.triggers[-1] = -1
                    while True:
                        self.serial.reset_input_buffer()
                        paquet = list(self.serial.read(5))
                        if self.check(paquet=paquet):
                            break

    def setup(self):
        """Find start byte.

        Parameters
        ----------
        serial : PySerial object
            The USB port to read

        Notes
        -----
        .. warning:: Will remove previously recorded data.
        """
        self.__init__(serial=None)  # Restart recording
        while True:
            self.serial.reset_input_buffer()
            paquet = list(self.serial.read(5))
            if self.check(paquet=paquet):
                break
        return self

    def waitBeat(self):
        """Read Oximeter until a beat is detected.
        """
        while True:
            if self.serial.inWaiting() >= 5:
                # Store Oxi level
                paquet = list(self.serial.read(5))
                if self.check(paquet):
                    self.add_paquet(paquet[2])
                    if any(self.triggers[-2:]):  # Peak found
                        self.stim[-1] = 1
                        break
                else:
                    print('Synch error')
        return self
