# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import sys
import socket
import threading
import unittest
from unittest import TestCase

import numpy as np

from systole import serialSim
from systole.recording import BrainVisionExG, Oximeter


class TestRecording(TestCase):
    def test_oximeter(self):
        serial = serialSim()
        oxi = Oximeter(serial=serial, add_channels=1)
        oxi.setup()
        serial.ppg = serial.ppg[-2:]  # To the end of recording
        oxi.read(10)
        oxi.find_peaks()

        oxi.save("test")
        assert os.path.exists("test.npy")
        os.remove("test.npy")

        oxi.save("test.txt")
        assert os.path.exists("test.txt")
        os.remove("test.txt")

        # Simulate events in recording
        for idx in np.random.choice(len(oxi.recording), 5):
            oxi.channels["Channel_0"][idx] = 1
        oxi.plot_events()

        oxi.plot_raw(clipping=False)

        oxi.serial.ppg = [1000]  # Insert error in recording
        with self.assertRaises(ValueError):
            oxi.readInWaiting(stop=True)

        serial = serialSim()
        oxi.serial.ppg = [1000, -1, 1000, -1, -1]  # Insert error in recording
        oxi.readInWaiting(stop=False)

        oxi.serial.ppg = [1000, -1, 1000, -1, -1]  # Insert error in recording
        oxi.waitBeat()
        oxi.find_peaks()

        oxi.peaks = []
        oxi.instant_rr = []
        oxi.times = []
        oxi.threshold = []

    def test_BrainVisionExG(self):
        data1 = b"\x8eEXC\x96\xc9\x86L\xafJ\x98\xbb\xf6\xc9\x14P\x8d\x00\x00\x00\x01\x00\x00\x00"
        data2 = b"\x08\x00\x00\x00\x00\x00\x00\x00\x00@\x8f@\x00\x00\x00\x00\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\xe0?33333\x13c@33333\x13c@EGG1\x00EGG2\x00EGG3\x00EGG4\x00EGG5\x00EGG6\x00RESP\x00PLETH\x00"
        data3 = b"\x8eEXC\x96\xc9\x86L\xafJ\x98\xbb\xf6\xc9\x14P\xa4\x02\x00\x00\x04\x00\x00\x00"
        data4 = b"\xad)\x01\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x04.F\x00\xc8\x9bE\x00\xb0HE\x00|\x05F\x00\x00IC\x00\x00!\xc3\x00`}F\x00@mE\x00\x90$F\x00\xe8\x93E\x00\xc0pE\x00\xb0\xbbE\x00\x00\xe8\xc2\x00\x00,\xc3\x00`}F\x00 mE\x00\xf8\xfbE\x00`\xdbD\x00@\tD\x00@\xa4E\x00\x00\x00\xc2\x00\x00\x1b\xc3\x00\\}F\x00\x00mE\x00\\]F\x000lE\x00\x18\x81E\x00\x08\x0bF\x00\x00H\xc3\x00\x00,\xc3\x00`}F\x00\x00mE\x00\x98\xeaE\x00\x88\xd1\xc5\x00`\xcdD\x00|Q\xc6\x00\x003\xc4\x00\x00\x00\xc3\x00`}F\x00\x00mE\x00\xf4\x18F\x00\x90L\xc5\x00\xf0\x15E\x00@\xe6\xc5\x00\x80T\xc4\x00\x00(\xc3\x00h}F\x00 mE\x00P\xfeE\x00\xd0\xcd\xc5\x00\x80N\xc5\x00\x80\xc0\xc4\x00\x80W\xc4\x00\x00*\xc3\x00h}F\x00@mE\x00\x90\xdcE\x00@\x07\xc6\x000\xea\xc5\x00\xf8\xa3E\x00\xc0l\xc4\x00\x00?\xc3\x00`}F\x00@mE\x00\xd8\xf0E\x00\x00\xe9\xc5\x00\xe8\xb2\xc5\x00\xd0eE\x00\x80f\xc4\x00\x00f\xc3\x00`}F\x00\x00mE\x00\xbc\nF\x00p\x99\xc5\x00\x00\xed\xc3\x00\xd0}\xc5\x00@S\xc4\x00\x00g\xc3\x00`}F\x00\x00mE\x00\xdc\x02F\x00\x80\xb2\xc5\x00`\x02\xc5\x00@\xfd\xc4\x00\x00Y\xc4\x00\x00s\xc3\x00h}F\x00\x00mE\x00\xe8\x04F\x00X\xac\xc5\x00 \xcf\xc4\x00`C\xc5\x00\x00U\xc4\x00\x00t\xc3\x00`}F\x00 mE\x00x\x01F\x00H\xbe\xc5\x00\x10%\xc5\x00`\xf1\xc4\x00\x80V\xc4\x00\x00q\xc3\x00`}F\x00 mE\x00\xb8\xd0E\x00P\x15\xc6\x00\xf0\x12\xc6\x00`\xfdE\x00\x80m\xc4\x00\x00^\xc3\x00`}F\x00@mE\x00|\x05F\x00H\xbf\xc5\x00\xf0%\xc5\x00@\x84\xc4\x00@S\xc4\x00\x00,\xc3\x00`}F\x00\x10mE\x008\x0fF\x00(\x8f\xc5\x00\x00\x85\xc3\x00\xd0\x8a\xc5\x00\x80K\xc4\x00\x00#\xc3\x00\\}F\x00\x00mE\x00\x84\x85F\x00\x90HF\x00\x80PE\x00@RD\x00@T\xc4\x00\x00\x13\xc3\x00`}F\x00@lE\x00x\x8eE\x00(\x8dE\x00\x80\xb0\xc5\x00\xe0\xbbE\x00\xc0Q\xc4\x00\x00@\xc3\x00`}F\x00\xd0jE\x00@\xc8D\x00\xe8b\xc6\x000\xdb\xc5\x00\x90\xc7\xc5\x00\xc0Q\xc4\x00\x00l\xc3\x00`}F\x00piE\x00t*F\x00\xd0)\xc5\x00\x00\x02C\x00\xc0\x1d\xc5\x00\xc0W\xc4\x00\x00a\xc3\x00`}F\x00\x00hE"

        def simulateEXG():
            # Run a server simulating BrainVisionExG amplifier connection
            server = socket.socket()
            server.bind(("127.0.0.1", 51244))
            server.listen(0)
            conn, addr = server.accept()
            with conn:
                conn.send(data1)
                conn.send(data2)
                conn.send(data3)
                conn.send(data4)

        # Start fake server in background thread
        server_thread = threading.Thread(target=simulateEXG)
        server_thread.start()

        # Test the clients basic connection and disconnection
        try:
            recorder = BrainVisionExG("127.0.0.1", port=51244, sfreq=1000)
            data = recorder.read(0.0001)

            assert list(data.keys()) == [
                "EGG1",
                "EGG2",
                "EGG3",
                "EGG4",
                "EGG5",
                "EGG6",
                "RESP",
                "PLETH",
            ]
            assert all([data[k].shape[0] == 20 for k in list(data.keys())])
        except (
            ConnectionRefusedError
        ):  # Add exception for GitHub actions that sometimes fail
            pass
        
        server_thread.join()

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
