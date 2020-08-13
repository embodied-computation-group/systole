# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import unittest
import threading
import matplotlib
import socket
from struct import pack
import numpy as np
from unittest import TestCase
from systole import serialSim
from systole.recording import Oximeter, BrainVisionExG


class TestRecording(TestCase):

    def test_oximeter(self):

        serial = serialSim()
        oxi = Oximeter(serial=serial, add_channels=1)
        oxi.setup()
        serial.ppg = serial.ppg[-2:]  # To the end of recording
        oxi.read(10)
        oxi.find_peaks()
        # Simulate events in recording
        for idx in np.random.choice(len(oxi.recording), 5):
            oxi.channels['Channel_0'][idx] = 1
        ax = oxi.plot_events()
        assert isinstance(ax, matplotlib.axes.Axes)

        ax = oxi.plot_hr()
        assert isinstance(ax, matplotlib.axes.Axes)

        ax = oxi.plot_recording()
        assert isinstance(ax, matplotlib.axes.Axes)

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

        oxi.save('test')
        assert os.path.exists("test.npy")
        os.remove("test.npy")
        oxi = Oximeter(serial=serial)

    def test_BrainVisionExG(self):

        def run_fake_server():
            # Run a server to listen for a connection and then close it
            server_sock = socket.socket()
            server_sock.bind(('127.0.0.1', 51244))
            server_sock.listen(0)
            conn, addr = server_sock.accept()
            conn.send('rawdata'.encode())
            server_sock.close()

        # Start fake server in background thread
        server_thread = threading.Thread(target=run_fake_server)
        server_thread.start()

        # Test the clients basic connection and disconnection
        game_client = BrainVisionExG('127.0.0.1', 51244)
        game_client.RecvData(1)
        game_client.SplitString('11'.encode() + '\x00'.encode())
        game_client.GetProperties(pack('<Ld', 1, 1) + pack('<d', 1))
        rawdata = pack('<LLL', 1, 1, 1) + pack('<f', 1) + pack('<L', 32)\
            + pack('<LLl', 1, 1, 1) + 'Channel'.encode() + '\x00'.encode()\
            + '1'.encode() + '\x00'.encode()
        channelCount = 1
        game_client.GetData(rawdata, channelCount)
        #game_client.read(1)
        game_client.close()

        # Ensure server thread ends
        server_thread.join()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
