# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from systole.reports import report_oxi
from unittest import TestCase
from systole import import_ppg


class TestReports(TestCase):

    def test_report_oxi(self):
        ppg = import_ppg('1')[0, :]  # Import PPG recording
        report_oxi(ppg, file_name=None)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
