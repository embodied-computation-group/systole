# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import unittest
from unittest import TestCase

import papermill as pm


class TestNotebooks(TestCase):
    def test_notebooks(self):
        """Test tutorial notebooks"""

        # Load tutorial notebooks from the GitHub repository
        url = "./source/notebooks/"
        for nb in [
            "1-PhysiologicalSignals.ipynb",
            "2-DetectingCycles.ipynb",
            "3-DetectingAndCorrectingArtefacts.ipynb",
            "4-HeartRateVariability.ipynb",
            "5-InstantaneousHeartRate.ipynb",
        ]:
            pm.execute_notebook(url + nb, "./tmp.ipynb")
        os.remove("./tmp.ipynb")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
