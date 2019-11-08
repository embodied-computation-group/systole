# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import serial
from cardioception.HeartBeatDiscrimination.task import trial, run
from cardioception.HeartBeatDiscrimination.parameters import getParameters
from ect.recording import Oximeter


parameters = getParameters()

# Set task path
cwd = parameters['path']

# Open seral port for Oximeter
oxi = Oximeter(serial=parameters['serial'], sfreq=75)
oxi.setup()
oxi.read(duration=2)

run(parameters)  # Run all trials

# Save result df
hbct.results_df.to_csv(cwd + '/Results/' + hbct.subject + '.txt')

hbct.win.close()
