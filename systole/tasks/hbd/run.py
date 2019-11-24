# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import serial
from ecg.tasks.hbd import task, parameters
from ecg.recording import Oximeter


parameters = parameters.getParameters()

# Set task path
cwd = parameters['path']

# Open seral port for Oximeter
oxi = Oximeter(serial=parameters['serial'], sfreq=75)
oxi.setup()
oxi.read(duration=2)

task.run(parameters)  # Run all trials
