# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from ecgTasks.ConfidenceWeightingTask.parameters import getParameters
from ecgTasks.ConfidenceWeightingTask.task import run
from psychopy import gui

# Create a GUI and store subject ID
g = gui.Dlg()
g.addField("Subject ID:")
g.show()
subject = g.data[0]

# Set global task parameters here
parameters = getParameters(subject)

# Run task
results_df = run(parameters, win=parameters['win'], confidenceRating=True,
                 runTutorial=True)
# Save results
results_df.to_csv(parameters['results'] + subject + '.txt')
parameters['win'].close()
