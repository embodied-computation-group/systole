# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from psychopy import gui
from ecg.Tasks.cwt import task, parameters

# Create a GUI and store subject ID
g = gui.Dlg()
g.addField("Subject ID:")
g.addField("Subject Number:")
g.show()

# Set global task parameters here
parameters = parameters.getParameters(subjectID=g.data[0],
                                      subjectNumber=g.data[1])

# Run task
results_df = task.run(parameters, win=parameters['win'], confidenceRating=True,
                      runTutorial=False)
# Save results
results_df.to_csv(parameters['results'] + g.data[0] + '.txt')
parameters['win'].close()
