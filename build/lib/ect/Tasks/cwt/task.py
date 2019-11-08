# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from psychopy import visual, event, core
import pandas as pd
import numpy as np
from ect.recording import Oximeter


def run(parameters, stairCase=None, win=None, confidenceRating=True,
        runTutorial=False):
    """Run the entire task based.

    Parameters
    ----------
    parameters : dict
        Task parameters.
    stairCase : Instance of staircase handler.
        If `None`, will use default values:
            data.StairHandler(startVal=30, nTrials=100, nUp=1, nDown=2,
                              stepSizes=[20, 12, 7, 4, 3, 2, 1],
                              stepType='lin', minVal=1, maxVal=100)
    win : psychopy window
        Instance of Psychopy window.
    confidenceRating : boolean
        Whether the trial show include a confidence rating scale.
    tutorial : boolean
        If `True`, will present a tutorial with 10 training trial with feedback
        and 5 trials with confidence rating.

    Returns
    -------
    results_df : Pandas DataFrame
        Dataframe containing behavioral results.
    """
    if win is not None:
        win = win
    if stairCase is None:
        stairCase = parameters['stairCase']

    oxiTraining = Oximeter(serial=parameters['serial'],
                           sfreq=75,
                           add_channels=1)

    # Show tutorial and training trials
    if runTutorial is True:
        tutorial(parameters, win, oxiTraining)

    oxiTask = Oximeter(serial=parameters['serial'],
                       sfreq=75,
                       add_channels=1)
    oxiTask.setup()
    oxiTask.read(duration=1)

    results_df, i = pd.DataFrame([]), 0  # Final DataFrame and trial count
    for i, condition in enumerate(parameters['conditions']):

        if i == 0:
            # Ask the participant to press 'Space' (default) to start
            messageStart = visual.TextStim(win, units='height',
                                           height=0.03,
                                           text='Press space to continue')
            messageStart.autoDraw = True  # Show instructions
            win.flip()
            event.waitKeys(keyList=parameters['startKey'])
            messageStart.autoDraw = False  # Hide instructions
            win.update()

        estimation, estimationRT, confidence, confidenceRT, alpha = trial(
                  parameters, condition, stairCase,
                  feedback=False, win=None, oxi=oxiTask)

        # Store results
        results_df = results_df.append([
                    pd.DataFrame({'Condition': [condition],
                                  'Estimation': [estimation],
                                  'EstimationRT': [estimationRT],
                                  'Confidence': [confidence],
                                  'ConfidenceRT': [confidenceRT],
                                  'Alpha': [alpha],
                                  'nTrials': [i]})], ignore_index=True)
    return results_df


def trial(parameters, condition, stairCase, feedback=False,
          confidenceRating=True, win=None, oxi=None):
    """Run one trial.

    Parameters
    ----------
    parameters : dict
        Task global parameters.
    condition : str | None
        The trial condition ['More', 'Less']
    stairCase : Psychopy stairHandler instance
        The staircase object used for the task.
    feedback : boolean
        If `True`, will provide a feedback to the participant. Only used for
        tutorials.
    confidenceRating : boolean
        If `True`, add a confidence rating scale after the decision.
    win : psychopy window instance | None
        If None, will use the default window provided in the global parameters.
    oxi : instance of Oximeter
        The Oximeter object used to record the PPG level.
    """
    if win is None:
        win = parameters['win']
    if oxi is None:
        oxi = parameters['oxi']

    # Restart the trial until participant provide response on time
    confidence, confidenceRT, accuracy = None, None, None

    # Fixation cross
    fixation = visual.GratingStim(win=win, mask='cross', size=0.1,
                                  pos=[0, 0], sf=0, rgb=-1)
    fixation.draw()
    win.update()
    core.wait(0.25)

    # Random selection of the condition (for training trials)
    if condition is None:
        condition = np.random.choice(['More', 'Less'])

    # Generate actual flicker frequency
    if stairCase is not None:
        if stairCase.intensities:
            alpha = int(stairCase.intensities[-1])
        else:
            alpha = int(stairCase.startVal)
        if condition == 'Less':
            alpha = -alpha
    else:
        if condition == 'More':
            alpha = 20
        elif condition == 'Less':
            alpha = -20

    # Load face stimulus image
    faceStim = visual.ImageStim(
            win=parameters['win'], units='height',
            image=parameters['pathStimuli'] + 'angry_happy0' +
            str(50 + alpha).zfill(2) + '.jpg')

    # Adjust the size of the image
    faceStim.size *= 0.3

    # Text component
    message = visual.TextStim(win, units='height', height=0.03,
                              pos=(0.0, -0.2), text=parameters['estimation'])
    faceStim.draw()
    message.draw()
    win.flip()

    # Start trigger
    oxi.readInWaiting()
    oxi.channels['Channel_0'][-1] = 1

    ###########
    # Responses
    ###########

    clock = core.Clock()
    responseKey = event.waitKeys(keyList=parameters['allowedKeys'],
                                 maxWait=parameters['respMax'],
                                 timeStamped=clock)
    win.flip()

    # End trigger
    oxi.readInWaiting()
    oxi.channels['Channel_0'][-1] = 1

    # Check for response provided by the participant
    if not responseKey:
        estimation, estimationRT = None, None
        # Record participant response (+/-)
        message = visual.TextStim(win, units='height', height=0.03,
                                  text='Too late')
        message.draw()
        win.flip()
        core.wait(1)
    else:
        estimation = responseKey[0][0]
        estimationRT = responseKey[0][1]

        # Is the answer Correct? Update the staircase model
        if (estimation == 'up') & (condition == 'More'):
            if stairCase is not None:
                stairCase.addResponse(1)
            accuracy = 1
        elif (estimation == 'down') & (condition == 'Less'):
            if stairCase is not None:
                stairCase.addResponse(1)
            accuracy = 1
        else:
            if stairCase is not None:
                stairCase.addResponse(0)
            accuracy = 0

        # Read oximeter
        oxi.readInWaiting()

        # Feedback
        if feedback is True:
            if accuracy == 0:
                acc = visual.TextStim(win, units='height',
                                      height=0.3, color=(1.0, 0.0, 0.0),
                                      text='False')
                acc.draw()
                win.flip()
                core.wait(2)
            elif accuracy == 1:
                acc = visual.TextStim(win, units='height',
                                      height=0.3, color=(0.0, 1.0, 0.0),
                                      text='Correct')
                acc.draw()
                win.flip()
                core.wait(2)
        else:

            ###################
            # Confidence rating
            ###################

            # Record participant confidence
            if confidenceRating is True:
                markerStart = np.random.choice(
                                np.arange(parameters['confScale'][0],
                                          parameters['confScale'][1]))
                ratingScale = visual.RatingScale(
                                 win,
                                 low=parameters['confScale'][0],
                                 high=parameters['confScale'][1],
                                 noMouse=True,
                                 labels=parameters['labelsRating'],
                                 acceptKeys='down',
                                 markerStart=markerStart)

                message = visual.TextStim(
                            win, units='height', height=0.03,
                            text=parameters['confidenceText'])

                # Wait for response
                clock = core.Clock()
                while clock.getTime() < parameters['maxRatingTime']:
                    if not ratingScale.noResponse:
                        ratingScale.markerColor = (0, 0, 1)
                        if clock.getTime() > parameters['minRatingTime']:
                            break
                    ratingScale.draw()
                    message.draw()
                    win.flip()

                win.flip()
                confidence = ratingScale.getRating()
                confidenceRT = ratingScale.getRT()

            # Hide instructions
            message.autoDraw = False
            win.flip()

    return estimation, estimationRT, confidence, confidenceRT, alpha


def tutorial(parameters, win=None, oxi=None):
    """Run tutorial.

    Parameters
    ----------

    """

    if win is None:
        win = parameters['win']
    if oxi is None:
        oxi = parameters['oxi']
