# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import serial
import numpy as np
from psychopy import visual, data


def getParameters(subjectID, subjectNumber):
    """High level function to define all hard coded experiment constants.

    Attributes
    ----------
    startKey: str
        Key used to launch the task or go to next steps.
    nTrials : int
        Total number of trials.
    allowedKeys : list of str
        Allowed keys for estimation response.
    respMax : int
        Maximum number of seconds to provide the estimation.
    confScale : list of int
        Range of the confidence rating scale.
    labelRating : list of str
        Labels of the confidence rating scale.
    minRatingTime : float
        The minimum time before a rating can be provided during the confidence
        rating (in seconds).
    maxRatingTime : float
        The maximum time for a confidence rating (in seconds).
    subjectID : str
        The subject identifiant.
    subjectNumber : int
        The subject number.
    condition : 1d-array
        Array of 0s and 1s encoding the conditions (1 : Higher, 0 : Lower). The
        length of the array is defined by `parameters['nTrials']`. If
        `parameters['nTrials']` is odd, will use `parameters['nTrials']` - 1
        to enseure an equal nuber of Higher and Lower conditions.
    path : str
        The task working directory.
    results : str
        The result directory.
    pathStimuli : str
        The directory containing the face stimuli.
    win : instance of Psychopy window
        The window where to run the task.
    serial : instance of PySerial
        The USB port used to record PPG signal.
    estimation : str
        Text used during the estimation phase.

    Returns
    -------
    parameters : dict
        Dictionnary containing the task parameters.
    """
    parameters = {}
    parameters['startKey'] = 'space'
    parameters['nTrials'] = 50
    parameters['allowedKeys'] = ['up', 'down']
    parameters['respMax'] = 8
    parameters['confScale'] = [1, 7]
    parameters['labelsRating'] = ['Guess', 'Certain'],
    parameters['minRatingTime'] = 1
    parameters['maxRatingTime'] = 3

    # Set default path /Results/ 'Subject ID' /
    parameters['subjectID'] = subjectID
    parameters['subjectNumber'] = subjectNumber

    # Create randomized condition vector
    parameters['conditions'] = np.hstack(
            [np.array(['More'] * round(parameters['nTrials']/2)),
             np.array(['Less'] * round(parameters['nTrials']/2))])
    np.random.shuffle(parameters['conditions'])  # Shuffle vector

    parameters['path'] = os.getcwd()
    parameters['results'] = parameters['path'] + '/Results/' + subjectID + '/'
    # Create Results directory of not already exists
    if not os.path.exists(parameters['results']):
        os.makedirs(parameters['results'])

    # Folder containg face stimuli
    parameters['pathStimuli'] = 'C:/Users/au646069/Google Drive/ECG_root/Code/PythonToolboxes/CWT/1stim-face-discrim/stimuli/'

    # Open window
    parameters['win'] = visual.Window(screen=0,
                                      fullscr=True,
                                      units='height')
    # Serial port
    parameters['serial'] = serial.Serial('COM4')

    parameters['stairCase'] = data.StairHandler(
                        startVal=30, nTrials=parameters['nTrials'], nUp=1,
                        nDown=2, stepSizes=[20, 12, 7, 4, 3, 2, 1],
                        stepType='lin', minVal=1, maxVal=50)

    parameters['estimation'] = 'Sad or Happy?'
    parameters['confidenceText'] = 'How confident are you?'

    return parameters
