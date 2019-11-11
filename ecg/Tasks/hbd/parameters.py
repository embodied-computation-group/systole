# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import serial
from psychopy import sound, visual

def getParameters(subjectID, subjectNumber):
    """High level function to define all hard coded experiment constants.

    Attributes
    ----------

    """
    parameters = {'startKey': 'space'}

    # Set default path /Results/ 'Subject ID' /
    parameters['subjectID'] = subjectID
    parameters['subjectNumber'] = subjectNumber

    parameters['path'] = os.getcwd()
    parameters['results'] = parameters['path'] + '/Results/' + subjectID + '/'
    # Create Results directory of not already exists
    if not os.path.exists(parameters['results']):
        os.makedirs(parameters['results'])

    # Set note played at trial start
    parameters['note'] = sound.backend_sounddevice.SoundDeviceSound(secs=0.5)

    # Open window
    parameters['win'] = visual.Window(screen=parameters['screenNb'],
                                      fullscr=True,
                                      units='height')
    # Serial port
    # Create the recording instance
    parameters['serial'] = serial.Serial('COM4',
                                         baudrate=9600,
                                         timeout=1/75,
                                         stopbits=1,
                                         parity=serial.PARITY_NONE)
    return parameters
