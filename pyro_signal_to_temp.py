def getcalibration(pyroname):
    import platform
    import sys
    import pandas as pd
    import numpy as np
    import os

    home = os.path.expanduser('~')

    pyroname = pyroname.upper()
    directory = ''
    filename = ''

    # specify directory and filename to get the calibration data from

    if platform.system() == 'Windows':
        directory = 'I:\\THERMISCHE ANALYSE\\Messmethoden\\PISA\\ '\
                    'PISA_Labor\\Kalibrationsmessungen'
        # changed directory for testing on Linux Laptop
        # which is not connected to ZAE
        filename = directory + '\\_Kalibration_Pyrometer.txt'
    elif platform.system() == 'Linux':
        # use directory to set path where "_Kalibration_Pyrometer.txt"
        # can be found, if same folder -> directory = ''
        # directory = '/home/mgessner/PythonCode/'
        directory = home + '/P3/'
        filename = directory + '_Kalibration_Pyrometer.txt'
    else:
        sys.exit('It is recommened to use Linux! (or Windows if you have to)')
    # read the data using pandas
    data = pd.read_csv(filename, delimiter='\t', decimal='.', engine='c',
                       usecols=[0, 1, 2, 3])

    value = np.array(data['value/mV'][data['pyrometer'] == pyroname])[-1]
    error = np.array(data['error/mV'][data['pyrometer'] == pyroname])[-1]

    value /= 1000   # change from mV to V
    error /= 1000

    # if pyroname in data['pyrometer']:
    #     print(data['value/mV'])

    return (value, error)


def pyrotemp(rawdata, pyrometer):
    import numpy as np

    C_2 = 0.014388  # m*K      constant
    T_90_Cu = 1357.77  # K     melting plateau of pure copper
    pyrometer = pyrometer.upper()

    rawdata = abs(rawdata)

    # each pyrometer has its own effective wavelenth
    # which is needed for the calculation later on
    if pyrometer == 'PVM':
        Lambda = 925.2  # nm
    elif pyrometer == 'PV':
        Lambda = 903.1  # nm
    elif pyrometer == 'PUV':
        Lambda = 904.4  # nm
    else:
        Lambda = 900.0  # nm

    # get value at the melding plateau for choosen
    # pyromter from calibration file
    # calibration data is in mV measured data in V so "/ 1000"
    (U_T90_Cu, sdU_T90_Cu) = getcalibration(pyrometer)

    Lambda = Lambda * 1e-9  # m
    # convert wavelenth from nm to m

    q = rawdata / U_T90_Cu   # scalar division to get simplified variable q

    # calculate real temperature from comparison with calibrated values
    # and convert it to numpy array type: float64
    T_90 = (C_2 / Lambda) / np.log(np.abs(((np.exp(C_2 / (Lambda * T_90_Cu)) +
                                            q - 1) / q)))

    return T_90


def pyrotemp2(rawdata, pyrometer, f_esd):
    import numpy as np

    C_2 = 0.014388  # m*K      constant
    T_90_Cu = 1357.77  # K     melting plateau of pure copper
    pyrometer = pyrometer.upper()

    rawdata = abs(rawdata)

    # each pyrometer has its own effective wavelenth
    # which is needed for the calculation later on
    if pyrometer == 'PVM':
        Lambda = 925.2  # nm
    elif pyrometer == 'PV':
        Lambda = 903.1  # nm
    elif pyrometer == 'PUV':
        Lambda = 904.4  # nm
    else:
        Lambda = 900.0  # nm

    # get value at the melding plateau for choosen
    # pyromter from calibration file
    # calibration data is in mV measured data in V so "/ 1000"
    (U_T90_Cu, sdU_T90_Cu) = getcalibration(pyrometer)

    Lambda = Lambda * 1e-9  # m
    # convert wavelenth from nm to m

    # scalar division to get simplified variable q
    q = rawdata / (U_T90_Cu * f_esd(rawdata))

    # calculate real temperature from comparison with calibrated values
    # and convert it to numpy array type: float64
    T_90 = (C_2 / Lambda) / np.log(np.abs(((np.exp(C_2 / (Lambda * T_90_Cu)) +
                                            q - 1) / q)))

    return T_90

# name = getcalibration('PV')
# print(name)
