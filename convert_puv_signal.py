def convert_puv_signal(bbhole, scan_volt):
    import numpy as np
    # import os
    import matplotlib.pyplot as plt
    # from tkinter import Tk as tk
    # from tkinter import filedialog as fd
    # from tkinter.filedialog import askopenfilename as aofn
    # from scipy.interpolate import interp1d as int1d
    # from intensity_to_temp_simple import intensity_to_temp_simple
    # from intensity_to_temp2_simple import intensity_to_temp2_simple
    from pyro_signal_to_temp import getcalibration
    # from pyro_signal_to_temp import pyrotemp  # , pyrotemp2
    # from load_ref_simple import load_ref
    # from scipy.signal import savgol_filter as sgf
    # from save_to import save_to
    import warnings
    # import ipdb
    warnings.simplefilter('ignore', RuntimeWarning)
    plt.ion()

    # get calibration data from file
    (U_T90_Cu, sdU_T90_Cu) = getcalibration('PUV')

    # get half shape of profiles
    m_scan = np.int(np.shape(scan_volt)[1] / 2)
    m_bbhole = np.int(np.shape(bbhole)[1] / 2)

    # find the maximum of profiles
    # scan_volt_c_max = np.argmax(scan_volt[:, m_scan, 1])

    # get center of black body hole "profiles"
    bbhole_c = bbhole[:, m_bbhole, 1]

    # find maximum place of  bbhole temperature to start fitting from
    arg_temp_max = np.argmax(bbhole_c)

    bbhole_c_argmax = bbhole[arg_temp_max:, m_bbhole, 1]

    # get the actual value in the middle
    scan_volt_c = scan_volt[:, m_scan, 1]

    scan_volt_c_argmax = scan_volt[arg_temp_max:, m_scan, 1]

    # # fit the signal in volt with polynom over time
    # p_scan_volt_c = np.poly1d(np.polyfit(scan_volt[arg_temp_max:, m_scan, 0],
    #                                      scan_volt_c_argmax, 9))

    # # fit the fit over time to the temperature of the blackbody
    # p_scan_volt = np.poly1d(np.polyfit(p_scan_volt_c(bbhole[arg_temp_max:,
    #                                                  m_bbhole, 0]),
    #                                    bbhole_c_argmax,
    #                                    9))

    p_recalc = np.poly1d(np.polyfit(scan_volt_c_argmax, bbhole_c_argmax, 9))

    # generate variable for temperature in the right shape
    scan_temp = scan_volt.copy()

    # save conversion of to temperature in new variable
    # scan_temp[:, :, 1] = p_scan_volt(scan_volt[:, :, 1])

    scan_temp[:, :, 1] = p_recalc(scan_volt[:, :, 1])

    # plt.figure('test')
    # plt.plot(scan_temp[:, m_scan, 0], scan_temp[:, m_scan, 1], label='old')
    # plt.plot(scan_temp_2[:, m_scan, 0], scan_temp_2[:, m_scan, 1], label='new')
    # plt.plot(bbhole[:, m_bbhole, 0], bbhole[:, m_bbhole, 1], label='bbhole')
    # plt.legend()
    # plt.grid()
    # input()
    # # print(scan_temp[:, m_scan, 1])
    # # print(np.shape(scan_temp))
    # print(m_scan)

    return(scan_temp)
