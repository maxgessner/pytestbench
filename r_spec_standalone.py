import numpy as np
import os
import easygui
import matplotlib.pyplot as plt
from tkinter import Tk as tk
from tkinter.filedialog import askopenfilename as aofn
from scipy.interpolate import interp1d as int1d
# from pyro_signal_to_temp import getcalibration
from pyro_signal_to_temp import pyrotemp, pyrotemp2
from convert_puv_signal import convert_puv_signal
from load_ref_simple import load_ref
# from save_to import save_to
import warnings
# import ipdb
warnings.simplefilter('ignore', RuntimeWarning)
plt.ion()

skip_data_choice = False

wl = 900e-9  # m
wl_bbhole = 903.1e-9  # nm            wavelength
wl_scan = 904.4e-9  # nm            wavelength
# n = 1

# sigma = 5.670367e-8  ### W m-2 K-4

C_2 = 0.014388  # m*K      constant
T_90_Cu = 1357.77  # K     melting plateau of pure copper

length = 39.6e-3

r_o = 3.8e-3
r_i = 3e-3

# geometric calculations
p = r_o * 2 * np.pi  # + r_i * 2 * np.pi
o_a = 4.776
# o_a = 2
hole = (r_o - r_i) * 5e-4
# s = (r_o**2 - r_i**2) * np.pi * (360. - o_a) / 360.
s = (r_o**2 - r_i**2) * np.pi - hole
# s = r_o**2 * np.pi * (360 - o_a) / 360

# physical settings
t_a = 300.  # K

# physical constants
sigma = 5.670367e-8  # W m-2 K-4

# set variable 'home' as home directory of active user
home = os.path.expanduser('~')

if skip_data_choice is True:
    path = ''

    # filenames of measurement files itself
    filename = 'Niob_B(3)_output.npy'
    # filename = 'Niob_B(2)_output.npy'
    # filename = 'Niob_B(1)_output.npy'
    # filename = 'Niob_B_output.npy'

    file = path + filename

elif skip_data_choice is False:
    window = tk()
    window.withdraw()

    file = aofn(filetypes=[('npy files', '*_output.npy')],
                initialdir='.')

    if file == ():
        exit()

    path, filename = os.path.split(file)
    path = path + '/'

    window.destroy()

    window = tk()
    window.withdraw()
    if file == () or filename == ():
        exit()

ref_file = ''

# load references
rho_ref_x, rho_ref_y = load_ref(ref_file + 'rho_100.npy.npz')

# convert reference values to functions
f_rho = int1d(rho_ref_x, rho_ref_y, kind='linear',
              bounds_error=False, fill_value=(rho_ref_y[0], rho_ref_y[-1]))

# read header file and split in array
headerfile = file[:-4] + '.h'
with open(headerfile, 'r') as hf:
    for val in hf:
        header = np.array(val.strip().split('\t'))

print('processing ...')
print(file)

# actually load data
data = np.load(file)

# split columns in explicit variables
for i in range(len(header)):
    if 'r_spec' in header[i]:
        # convert column 'r_spec' to variable
        voltage = np.stack((data[:, :, 0], data[:, :, i]), axis=2)
        # np.save(file[:-4] + '_voltage.npy', voltage)

    # if header[i] == 'r_spec2':
    #     # convert column 'r_spec2' to variable
        voltage2 = np.stack((data[:, :, 0], data[:, :, -1], data[:, :, i]),
                            axis=2)
        # np.save(file[:-4] + '_voltage2.npy', voltage2)

    if 'current' in header[i]:
        # convert column 'currnet' to variable
        # measurement is already converted in amps so NO factor 500
        current = np.stack((data[:, :, 0], data[:, :, i]), axis=2)
        # np.save(file[:-4] + '_current.npy', current)

    if 'pv' in header[i]:
        # convert column 'pv' to variable
        bbhole_volt = np.stack((data[:, :, 0], data[:, :, i]), axis=2)
        bbhole = bbhole_volt.copy()
        # convert pyrometer signal to real temperature
        bbhole[:, :, 1] = pyrotemp(bbhole[:, :, 1], 'PV')
        # esd_ref_x, esd_ref_y = load_ref(ref_file + '/esd_100.npy.npz')
        # f_esd = int1d(esd_ref_x, esd_ref_y, kind='linear',
        #               bounds_error=False,
        #               fill_value=(esd_ref_y[0], esd_ref_y[-1]))
        # bbhole[:, 1] = pyrotemp2(bbhole[:, 1], 'PV', f_esd)
        # print(bbhole2[:, :, 0])
        # print(bbhole2[:, :, 2])

        # plt.figure('bbhole')
        # plt.plot(bbhole[:, 0], bbhole[:, 1])
        # input()
        # np.save(file[:-4] + '_bbhole.npy', bbhole)
    if 'puv' in header[i]:
        # convert column 'puv' to variable
        scan_volt = np.stack((data[:, :, 0], data[:, :, i]), axis=2)
        scan = scan_volt.copy()
        # scan[:, :, 1] = pyrotemp2(scan_volt[:, :, 1], 'PUV', f_esd)
        scan = convert_puv_signal(bbhole, scan_volt)
        scan_with_x = np.stack((data[:, :, 0],
                                data[:, :, -1],
                                data[:, :, i]), axis=2)

    # print(voltage)

# get median temperature to set range for heating phase
mtemp = np.int(np.shape(scan)[1] / 2)

# get point of maximum temperature
max_temp_point = np.int((np.argwhere(scan[:, mtemp, 1] ==
                        np.max(scan[:, mtemp, 1])))[0])

# get mean values per scan of voltage and current
voltage_drop = np.mean(voltage[:, :, 1], axis=1)
mcurrent = np.mean(current[:, :, 1], axis=1)

# find position of maximum current and maximum voltage
max_current_pos = np.argmax(mcurrent)
max_voltage_pos = np.argmax(voltage_drop)

# find where voltage is off due to 0 current
voltage_stop = np.min(np.where(voltage_drop[max_voltage_pos:] < 0.01)) + \
    max_voltage_pos

# find where current is off
current_stop = np.min(np.where(current[max_current_pos:] < 10)) + \
    max_current_pos

# set stop whatever happens first, current off or voltage off
stop = np.min(voltage_stop, current_stop)
stop -= 1

# cut arrays to only containg values until current off or voltage off
voltage_drop = voltage_drop[:stop]
mcurrent = mcurrent[:stop]

# calculate resistance
R = np.abs(np.true_divide(voltage_drop, mcurrent))
# print(R)

rho = R * s / length

scan = scan[:stop]
bbhole = bbhole[:stop]
# R = R[:stop]

r_mean = voltage_drop[1] / mcurrent[-1] * s / length
temp_mean = np.mean(scan[-1, :, 1])

# fit_method = easygui.buttonbox('select option for fit method',
#                                'fit_method',
#                                ('curve_fit',
#                                 'least_squares',
#                                 'fmin',
#                                 'brute',
#                                 'minimize_scalar',
#                                 'fminbound'))
msg = 'select option for fit method'
title = 'fit_method'
choices = ['curve_fit',
           'least_squares',
           'fmin',
           'brute',
           'minimize_scalar',
           'fminbound']
fit_method = easygui.choicebox(msg, title, choices)

# print(fit_method)
if fit_method == None:
    exit()

# options fot variable fit_method
# fit_method = 'curve_fit'
# fit_method = 'least_squares'
# fit_method = 'fmin'
# fit_method = 'brute'
# fit_method = 'minimize_scalar'
# fit_method = 'fminbound'


# initializion values
temp_0 = np.mean(scan[-1, :, 1])
# alpha_0 = np.array(0.001)
alpha_0 = np.array(0)
# alpha_0 = np.array(0.0002)
# alpha_0 = np.array(0.1)

global value

# make reference as a function
f_rho = int1d(rho_ref_x, rho_ref_y, kind='linear',
              bounds_error=False, fill_value=(rho_ref_y[0], rho_ref_y[-1]))

# generate value to expand from
R_0 = (r_mean * length / s) / np.shape(scan)[1]

# models for fiting
def model(alpha, temp):
    R_sum = np.sum(R_0 * (1 + (alpha * (temp - temp_0))))
    return R_sum


def residual(alpha, temp, R_ges):
    rest = (model(alpha, temp) - R_ges) / (R_ges)
    # return np.array(rest).ravel()
    return rest


def cf_model(temp, alpha):
    return(np.sum(R_0 * (1 + alpha * (temp - temp_0))))


def residual_2(alpha):
    rest = (np.sum(R_0 * (1 + (alpha * (temp2 - temp_0)))) - R_ges)
    return rest


# boundary conditions
R_ges = voltage_drop[-1] / mcurrent[-1]

temp2 = scan[-1, :, 1]

if fit_method == 'curve_fit':
    ###########################################################################
    ##### START: try with curve_fit ###########################################
    ###########################################################################

    from scipy.optimize import curve_fit

    print('curve_fit')

    R_ges = voltage_drop[-1] / mcurrent[-1]

    popt, pcov = curve_fit(cf_model, np.mean(scan[-1, :, 0]), R_0,
                           bounds=(0, 0.1))

    print(popt)

    value = popt

    ###########################################################################
    ##### END: try with curve_fit #############################################
    ###########################################################################


if fit_method == 'least_squares':
    ###########################################################################
    ##### START: try with least_squares #######################################
    ###########################################################################

    from scipy.optimize import least_squares

    print('least_squares')

    res = least_squares(residual, alpha_0, args=(scan[-1, :, 1], R_ges),
                        verbose=2,
                        bounds=(-0.01, 0.01),
                        # xtol=2.22044604926e-16,
                        # ftol=2.22044604926e-16,
                        # gtol=2.22044604926e-16,
                        # xtol=0,
                        gtol=0,
                        # ftol=0,
                        # loss='cauchy',
                        # method='trf',
                        max_nfev=1000,
                        # tr_solver='exact',
                        # loss='linear',
                        f_scale=0.0001,
                        # x_scale=1e10,
                        # diff_step=0.00001,
                        )

    print(res['x'])

    value = res['x']

    ###########################################################################
    ##### END: try with least_squares #########################################
    ###########################################################################


if fit_method == 'fmin':
    ###########################################################################
    ##### START: try with fmin ################################################
    ###########################################################################
    from scipy.optimize import fmin

    print('fmin')

    res = fmin(residual_2, alpha_0, disp=True, retall=True,
               ftol=1e-10,
               xtol=1e-10,
               maxfun=1e5,
               full_output=False,
               )

    print(res)

    value = res[0]

    ###########################################################################
    ##### START: try with fmin ################################################
    ###########################################################################


if fit_method == 'brute':
    ###########################################################################
    ##### START: try with brute ###############################################
    ###########################################################################
    from scipy.optimize import brute

    print('brute')

    res = brute(residual_2, [slice(0, 0.001, 0.00000001)],
                full_output=True,
                workers=-1
                )

    print(res)

    value = res[0]
    ###########################################################################
    ##### START: try with brute ###############################################
    ###########################################################################


if fit_method == 'minimize_scalar':
    ###########################################################################
    ### START: new test with minimize_scalar ##################################
    ###########################################################################
    from scipy.optimize import minimize_scalar

    print('minimize_scalar')

    res = minimize_scalar(residual, args=(scan[-1, :, 1], R_ges),
                          bounds=(0, 0.001),
                          # bracket=(0.01, 0.001),
                          tol=1e-10,
                          method='bounded',
                          options={'xatol':1e-5,
                                   'disp':3,
                                   }
                          # retall=True,
                          )

    print(res)

    value = res['x']

    ###########################################################################
    ### END: new test with minimize_scalar ####################################
    ###########################################################################


if fit_method == 'fminbound':
    ###########################################################################
    ##### START: try with fminbound ###########################################
    ###########################################################################
    from scipy.optimize import fminbound

    print('fminbound')

    res = fminbound(residual, 0, 0.001, args=(scan[-1, :, 1], R_ges),
                    # disp=True,
                    # retall=True,
                    )

    print(res)

    value = res

    ###########################################################################
    ### END: new test with fminbound ##########################################
    ###########################################################################

# temperature profile
r_temp = scan[-1, :, 1]

# generate rho from fit and temperature values
r = f_rho(temp_0) * (1 + (value * (scan[-1, :, 1] - temp_0)))

# generate temperature range from profile
temp_range = np.linspace(np.min(scan[-1, :, 1]), np.max(scan[-1, :, 1]), 1000)

# generate polynom from calculated values
p_r_3 = np.poly1d(np.polyfit(r_temp, r, 3))
r_3 = p_r_3(temp_range)

# reference value from cezairliyan
r_cezairliyan = 10.74 + 3.396e-2 * temp_range - 1.823e-6 * temp_range**2
r_cezairliyan *= 1e-8

# mean value of resistivity NOT fitted
r_mean = voltage_drop[1] / mcurrent[-1] * s / length


plt.figure('diff rho')
plt.title(r'$\rho$: deviation from input values')
plt.plot(r_temp, (r - f_rho(r_temp)) / f_rho(r_temp) * 100, '.',
         label='diff [%]')
plt.xlabel('temperature T / K')
plt.ylabel('difference from lit. value / %')
plt.ticklabel_format(useOffset=False)
plt.grid()
plt.legend()

plt.figure('resistivity vs temperature')
plt.title(r'specific resistivity $\rho$ / $\Omega * m$')
plt.xlabel(r'temperature T / $K$')
plt.ylabel(r'specific resistivity $\rho$ / $\Omega * m$')
plt.ticklabel_format(useOffset=False)

plt.plot(r_temp, r, 'x', label=r'$\rho$ cal')
plt.plot(temp_range, r_3, 'x', label=r'$\rho$ cal')
plt.plot(temp_range, r_cezairliyan, label=r'$\rho$ cezairliyan')
plt.plot(temp_mean, r_mean, 'bx', label='rho_mean')

plt.plot(temp_range, f_rho(temp_range), 'r-', label=r'$\rho$ ref')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
plt.grid()
plt.legend()
input()

# save_to(file, r, 'r_spec')
