def save_to(filename, var, varname='none'):
    """[summary]

    save calculated variable data to folder with original data

    Parameters
    ----------
    filename : {string}
        filename of loaded data file
    var : {polyfit1d}
        calculated data as polyfit1d
    varname : {string}
        name of this calculated variable
    """
    import numpy as np

    # check if user wants to save data
    print('save calculated data to data folder? [Y/n]')
    if input() == 'n' or input() == 'q':
        return(0)

    # if no variable name was given ask the user to input one
    if varname == 'none':
        print('please enter name of calclated variable to save:')
        varname = input()

    # split the given filename and append the variable name
    filename_var = filename.split('.')[0] + '_' + str(varname) + '.npy'

    # hint the user where the file was saved to and what the filename is
    print(str(varname) + ' saved to:\n')
    print(filename_var + '\n')

    # actually save the variable to the data folder
    np.save(filename_var, var)


# some depreciated code to check if save_to works
# import numpy as np

# esd = np.arange((10))
# save_to('/home/max/P3/tc_sim_p3_V12/test.npy', esd, 'esd')
# # save_to('/home/max/P3/tc_sim_p3_V12/test.npy', esd)
# # save_to('test.npy', esd)
