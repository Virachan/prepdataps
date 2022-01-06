#!/usr/bin/env python
# -*- coding: utf-8 -*-

# NOTE: Include license!!!
# Author: Viraja Khatu


# ---------------------------------------------------------------------------------------------------------------------
# Import required modules
import numpy as np
from random import gauss
import statistics as stats
from astropy.io import ascii
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from scipy.interpolate import splrep, splev
import os, log


# ---------------------------------------------------------------------------------------------------------------------
def start(data_path, target_filenames, reference_filenames, interactive=True, windows_x=[], spline_order=3, \
    continuum_points=[], continuum_window_width=10, recovered_curve_filenames=[]):
    """
    Recover the shape of the target curves based on the shape of the reference curves.  The curves should be 
    one-dimensional (1-D), e.g., 1-D spectra.  A spline function is used to approximate shapes of the reference curves
    and the shapes of the target curves are estimated by interpolating the shapes of the reference curves at specified 
    target x quantities.  Errors in interpolated values for target curve windows are given by statistical errors 
    calculated from fitting the spline function to mock data generated in the reference curve windows.  In the 
    interactive mode, the scripts allows visual inspection at all stages of the recovery process.  The reference and 
    target curves may have weights that the fitting function uses to calculate the standard deviation of the fit 
    coefficients.

    INPUTS:
        - data_path - absolute path to the directory where the data are located; string
        - target_filenames - filenames of the target curves to be recovered; list (of strings)
        - reference_filenames - filenames of reference curves; list (of strings).  Number of filenames must match with 
                                the number of target filenames.
    
    OPTIONAL INPUTS:
        - interactive - if True, spectra are displayed at all intermediate stages of the recovery process for visual 
                        inspection and the order of the split fit can be set by the user; boolean.  Default is "True".
        - windows_x - starting and ending x quantities of the curve windows to be recovered in target spectra (same 
                      units as input curves); list of lists of strings (of comma-separated starting and ending values 
                      separated by a colon).  Default is an empty list.
        - spline_order - order of the spline function to approximate the reference curve shape; int.  Default is 3.
        - continuum_points - x values of the continuum free of any absorption or emission (same units as input x 
                             quantity); string (of floats separated by commas and no spaces).  Default is an empty 
                             list.
        - continuum_window_width - +/-"window" about the continuum point to be used to calculate the median continuum 
                                   error (same units as the input x quantity); float.  Default is 10.
        - recovered_curve_filenames - filenames of the recovered curves; list of (strings).  If a list is provided, the
                                      number of filenames must match with the number of target filenames.  If list is 
                                      empty, the target filenames are used to save the recovered curve files are used.  
                                      Default is an empty list.


    OUTPUTS:
        - ASCII files containing columns: x quantity (same units as input curves), y quantity (same units as input 
          curves), and error in y quantity (same units as input curves)
        - Updates or saves an ASCII file containing flag status of data points witth columns: pixel number and flag
    
    NOTES:
        1. Input target and reference curves must have the same units.
    """
    logger = log.getLogger('recoverCurveShapes.start')


    logger.info('###################################################')
    logger.info('#                                                 #')
    logger.info('#          START -- Recover curve shapes          #')
    logger.info('#                                                 #')
    logger.info('###################################################\n')


    ################################################################### 
    ##                                                               ##
    ##          START -- Recover Curve Shapes General Setup          ##
    ##                                                               ##
    ###################################################################

    
    # Print absolute path to the current working directory
    working_dir_path = os.getcwd()
    logger.info('Current working directory: %s', working_dir_path)


    # Check if the path to the data directory is provided
    if not data_path:
        logger.error('##########################################################################################')
        logger.error('##########################################################################################')
        logger.error('#                                                                                        #')
        logger.error('#          ERROR in recoverCurveShapes: Absolute path to the data directory not          #')
        logger.error('#                             available.  Exiting script.                                #')
        logger.error('#                                                                                        #')
        logger.error('##########################################################################################')
        logger.error('##########################################################################################\n')
        raise SystemExit
    else:
        logger.info('Absolute path to the data directory available.')
        logger.info('Data directory: %s', data_path)


    # Check if target and reference curves available in the data directory
    # Target curves
    curves_not_available_ctr = 0
    
    for filename in target_filenames:
        if not os.path.exists(data_path + '/' + filename):
            logger.info('Curve not available: %s', data_path + '/' + filename)
            curves_not_available_ctr += 1
    
    if curves_not_available_ctr > 0:
        logger.error('#############################################################################################')
        logger.error('#############################################################################################')
        logger.error('#                                                                                           #')
        logger.error('#          ERROR in recoverCurveShapes: %s target curves not available in the data          #', \
            curves_not_available_ctr)
        logger.error('#                              directory.  Exiting script.                                  #')
        logger.error('#                                                                                           #')
        logger.error('#############################################################################################')
        logger.error('#############################################################################################\n')
        raise SystemExit
    
    else:
        logger.info('All target curves avialable.')
    
    # Reference curves
    curves_not_available_ctr = 0
    
    for filename in reference_filenames:
        if not os.path.exists(data_path + '/' + filename):
            logger.info('Curve not available: %s', data_path + '/' + filename)
            curves_not_available_ctr += 1
    
    if curves_not_available_ctr > 0:
        logger.error('##########################################################################################')
        logger.error('##########################################################################################')
        logger.error('#                                                                                        #')
        logger.error('#          ERROR in correctSlitlosses: %s reference curves not available in the          #', \
            curves_not_available_ctr)
        logger.error('#                           data directory.  Exiting script.                             #')
        logger.error('#                                                                                        #')
        logger.error('##########################################################################################')
        logger.error('##########################################################################################\n')
        raise SystemExit
    
    else:
        logger.info('All reference curves avialable.')
    

    # Check if number of lists of curve windows to recover equal to the number of target curves
    if len(windows_x) > len(target_filenames):
        logger.error('########################################################################################')
        logger.error('########################################################################################')
        logger.error('#                                                                                      #')
        logger.error('#          ERROR in correctSlitlosses: Lists of curve windows to recover more          #')
        logger.error('#             than the number of target curves provided.  Exiting script.              #')
        logger.error('#                                                                                      #')
        logger.error('########################################################################################')
        logger.error('########################################################################################\n')
        raise SystemExit
    
    elif len(windows_x) < len(target_filenames):
        logger.error('########################################################################################')
        logger.error('########################################################################################')
        logger.error('#                                                                                      #')
        logger.error('#          ERROR in correctSlitlosses: Lists of curve windows to recover less          #')
        logger.error('#             than the number of target curves provided.  Exiting script.              #')
        logger.error('#                                                                                      #')
        logger.error('########################################################################################')
        logger.error('########################################################################################\n')
        raise SystemExit
    
    else:
        logger.info('Number of lists of curve windows to recover is equal to the number of target curves.')
    

    # Check if continuum points to calculate median error for target provided
    if continuum_points:
        logger.info('Continuum points available.')
    else:
        logger.info('Continuum points not available.')


    # Check if recovered curve filenames provided
    if not recovered_curve_filenames:
        logger.warning('#########################################################################################')
        logger.warning('#########################################################################################')
        logger.warning('#                                                                                       #')
        logger.warning('#          WARNING in correctSlitlosses: Output filenames for recovered curves          #')
        logger.warning('#                                   not available.                                      #')
        logger.warning('#                                                                                       #')
        logger.warning('#########################################################################################')
        logger.warning('#########################################################################################\n')
        logger.info('Using the input target curve filenames as the output target curve filenames.')
    else:
        logger.info('Output filenames available.')
    

    ###################################################################### 
    ##                                                                  ##
    ##          COMPLETE -- Recover Curve Shapes General Setup          ##
    ##                                                                  ##
    ######################################################################
    
    
    # Loop through all target curves
    for i in range(len(target_filenames)):
        
        target_filename = data_path + '/' + target_filenames[i]
        logger.info('Working on curve %s: %s', i+1, target_filename)

        
        # Load target curve
        target = ascii.read(target_filename)
        x_qty_target = target['col1']  ## target x quantity
        y_qty_target = target['col2']  ## target y quantity
        error_target = target['col3']  ## target error in y quantity


        # Load reference curve
        reference_filename = data_path + '/' + reference_filenames[i]
        reference = ascii.read(reference_filename)
        x_qty_reference = reference['col1']  ## reference x quantity
        y_qty_reference = reference['col2']  ## reference y quantity
        error_reference = reference['col3']  ## reference error in y quantity
    
        
        if interactive:        
            # Plot original target and reference curves
            plot_title = target_filename[target_filename.rfind('/')+1:] + ': Before recovery'
            plt.title(plot_title, fontsize=16)  ## plot title
            plt.xlabel('Spectral axis', fontsize=16)  ## label X axis
            plt.ylabel('Counts or count rate', fontsize=16)  ## label Y axis
            
            plt.plot(x_qty_target, y_qty_target, color='k', linestyle='-', linewidth=2, label='Target')  ## plot y 
                                                                           ## quantity of the target on the Y axis
            plt.fill_between(x_qty_target, y_qty_target - error_target, y_qty_target + error_target, color='gray', \
                alpha=0.5)  ## plot error in y quantity of the target on the Y axis
            
            plt.plot(x_qty_reference, y_qty_reference, color='g', linestyle='-', linewidth=2, label='Reference')  
                                                               ## plot y quantity of the reference on the Y axis
            plt.fill_between(x_qty_reference, y_qty_reference - error_reference, y_qty_reference + error_reference, \
                color='g', alpha=0.5)  ## plot error in y quantity of the reference on the Y axis
            
            plt.legend(loc='best', fontsize=16)  ## assign legend
            
            plt.tick_params(axis='x', labelsize=16)  ## assign axes label size
            plt.tick_params(axis='y', labelsize=16)

            plt.show()
    

        # Parse the starting and ending x quantities of the curve windows to be recovered
        if not windows_x[i]:
            # If windows not available, warn the user
            logger.warning('######################################################################################')
            logger.warning('######################################################################################')
            logger.warning('#                                                                                    #')
            logger.warning('#          WARNING in recoverCurveShapes: Windows to recover not available.          #')
            logger.warning('#                                                                                    #')
            logger.warning('######################################################################################')
            logger.warning('######################################################################################\n')
            logger.info('Skipping recovering curve shapes.\n')
            continue

        else:
            # If windows available, parse the starting and ending values
            logger.info('Windows to recover available.')

            logger.info('Parsing the starting and ending x quantities of the curve windows to be recovered.')
            windows_x_lb = np.array([x.split(':')[0] for x in windows_x[i]]).astype(float) ## stores lower bounds of 
                                                                                      ## all windows to be recovered
            windows_x_ub = np.array([x.split(':')[1] for x in windows_x[i]]).astype(float)  ## stores upper bounds of
                                                                                       ## all windows to be recovered
            
            pixels_to_recover = []

            for j in range(len(windows_x[i])):                
                if (windows_x_lb[j] < x_qty_target[0]) or (windows_x_ub[j] > x_qty_target[-1]):
                    logger.error('##############################################################################')
                    logger.error('##############################################################################')
                    logger.error('#                                                                            #')
                    logger.error('#          ERROR in recoverCurveShapes: Invalid bounds detected for          #')
                    logger.error('#              curve windows to be recovered.  Exiting script.               #')
                    logger.error('#                                                                            #')
                    logger.error('##############################################################################')
                    logger.error('##############################################################################\n')
                    raise SystemExit
                
                elif windows_x_lb[j] == windows_x_ub[j]:
                    logger.error('##############################################################################')
                    logger.error('##############################################################################')
                    logger.error('#                                                                            #')
                    logger.error('#          ERROR in recoverCurveShapes: Same lower and upper bounds          #')
                    logger.error('#                detected for curve windows to be recovered.                 #')
                    logger.error('#                              Exiting script.                               #')
                    logger.error('#                                                                            #')
                    logger.error('##############################################################################')
                    logger.error('##############################################################################\n')
                    raise SystemExit
                
                elif windows_x_lb[j] > windows_x_ub[j]:
                    logger.error('###############################################################################')
                    logger.error('###############################################################################')
                    logger.error('#                                                                             #')
                    logger.error('#          ERROR in recoverCurveShapes: Reversed bounds detected for          #')
                    logger.error('#            curve windows to be recovered where the upper bound is           #')
                    logger.error('#                smaller than the lower bound.  Exiting script.               #')
                    logger.error('#                                                                             #')
                    logger.error('###############################################################################')
                    logger.error('###############################################################################\n')
                    raise SystemExit
                
                elif windows_x_ub[j] > windows_x_lb[j]:
                    window_indices = np.where((x_qty_target >= windows_x_lb[j]) & (x_qty_target <= windows_x_ub[j]))
                                                                      ## indices of the curve window to be recovered
                    x_qty_window_target = x_qty_target[window_indices]
                    for k in x_qty_window_target:
                        pixels_to_recover.append(k)

                else:
                    pass
        

        # Assign flag status to individual pixels in the target spectrum
        logger.info('Assigning flag status to individual pixels (1 for pixels to be recovered and 0 for pixels ' + \
            'not to be recovered).')

        flag_status_filename_1 = target_filename[target_filename.rfind('/')+1:]
        if flag_status_filename_1.find('_') == -1:
            flag_status_filename = flag_status_filename_1[:flag_status_filename_1.rfind('.')] + '_flags.dat'
        else:
            flag_status_filename = flag_status_filename_1[:flag_status_filename_1.find('_')] + '_flags.dat'
        flag_status_path = data_path + '/' + flag_status_filename
        
        # Check if flag status file already exists
        if os.path.exists(flag_status_path):
            # If yes, append flag status of the spectrum to the file
            logger.info('Flag status file already exits.')
            logger.info('Appending the existing flag status file.')

            flags = ascii.read(flag_status_path)
            x_qty_flags = flags['col1']  ## flags x quantity
            y_qty_flags = flags['col2']  ## flags y quantity
            
            flags_dict = {}  ## organizes the flag status in a dictionary
            for j in range(len(x_qty_flags)):
                flags_dict[x_qty_flags[j]] = y_qty_flags[j]

            for j in pixels_to_recover:
                flags_dict[j] = 1  ## assign flag status 1 for pixels to be recovered
            
            flags_dict_keys = sorted(flags_dict)
            flag_status = [flags_dict[k] for k in flags_dict_keys]

            np.savetxt(flag_status_path, np.c_[flags_dict_keys, flag_status]); \
                logger.info('Saved an ASCII file named %s.', flag_status_path)
        
        else:
            # If no, save flag status of the spectrum in an ASCII file
            logger.info('Flag status file does not exit.')
            logger.info('Generating new flag status file.')

            flag_status = []  ## stores flag status of individual pixels
            
            for j in range(len(x_qty_target)):
                if x_qty_target[j] in pixels_to_recover:
                    flag_status.append(1)
                
                elif x_qty_target[j] not in pixels_to_recover:
                    flag_status.append(0)  ## assign flag status 0 for pixels not to be recovered
                
                else:
                    logger.error('###########################################################################')
                    logger.error('###########################################################################')
                    logger.error('#                                                                         #')
                    logger.error('#          ERROR in recoverCurveShapes: Invalid x quantity value          #')
                    logger.error('#               detected to be recovered.  Exiting script.                #')
                    logger.error('#                                                                         #')
                    logger.error('###########################################################################')
                    logger.error('###########################################################################\n')
                    raise SystemExit

            np.savetxt(flag_status_path, np.c_[x_qty_target, flag_status]); \
                logger.info('Saved an ASCII file named %s.', flag_status_path)
        

        if continuum_points:
            logger.info('Setting the median error of the continuum window as the standard deviation to generate ' + \
                'simulated data for computing statistical errors.')

            # Calculate the median error in the continuum regions of the spectra that are free of absorption or 
            # emission
            logger.info('Calculating median error in the continuum regions of the spectrum.')
            
            continuum_window_error = []
            for j in range(len(continuum_points)):
                continuum_window_indices = np.where((x_qty_target >= continuum_points[j] - continuum_window_width) & \
                    (x_qty_target <= continuum_points[j] + continuum_window_width))
                continuum_window_error = continuum_window_error + \
                                         list(*[error_target[k] for k in continuum_window_indices])
            
            continuum_error_median = stats.median(continuum_window_error)
        
        else:
            logger.info('Continuum points not available.  Setting the errors in the recovery windows of the ' + \
                'reference as the standard deviation to generate simulated data for computing statistical errors.')
            
            continuum_error_median = ''
        
        
        y_qty_target_rcov = dc(y_qty_target)  ## create a deep copy of the y quantity array of the target to store the
                                              ## y quantity array with recovered curve windows
        error_target_rcov = dc(error_target)  ## create a deep copy of the error in the y quantity array of the target
                                              ## to store the error in the y quantity array with recovered curve
                                              ## windows
        # NOTE: A deep copy allows to maintain the orginal list or array separate from the new list or array so that
        # when the new list or array is edited, the original does not get appended.
        
        
        # Fit spline function to the reference curve windows and evaluate shape of corresponding target windows
        logger.info('Fitting spline function to reference curve windows and evaluating shape of corresponding ' + \
            'target windows.')
        
        for j in range(len(windows_x_lb)):
            window_indices = np.where((x_qty_target >= windows_x_lb[j]) & (x_qty_target <= windows_x_ub[j]))
            x_qty_window_reference = x_qty_reference[window_indices]
            y_qty_window_reference = y_qty_reference[window_indices]
            error_window_reference = error_reference[window_indices]
            x_qty_window_target = x_qty_target[window_indices]

            y_qty_window_shape_reference = splrep(x_qty_window_reference, y_qty_window_reference, \
                w=1 / error_window_reference, k=spline_order)
            y_qty_window_shape_target = splev(x_qty_window_target, y_qty_window_shape_reference)
            
            
            if interactive:
                # Plot original target and reference curves with shape of the target curve windows
                plot_title = target_filename[target_filename.rfind('/')+1:] + ': Shape of target curve window'
                plt.title(plot_title, fontsize=16)
                plt.xlabel('Spectral axis', fontsize=16)
                plt.ylabel('Counts or count rate', fontsize=16)
                
                plt.plot(x_qty_target, y_qty_target, color='k', linestyle='-', linewidth=2, label='Target')
                plt.fill_between(x_qty_target, y_qty_target - error_target, y_qty_target + error_target, color='gray',\
                    alpha=0.5)

                plt.plot(x_qty_reference, y_qty_reference, color='g', linestyle='-', linewidth=2, label='Reference')  
                plt.fill_between(x_qty_reference, y_qty_reference - error_reference, \
                    y_qty_reference + error_reference, color='g', alpha=0.5)
                
                plt.plot(x_qty_window_target, y_qty_window_shape_target, color='r', linestyle='-', linewidth=2, \
                    label='Target curve window')  ## plot shape of the target curve window on the Y axis
                
                plt.legend(loc='best', fontsize=16)
                
                plt.tick_params(axis='x', labelsize=16)
                plt.tick_params(axis='y', labelsize=16)

                plt.show()

            
            # Compute statistical uncertainties for the recovered target curve window
            y_qty_window_mock_shape_target_points = np.zeros((1000, len(x_qty_window_reference)))  ## stores recovered 
                                                                             ## values of the mock target curve window
            

            for iteration in range(1000):
                if continuum_error_median:
                    y_qty_window_mock_data = [gauss(y_qty_target[k], continuum_error_median) \
                        for k in window_indices[0]]  ## randomly select y quantity values from a gaussian distribution
                else:
                    y_qty_window_mock_data = [gauss(y_qty_reference[k], error_reference[k]) \
                        for k in window_indices[0]]
                y_qty_window_mock_shape_target_points[iteration,:] = y_qty_window_mock_data
            
            error_window_shape_target = [stats.stdev(y_qty_window_mock_shape_target_points[:,k]) \
                    for k in range(len(x_qty_window_reference))]  ## calculate standard deviation of the recovered
                                                                  ## values of the mock target curve window as error
                                                                  ## in the recovered values
            

            # Replace target curve windows with recovered values and their errors
            for k in range(len(x_qty_window_reference)):
                y_qty_target_rcov[window_indices[0][k]] = gauss(y_qty_window_shape_target[k], \
                    error_window_shape_target[k])
                error_target_rcov[window_indices[0][k]] = error_window_shape_target[k]
        

        if interactive:
            # Plot original and recovered target curves
            plot_title = target_filename[target_filename.rfind('/')+1:] + ': After recovery'
            plt.title(plot_title, fontsize=16)
            plt.xlabel('Spectral axis', fontsize=16)
            plt.ylabel('Counts or count rate', fontsize=16)
            
            plt.plot(x_qty_target, y_qty_target, color='k', linestyle='-', linewidth=2, label='Original target')
            plt.fill_between(x_qty_target, y_qty_target - error_target, y_qty_target + error_target, color='gray', \
                alpha=0.5)
            
            plt.plot(x_qty_target, y_qty_target_rcov, color='purple', linestyle='-', linewidth=2, \
                label='Recovered target')  ## plot target curve combined with recovered windows on the Y axis
            plt.fill_between(x_qty_target, y_qty_target_rcov - error_target_rcov, \
                y_qty_target_rcov + error_target_rcov, color='purple', alpha=0.5)  ## plot error in the target curve
                                                                    ## combined with recovered windows on the Y axis
            
            plt.legend(loc='best', fontsize=16)
            
            plt.tick_params(axis='x', labelsize=16)
            plt.tick_params(axis='y', labelsize=16)

            plt.show()
         

        # Save target curve with recovered windows in an ASCII file
        if not recovered_curve_filenames:
            target_filename_rcov = target_filename
        else:
            target_filename_rcov = data_path + '/' + recovered_curve_filenames[i]

        np.savetxt(target_filename_rcov, np.c_[x_qty_target, y_qty_target_rcov, error_target_rcov]); \
            logger.info('Saved an ASCII file named %s.', target_filename_rcov)
    

    logger.info('######################################################')
    logger.info('#                                                    #')
    logger.info('#          COMPLETE -- Recover curve shapes          #')
    logger.info('#                                                    #')
    logger.info('######################################################\n')

    return


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Set log file
    log.configure('recoverCurveShapes.log', filelevel='INFO', screenlevel='DEBUG')

    start(data_path='/Users/viraja/GitHub/prepdataps/example_spectra', \
        target_filenames=['strcgN20190329S0085std_p_s_v1.dat'], \
        reference_filenames=['mean_error_std_p_s_500.dat'], \
        interactive=True, \
        windows_x = [['2001:2850']], \
        spline_order=3, \
        continuum_points=[2590, 3940, 4550], \
        continuum_window_width=10., \
        recovered_curve_filenames=['strcgN20190329S0085std_p_s.dat'])
