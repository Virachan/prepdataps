#!/usr/bin/env python
# -*- coding: utf-8 -*-

# NOTE: Include license!!!
# Author: Viraja Khatu


# ---------------------------------------------------------------------------------------------------------------------
# Import required modules
import numpy as np
from pyraf import iraf
import statistics as stats
from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.signal import cspline1d_eval
from utilities import error_add_sub, error_mul_div_var, error_mul_div_const
from configparser import ConfigParser
import os, sys, log


# ---------------------------------------------------------------------------------------------------------------------
def start(configfile):
    """
    Correct for slitlosses - light from the target spectra lost due to narrow slits.  The amount of light lost can be 
    estimated with reference spectra taken with wide slits on the same night.  However, correcting for the slitlosses
    requires the standard star spectrum taken in the same slit as the science spectrum.  This script corrects for the 
    following sources of slitlosses.
        1. Wavelength-dependence
    
    The shape of the slitlosses across the spectra is approximated by fitting a spline to the ratios of the target to 
    reference spectra.  Theere are two ways of calculating the ratios:
        1. "one_to_one": where the target spectra are divided by the reference spectra at every point in the x quantity
        2. "binned": where both the target and the reference spectra are binned in the x quantity with a specified bin
                     size, and the ratios are of the mean y quantities of the bins whose errors are given by 
                     propoagated errors
    
    The script can be run interactively, where the spectra can be visualized at all stages of the correction process,
    whereas the non-interactive mode does not allow visualizations.  Spline fitting with this script is only available 
    non-interactively, but the order of the fit can be set by the user.  Errors in slitloss-corrected spectra are given 
    by the propagated errors from spline fitting.

    INPUTS:
        - data_path - absolute path to the directory where the data are located; string
        - target_filenames - filenames of spectra to be corrected for slitlosses; string (of strings separated by 
                             commas and no spaces).  These must be ASCII files containing 3 columns: pixels or 
                             wavelength, counts or flux, and error in counts or flux.
        - reference_filenames - filenames of spectra to be used as references for correcting target spectra; list (of 
                                strings separated by commas and no spaces).  These must be ASCII files containing 3 
                                columns: pixels or wavelength, counts or flux, and error in counts or flux (with same 
                                units as input curves).  Number of filenames must match with the number of target 
                                filenames.
        - interactive - if True, spectra are displayed at all stages of the correction process for visual inspection; 
                        boolean.  Default is "True".
        - ratios_method - method for calculating spectral ratios; string.  Options are: "one_to_one" and "binned".  
                          Default is "one_to_one".
        - spectral_bin_size - size of the spectral bins in the x quantity for calculating spectral ratios by the 
                              "binned" method; float.  If the "ratios_method" parameter is set to "binned" and the 
                              "spectral_bin_size" is not specified, the script uses the default bin size of 10 (same 
                              units as the input x quantity).  If the "ratios_method" parameter is set to "one_to_one" 
                              and the "spectral_bin_size" is specified, the "ratios_method" parameter prevails.  
                              Default is 10.
        - spline_order - order of the spline function to approximate the slitlosses; int.  Default is 1.
        - output_spectra_filename_suffix - suffix of the slitloss-corrected spectra filenames; string.  Default is 
                                           "_l".
        - slitloss_model_results_filename - filename to store the slitloss-correction model results.  Default is 
                                            "slitloss_model_results.dat".

    OUTPUTS:
        - ASCII files containing columns: x quantity, y quantity, and error in y quantity (all quantities in the same 
          units as input spectra)
        - ASCII files containing model with columns: x quantity, model y quantity, data y quantity, and weight of data 
          y quantity (again, all quantities in the same units as input spectra)
        - ASCII file containing results of the spline model fit to spectral ratios in columns: model filename, 
          root-mean-squared (RMS), and reduced chi-squared statistic
    """
    logger = log.getLogger('correctSlitlosses.start')


    logger.info('#####################################################')
    logger.info('#                                                   #')
    logger.info('#          START -- Correct for slitlosses          #')
    logger.info('#                                                   #')
    logger.info('#####################################################\n')


    ##################################################################### 
    ##                                                                 ##
    ##          START -- Correct for Slitlosses General Setup          ##
    ##                                                                 ##
    #####################################################################

    
    # Set up IRAF
    iraf.utilities

    # Reset to default parameters the used IRAF tasks
    iraf.unlearn(iraf.utilities)
    
    # Set clobber to 'yes' for the script.  This still does not make the gemini tasks overwrite files, so: YOU WILL 
    # LIKELY HAVE TO REMOVE FILES IF YOU RE_RUN THE SCRIPT.
    user_clobber=iraf.envget("clobber")
    iraf.reset(clobber='yes')
    

    # Print absolute path to the current working directory
    working_dir_path = os.getcwd()
    logger.info('Current working directory: %s', working_dir_path)


    # Import config parameters
    logger.info('Importing configuration parameters from %s.\n', configfile)
    config = ConfigParser()
    config.optionxform = str  ## make options case-sensitive
    config.read(configfile)
    
    # Read correctSlitlosses specific config
    data_path = config.get('correctSlitlosses','data_path')
    target_filenames = config.get('correctSlitlosses','target_filenames')
    reference_filenames = config.get('correctSlitlosses','reference_filenames')
    interactive = config.getboolean('correctSlitlosses','interactive')
    ratios_method = config.get('correctSlitlosses','ratios_method')
    spectral_bin_size = config.getfloat('correctSlitlosses','spectral_bin_size')
    spline_order = config.getint('correctSlitlosses','spline_order')
    output_spectra_filename_suffix = config.get('correctSlitlosses','output_spectra_filename_suffix')
    slitloss_model_results_filename = config.get('correctSlitlosses','slitloss_model_results_filename')


    # Check if the path to the data directory is provided
    if not data_path:
        logger.error('#########################################################################################')
        logger.error('#########################################################################################')
        logger.error('#                                                                                       #')
        logger.error('#          ERROR in correctSlitlosses: Absolute path to the data directory not          #')
        logger.error('#                            available.  Exiting script.                                #')
        logger.error('#                                                                                       #')
        logger.error('#########################################################################################')
        logger.error('#########################################################################################\n')
        raise SystemExit
    else:
        logger.info('Absolute path to the data directory available.')
        logger.info('Data directory: %s', data_path)

    
    # Check if spectra provided by the user
    # Target spectra
    if not target_filenames:
        logger.error('##############################################################################')
        logger.error('##############################################################################')
        logger.error('#                                                                            #')
        logger.error('#          ERROR in correctSlitlosses: Target spectra not detected.          #')
        logger.error('#                              Exiting script.                               #')
        logger.error('#                                                                            #')
        logger.error('##############################################################################')
        logger.error('##############################################################################\n')      
        raise SystemExit
    else:
        logger.info('Target spectra detected.')
        target_filenames = [x for x in target_filenames.split(',')]  ## convert comma-separated string of strings into
                                                                     ## a list of strings
        logger.info('Target spectra: %s', target_filenames)
    
    # Reference spectra
    if not reference_filenames:
        logger.error('#################################################################################')
        logger.error('#################################################################################')
        logger.error('#                                                                               #')
        logger.error('#          ERROR in correctSlitlosses: Reference spectra not detected.          #')
        logger.error('#                               Exiting script.                                 #')
        logger.error('#                                                                               #')
        logger.error('#################################################################################')
        logger.error('#################################################################################\n')      
        raise SystemExit
    else:
        logger.info('Reference spectra detected.')
        target_filenames = [x for x in reference_filenames.split(',')]  ## convert comma-separated string of strings
                                                                        ## into a list of strings
        logger.info('Reference spectra: %s', reference_filenames)


    # Check if target and reference spectra available in the data directory
    # Target spectra
    spectra_not_available_ctr = 0
    
    for filename in target_filenames:
        if not os.path.exists(data_path + '/' + filename):
            logger.info('Spectrum not available: %s', data_path + '/' + filename)
            spectra_not_available_ctr += 1
    
    if spectra_not_available_ctr > 0:
        logger.error('#############################################################################################')
        logger.error('#############################################################################################')
        logger.error('#                                                                                           #')
        logger.error('#          ERROR in correctSlitlosses: %s target spectra not available in the data          #', \
            spectra_not_available_ctr)
        logger.error('#                              directory.  Exiting script.                                  #')
        logger.error('#                                                                                           #')
        logger.error('#############################################################################################')
        logger.error('#############################################################################################\n')
        raise SystemExit
    
    else:
        logger.info('All target spectra avialable.')
    
    # Reference spectra
    spectra_not_available_ctr = 0
    
    for filename in reference_filenames:
        if not os.path.exists(data_path + '/' + filename):
            logger.info('Spectrum not available: %s', data_path + '/' + filename)
            spectra_not_available_ctr += 1
    
    if spectra_not_available_ctr > 0:
        logger.error('###########################################################################################')
        logger.error('###########################################################################################')
        logger.error('#                                                                                         #')
        logger.error('#          ERROR in correctSlitlosses: %s reference spectra not available in the          #', \
            spectra_not_available_ctr)
        logger.error('#                           data directory.  Exiting script.                              #')
        logger.error('#                                                                                         #')
        logger.error('###########################################################################################')
        logger.error('###########################################################################################\n')
        raise SystemExit
    
    else:
        logger.info('All reference spectra avialable.')
    

    # Check if spectral bin size provided by the user according to the method specified for calculating spectral ratios
    if (ratios_method != 'one_to_one') and (ratios_method != 'binned'):
        logger.warning('###########################################################################################')
        logger.warning('###########################################################################################')
        logger.warning('#                                                                                         #')
        logger.warning('#          WARNING in correctSlitlosses: Invalid method for calculating spectral          #')
        logger.warning('#                                    ratios detected.                                     #')
        logger.warning('#                                                                                         #')
        logger.warning('###########################################################################################')
        logger.warning('###########################################################################################\n')
        logger.info('Setting the method for calculating spectral ratios to the default "one_to_one".')
        ratios_method = 'one_to_one'
        logger.info('Method for calculating spectral ratios: %s', ratios_method)
    
    elif (ratios_method == 'one_to_one') and spectral_bin_size:
        logger.info('Method for calculating spectral ratios: %s', ratios_method)
        logger.warning('##########################################################################################')
        logger.warning('##########################################################################################')
        logger.warning('#                                                                                        #')
        logger.warning('#          WARNING in correctSlitlosses: Not expecting both the "ratios_method"          #')
        logger.warning('#                   and the "spectral_bin_size" parameters to be set.                    #')
        logger.warning('#                                                                                        #')
        logger.warning('##########################################################################################')
        logger.warning('##########################################################################################\n')
        logger.info('Setting the size of the spectral bins to an empty string.')
        spectral_bin_size = ''

    elif (ratios_method == 'binned') and (not spectral_bin_size):
        logger.warning('######################################################################################')
        logger.warning('######################################################################################')
        logger.warning('#                                                                                    #')
        logger.warning('#          WARNING in correctSlitlosses: Size of the spectral bins in the x          #')
        logger.warning('#                         quantity not provided by the user.                         #')
        logger.warning('#                                                                                    #')
        logger.warning('######################################################################################')
        logger.warning('######################################################################################\n')
        logger.info('Setting the size of the spectral bins in the x quantity to 10 (same units as input x quantity).')
        spectral_bin_size = 10.

    elif (ratios_method == 'binned') and spectral_bin_size:
        logger.info('Size of the spectral bins in the x quantity: %s', spectral_bin_size)
    
    else:
        pass

    
    # Check if order of spline function provided by the user; if not, set to default if available
    if not spline_order:
        logger.warning('################################################################################')
        logger.warning('################################################################################')
        logger.warning('#                                                                              #')
        logger.warning('#          WARNING in correctSlitlosses: Order of the spline function          #')
        logger.warning('#                                not detected.                                 #')
        logger.warning('#                                                                              #')
        logger.warning('################################################################################')
        logger.warning('################################################################################\n')
        logger.info('Setting the order of the spline function to the default "1".')
        spline_order = 1
    else:
        logger.info('Order of the spline function detected.')
        logger.info('Order of the spline function: %s', spline_order)

    
    # Check if output filename suffix provided by the user; if not, set to default if available
    if not output_spectra_filename_suffix:
        logger.warning('########################################################################################')
        logger.warning('########################################################################################')
        logger.warning('#                                                                                      #')
        logger.warning('#          WARNING in correctSlitlosses: Output filename suffix not detected.          #')
        logger.warning('#                                                                                      #')
        logger.warning('########################################################################################')
        logger.warning('########################################################################################\n')
        logger.info('Setting the output filename suffix to the default "_l".')
        output_spectra_filename_suffix = '_l'
    else:
        logger.info('Output filename suffix detected.')
        logger.info('Output filename suffix: %s', output_spectra_filename_suffix)

    
    # Check if results filename provided by the user; if not, set to default if available
    if not slitloss_model_results_filename:
        logger.warning('##################################################################################')
        logger.warning('##################################################################################')
        logger.warning('#                                                                                #')
        logger.warning('#          WARNING in correctSlitlosses: Results filename not detected.          #')
        logger.warning('#                                                                                #')
        logger.warning('##################################################################################')
        logger.warning('##################################################################################\n')
        logger.info('Setting the results filename to the default "slitloss_model_results.dat".')
        slitloss_model_results_filename = 'slitloss_model_results.dat'
    else:
        logger.info('Results filename detected.')
        logger.info('Results filename: %s', slitloss_model_results_filename)
    
    
    # Define required variables
    logger.info('Defining required variables.')
    spectral_ratios_model_filenames = []  ## stores filenames of the spectral ratios model for slilosses
    spectral_ratios_model_rms = []  ## stores RMS of the spline model fit to spectral ratios
    spectral_ratios_model_chi2_reduced = []  ## stores reduced chi-squared statistic of the spline model fit to 
                                             ## spectral ratios


    ######################################################################## 
    ##                                                                    ##
    ##          COMPLETE -- Correct for Slitlosses General Setup          ##
    ##                                                                    ##
    ########################################################################
    

    # Loop through all target spectra
    for i in range(len(target_filenames)):
        spectrum_target_filename = data_path + '/' + target_filenames[i]
        logger.info('Working on spectrum %s: %s', i+1, spectrum_target_filename)

        
        # Load target spectrum
        target = ascii.read(spectrum_target_filename)
        x_qty_target = target['col1']  ## target x quantity
        y_qty_target = target['col2']  ## target y quantity
        error_target = target['col3']  ## target error in y quantity


        # Load reference spectrum
        reference_filename = data_path + '/' + reference_filenames[i]
        reference = ascii.read(reference_filename)
        x_qty_reference = reference['col1']  ## reference x quantity
        y_qty_reference = reference['col2']  ## reference y quantity
        error_reference = reference['col3']  ## reference error in y quantity


        y_qty_zero = np.zeros((len(x_qty_target)))  ## define the no flux line
    
        
        if interactive:        
            # Plot original target and reference spectra
            plot_title = spectrum_target_filename[spectrum_target_filename.rfind('/')+1:] + ': Before slitloss ' + \
                'correction'
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

            plt.plot(x_qty_target, y_qty_zero, color='k', linestyle='--', linewidth=2, label='No flux line')  
                                                    ## plot the line where no flux is received on the Y axis
            
            plt.legend(loc='best', fontsize=16)  ## assign legend
            
            plt.tick_params(axis='x', labelsize=16)  ## assign axes label size
            plt.tick_params(axis='y', labelsize=16)

            plt.show()
        
        
        # Calculate ratios of the target to reference spectra
        logger.info('Calculating ratios of the target to reference spectra.')
        
        # One-to-one
        if ratios_method == 'one_to_one':
            x_qty_spectral_ratios = x_qty_target
            y_qty_spectral_ratios = [y_qty_target[j] / y_qty_reference[j] for j in range(len(y_qty_target))]  
                                                                                ## calculate spectral ratios
            error_spectral_ratios = [error_mul_div_var(y_qty_spectral_ratios[j], \
                [y_qty_target[j], y_qty_reference[j]], [error_target[j], error_reference[j]]) \
                for j in range(len(y_qty_spectral_ratios))]  ## calculate error in spectral ratios
        
        # In spectral bins of the size specified by the user
        elif ratios_method == 'binned':
            spectral_windows_number, spectral_window_mins, spectral_window_maxs = generateSpectralWindows(x_qty_target, \
                x_qty_reference, spectral_bin_size)  ## generate spectral bins
            
            x_qty_target_window_means, y_qty_target_window_means, error_target_window_means = \
                computeSpectralWindowMeans(x_qty_target, y_qty_target, error_target, spectral_windows_number, \
                spectral_window_mins, spectral_window_maxs)  ## calculate mean y quantities of the spectral bins
            
            x_qty_reference_window_means, y_qty_reference_window_means, error_reference_window_values = \
                computeSpectralWindowMeans(x_qty_reference, y_qty_reference, error_reference, spectral_windows_number, \
                spectral_window_mins, spectral_window_maxs)
            
            x_qty_spectral_ratios = x_qty_target_window_means
            y_qty_spectral_ratios = [y_qty_target_window_means[j] / y_qty_reference_window_means[j] \
                for j in range(len(y_qty_target_window_means))]
            error_spectral_ratios = [error_mul_div_var(y_qty_spectral_ratios[j], \
                [y_qty_target_window_means[j], y_qty_reference_window_means[j]], \
                [error_target_window_means[j], error_reference_window_values[j]]) \
                for j in range(len(y_qty_spectral_ratios))]
        
        else:
            logger.warning('###################################################################################')
            logger.warning('###################################################################################')
            logger.warning('#                                                                                 #')
            logger.warning('#          WARNING in correctSlitlosses: Method for calculating spectral          #')
            logger.warning('#                   invalid or not specified by the user.                         #')
            logger.warning('#                                                                                 #')
            logger.warning('###################################################################################')
            logger.warning('###################################################################################\n')
            logger.info('Setting the method for calculating spectral ratios to the default "one_to_one".')
            ratios_method = 'one_to_one'
            
            x_qty_spectral_ratios = x_qty_target
            y_qty_spectral_ratios = [y_qty_target[j] / y_qty_reference[j] for j in range(len(y_qty_target))]
            error_spectral_ratios = [error_mul_div_var(y_qty_spectral_ratios[j], \
                [y_qty_target[j], y_qty_reference[j]], [error_target[j], error_reference[j]]) \
                for j in range(len(y_qty_spectral_ratios))]


        if interactive:
            # Plot spectral ratios
            plot_title = spectrum_target_filename[spectrum_target_filename.rfind('/')+1:] + ': Spectral ratios'
            plt.title(plot_title, fontsize=16)
            plt.xlabel('Spectral axis', fontsize=16)
            
            plt.plot(x_qty_spectral_ratios, y_qty_spectral_ratios, color='k', marker='o', \
                label='Target/reference spectra')  ## plot spectral ratios on the Y axis
            plt.fill_between(x_qty_spectral_ratios, np.array(y_qty_spectral_ratios) - np.array(error_spectral_ratios),\
                np.array(y_qty_spectral_ratios) + np.array(error_spectral_ratios), color='gray', alpha=0.5)  ## plot 
                                                                           ## error in spectral ratios on the Y axis

            plt.plot(x_qty_target, y_qty_zero, color='k', linestyle='--', linewidth=2, label='No flux line')  

            plt.legend(loc='best', fontsize=16)
            
            plt.tick_params(axis='x', labelsize=16)
            plt.tick_params(axis='y', labelsize=16)

            plt.show()
        
        
        # Save spectral ratios in an ASCII file
        spectral_ratios_filename = spectrum_target_filename[:spectrum_target_filename.rfind('.')] + '_ratios.dat'
        np.savetxt(spectral_ratios_filename, \
            np.c_[x_qty_spectral_ratios, y_qty_spectral_ratios, error_spectral_ratios]); \
            logger.info('Saved an ASCII file named %s.', spectral_ratios_filename)

        
        # Fit spline function to the spectral ratios
        logger.info('Fitting spline function to the spectral ratios.')
        
        spectral_ratios_model_filename = spectral_ratios_filename[:spectral_ratios_filename.rfind('.')] + '_model.dat'
        spectral_ratios_model_filenames.append(spectral_ratios_model_filename)  ## append spline model results filename
                                                                                ## to be saved to an ASCII file
        f = open(spectral_ratios_model_filename, 'w') 
        sys.stdout = f  ## write the terminal output from the spline fitting IRAF task to the file created in the 
                        ## previous command
        
        iraf.curfit(input=spectral_ratios_filename, function='spline3', order=spline_order, weighting='instrumental', \
            interactive='no', axis=1, listdata='no', verbose='yes', calctype='double', power='no', device='stdgraph', \
            cursor='', mode='stdout')
        # NOTE: The output model points of the <curfit> task are in columns: X_input, Y_calculated, Y_input, W_input,
        # where W_input are the input weights assigned to the error in Y_input.
        
        f.close()
        # NOTE: Writing the output of the IRAF task to a file adds random characters on the first line of the file.  
        # The random characters are later removed in the script while appending only the model points in a new text 
        # file with the same name (that replaces the one created with the above commands.)
        

        # Parse contents of the model results file
        model_points = []  ## stores lines of model points (from the model results file) as lists
        with open(spectral_ratios_model_filename, 'r') as model_file:
            model_results = model_file.readlines()
            
            # Delete the first line of the model results file that contains some random characters
            del model_results[0]

            # Append spline model results to be saved to an ASCII file
            model_rms = float(model_results[14].split()[3])
            spectral_ratios_model_rms.append(model_rms)
            logger.info('Spectral ratios model RMS: %s', model_rms)
            
            model_chi2_reduced = float(model_results[15].split()[8])**2
            spectral_ratios_model_chi2_reduced.append(model_chi2_reduced)
            logger.info('Spectral ratios model reduced chi-squared: %s', model_chi2_reduced)

            for result in model_results:
                if (result[0] != '#') and (result[0] != '\n'):
                    model_points.append(result.strip() + '\n')
                else:
                    pass
        
        with open(spectral_ratios_model_filename, 'w') as model_file:
            for line in model_points:
                model_file.write(line)
        
        
        # Load slitloss correction spline model
        spectral_ratios_model_filename = spectral_ratios_filename[:spectral_ratios_filename.rfind('.')] + '_model.dat'
        spectral_ratios_model = ascii.read(spectral_ratios_model_filename)
        x_qty_spectral_ratios_model = spectral_ratios_model['col1']
        y_qty_spectral_ratios_model = spectral_ratios_model['col2']
        

        if interactive:
            # Plot spectral ratios with the spline fit
            plot_title = spectrum_target_filename[spectrum_target_filename.rfind('/')+1:] + ': Spline fit to ' + \
                'spectral ratios'
            plt.title(plot_title, fontsize=16)
            plt.xlabel('Spectral axis', fontsize=16)
            plt.ylabel('Counts or count rate', fontsize=16)
            
            plt.plot(x_qty_spectral_ratios, y_qty_spectral_ratios, color='k', marker='o', \
                label='Target/reference spectra')
            plt.fill_between(x_qty_spectral_ratios, np.array(y_qty_spectral_ratios) - np.array(error_spectral_ratios),\
                np.array(y_qty_spectral_ratios) + np.array(error_spectral_ratios), color='gray', alpha=0.5)
            
            plt.plot(x_qty_spectral_ratios_model, y_qty_spectral_ratios_model, color='r', linestyle='-', linewidth=2, \
                label='Spline fit to ratios')  ## plot the spline fit to spectral ratios on the Y axis

            plt.plot(x_qty_target, y_qty_zero, color='k', linestyle='--', linewidth=2, label='Zero flux')

            plt.legend(loc='best', fontsize=16)
            
            plt.tick_params(axis='x', labelsize=16)
            plt.tick_params(axis='y', labelsize=16)

            plt.show()
        
        if ratios_method == 'binned':
            y_qty_spectral_ratios_model = cspline1d_eval(y_qty_spectral_ratios_model, x_qty_target, spectral_bin_size)
                                                      ## evaluate one-dimensional split fit at all target x quantities


        # Scale original target spectrum to correct for slitlosses
        logger.info('Scaling the original target spectrum to correct for slitlosses.')
        y_qty_target_lcorr = y_qty_target / y_qty_spectral_ratios_model
        error_target_lcorr = [error_mul_div_const(error_target[j], 1. / y_qty_spectral_ratios_model[j]) \
            for j in range(len(y_qty_target_lcorr))]
        

        if interactive:
            # Plot original and sliloss-corrected target spectra
            plot_title = spectrum_target_filename[spectrum_target_filename.rfind('/')+1:] + ': After slitloss ' + \
                'correction'
            plt.title(plot_title, fontsize=16)
            plt.xlabel('Spectral axis', fontsize=16)
            plt.ylabel('Counts or count rate', fontsize=16)
            
            plt.plot(x_qty_target, y_qty_target, color='k', linestyle='-', linewidth=2, label='Original target')
            plt.fill_between(x_qty_target, y_qty_target - error_target, y_qty_target + error_target, color='gray', \
                alpha=0.5)

            plt.plot(x_qty_reference, y_qty_reference, color='g', linestyle='-', linewidth=2, label='Reference')
            plt.fill_between(x_qty_reference, y_qty_reference - error_reference, y_qty_reference + error_reference, \
                color='g', alpha=0.5)
            
            plt.plot(x_qty_target, y_qty_target_lcorr, color='purple', linestyle='-', linewidth=2, \
                label='Slitloss-corrected target')  ## plot slitloss-corrected y quantities of the target on the Y axis
            plt.fill_between(x_qty_target, y_qty_target_lcorr - np.array(error_target_lcorr), y_qty_target_lcorr + \
                np.array(error_target_lcorr), color='purple', alpha=0.5)
            
            plt.legend(loc='best', fontsize=16)
            
            plt.tick_params(axis='x', labelsize=16)
            plt.tick_params(axis='y', labelsize=16)

            plt.show()


        # Save spectrum corrected for slitlosses in an ASCII file
        target_filename_lcorr = spectrum_target_filename[:spectrum_target_filename.rfind('.')] + \
            output_spectra_filename_suffix + '.dat'
        np.savetxt(target_filename_lcorr, np.c_[x_qty_target, y_qty_target_lcorr, error_target_lcorr]); \
            logger.info('Saved an ASCII file named %s.', target_filename_lcorr)
        
    
    # Save spline model results in an ASCII file
    spectral_ratios_model_filenames = [filename[filename.rfind('/')+1:] for filename in spectral_ratios_model_filenames]
    spectral_ratios_models_info_all = Table([spectral_ratios_model_filenames, spectral_ratios_model_rms, \
        spectral_ratios_model_chi2_reduced], names=['#model_filename', 'rms', 'chi2_reduced'])
    ascii.write(spectral_ratios_models_info_all, data_path + '/' + slitloss_model_results_filename, delimiter='\t', \
        overwrite=True); \
        logger.info('Saved an ASCII file named %s.\n', data_path + '/' + slitloss_model_results_filename)
    

    logger.info('#########################################################')
    logger.info('#                                                       #')
    logger.info('#          COMPLETE -- Correct for slit losses          #')
    logger.info('#                                                       #')
    logger.info('#########################################################\n')

    return


################################################################################################################
##                                                  ROUTINES                                                  ##
################################################################################################################

def generateSpectralWindows(x_target, x_reference, window_width):
    """
    Generates spectral windows of the x quantity with lower bounds, upper bounds, and the number of windows.

    INPUTS:
        - x_target - x quantity of the target spectrum; array (of floats)
        - x_reference - x quantity of the reference spectrum; array (of floats)
        - window_width - width of the spectral windows; float
    
    OUTPUTS:
        - windows_number - number of spectral windows of the x quantity; int
        - window_mins - lower bounds of the spectral windows; list (of floats)
        - window_maxs - upper bounds of the spectral windows; list (of floats)
    """
    logger = log.getLogger('generateSpectralWindows')

    logger.info('Generating spectral windows of the x quantity.')

    # Set the maximum of target and reference starting x quantities as the starting x quantity for computing spectral 
    # ratios
    x_target_min = np.min(x_target)
    x_reference_min = np.min(x_reference)
    x_start = np.max([x_target_min, x_reference_min])  
    # Set the minimum of target and reference ending x quantities as the ending x quantity for computing spectral 
    # ratios
    x_target_max = np.max(x_target)
    x_reference_max = np.max(x_reference)
    x_end = np.min([x_target_max, x_reference_max])  
    
    # Calculate the number of spectral windows of the x quantity in which the spectrum can be divided
    windows_number = int((x_end - x_start) / window_width)
    
    window_mins = []  ## stores minimum values of the spectral windows
    window_maxs = []  ## stores maximum values of the spectral windows
    for window_ctr in range(windows_number):
        if window_ctr == 0:
            window_mins.append(x_start)
        else:
            window_mins.append(window_mins[window_ctr-1] + window_width)
        window_maxs.append(window_mins[window_ctr] + window_width)
        window_ctr += 1

    return windows_number, window_mins, window_maxs


# ---------------------------------------------------------------------------------------------------------------------
def computeSpectralWindowMeans(x, y, err, windows_number, window_mins, window_maxs):
    """
    Calculates mean values of the x quantity windows -- mean x quantity, mean y quantity, and propagated error in the
    mean y quantity.

    INPUTS:
        - x - x quantity of the spectrum; array (of floats)
        - y - x quantity of the spectrum; array (of floats)
        - err - error in y quantity of the spectrum; array (of floats)
        - windows_number - number of spectral windows of the x quantity; int
        - window_mins - lower bounds of the spectral windows; list (of floats)
        - window_maxs - upper bounds of the spectral windows; list (of floats)
    
    OUTPUTS:
        - x_window_means - mean x quantities of the x quantity windows; list (of floats)
        - y_window_means - mean y quantities of the y quantity windows; list (of floats)
        - err_window_means - error in the mean y quantities in the windows; list (of floats)
    """
    logger = log.getLogger('computeSpectralWindowMeans')

    logger.info('Computing spectral window means.')
    
    x_window_means = []  ## stores mean x quantities 
    y_window_means = []  ## stores mean y quantities
    err_window_means = []  ## stores error in means of y quantities

    for window_ctr in range(windows_number):
        window_indices = np.where((x >= window_mins[window_ctr]) & (x <= window_maxs[window_ctr]))  ## store indices
                                                                               ## of the x quantities in the windows
        
        x_window_means.append(stats.mean(*[x[i] for i in window_indices]))  ## calculate mean x quantities of the
                                                                            ## windows
        y_window_means.append(stats.mean(*[y[i] for i in window_indices]))  ## calculate mean y quantities of the 
                                                                            ## windows
        err_window_values = list(err[window_indices])
        err_window_means.append(error_mul_div_const(error_add_sub(err_window_values), 1. / len(err_window_values)))
                                                             ## calculate error in mean y quantities of the windows
    
    return x_window_means, y_window_means, err_window_means


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Set log file
    log.configure('correctSlitlosses.log', filelevel='INFO', screenlevel='DEBUG')

    start('prepdataps.cfg')
