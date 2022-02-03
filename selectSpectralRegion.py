#!/usr/bin/env python
# -*- coding: utf-8 -*-

# NOTE: Include license!!!
# Author: Viraja Khatu


# ---------------------------------------------------------------------------------------------------------------------
# Import required modules
import numpy as np
from astropy.io import ascii
from configparser import ConfigParser
import os, log


# ---------------------------------------------------------------------------------------------------------------------
def start(configfile):
    """
    Selects a desired spectral region within specified x quantity bounds.  If the x quantity is wavelength, the spectra 
    can be in restframe or observed frame.

    INPUTS:
        - data_path - absolute path to the directory where the data are located; string
        - spectra_filenames - filenames of the input spectra; string (of strings separated by commas and no spaces).  
                              These must be ASCII files containing 3 columns: pixels or wavelength, counts or flux, and 
                              error in counts or flux.
        - input_x - units of input X axis; string.  Options are: "p" for pixels and "w" for wavelenth.
        - x_region_start - starting x quantity of desired spectral region; float
        - x_region_end - ending x quantity of desired spectral region; float
        - objects_z - redshifts of the input objects; string (of floats separated by commas and no spaces).  Default is 
                      an empty string.
        - spectra_frame - frame of spectra; string.  Options are: "restframe" and "observed".  This parameter is valid
                          only if the x quantity is wavelength.  Default is "restframe".
        - x_bounds_frame - frame of the x quantity bounds for desired spectral region; string.  Options are: 
                           "restframe" and "observed".  This parameter is valid only if the x quantity is wavelength.  
                           Default is "restframe".
        - output_spectra_filename_suffix - suffix of the spectral-region-selected spectra filenames; string.  Default 
                                           is "_s".

    OUTPUTS:
        - ASCII files containing columns: x quantity, y quantity, and error in y quantity (all quantities in the same 
          units as input spectra)
    """
    logger = log.getLogger('selectSpectralRegion.start')

    logger.info('#############################################################')
    logger.info('#                                                           #')
    logger.info('#          START -- Select desired spectral region          #')
    logger.info('#                                                           #')
    logger.info('#############################################################\n')


    ##################################################################### 
    ##                                                                 ##
    ##          START -- Select Spectral Region General Setup          ##
    ##                                                                 ##
    #####################################################################


    # Print absolute path to the current working directory
    working_dir_path = os.getcwd()
    logger.info('Current working directory: %s', working_dir_path)


    # Import config parameters
    logger.info('Importing configuration parameters from %s.\n', configfile)
    config = ConfigParser()
    config.optionxform = str  ## make options case-sensitive
    config.read(configfile)
    
    # Read selectSpectralRegion specific config
    data_path = config.get('selectSpectralRegion','data_path')
    spectra_filenames = config.get('selectSpectralRegion','spectra_filenames')
    input_x = config.get('selectSpectralRegion','input_x')
    x_region_start = config.getfloat('selectSpectralRegion','x_region_start')
    x_region_end = config.getfloat('selectSpectralRegion','x_region_end')
    object_z = config.get('selectSpectralRegion','object_z')
    spectra_frame = config.get('selectSpectralRegion','spectra_frame')
    x_bounds_frame = config.get('selectSpectralRegion','x_bounds_frame')
    output_spectra_filename_suffix = config.get('selectSpectralRegion','output_spectra_filename_suffix')


    # Check if the path to the data directory provided by the user
    if not data_path:
        logger.error('############################################################################################')
        logger.error('############################################################################################')
        logger.error('#                                                                                          #')
        logger.error('#          ERROR in selectSpectralRegion: Absolute path to the data directory not          #')
        logger.error('#                            available.  Exiting script.                                   #')
        logger.error('#                                                                                          #')
        logger.error('############################################################################################')
        logger.error('############################################################################################\n')      
        raise SystemExit
    else:
        logger.info('Absolute path to the data directory available.')
        logger.info('Data directory: %s', data_path)


    # Check if spectra provided by the user
    if not spectra_filenames:
        logger.error('###########################################################################################')
        logger.error('###########################################################################################')
        logger.error('#                                                                                         #')
        logger.error('#          ERROR in selectSpectralRegion: Spectra not detected.  Exiting script.          #')
        logger.error('#                                                                                         #')
        logger.error('###########################################################################################')
        logger.error('###########################################################################################\n')      
        raise SystemExit
    else:
        logger.info('Spectra detected.')
        spectra_filenames = [x for x in spectra_filenames.split(',')]  ## convert comma-separated string of filenames 
                                                                       ## into a list of strings
        logger.info('Spectra: %s', spectra_filenames)
    
    # Check if spectra available in the data directory
    spectra_not_available_ctr = 0

    for filename in spectra_filenames:
        filepath = data_path + '/' + filename
        if not os.path.exists(filepath):
            logger.info('Spectrum not available: %s', filepath)
            spectra_not_available_ctr += 1
    
    if spectra_not_available_ctr > 0:
        logger.error('#########################################################################################')
        logger.error('#########################################################################################')
        logger.error('#                                                                                       #')
        logger.error('#          ERROR in selectSpectralRegion: %s spectra not available in the data          #', \
            spectra_not_available_ctr)
        logger.error('#                              directory.  Exiting script.                              #')
        logger.error('#                                                                                       #')
        logger.error('#########################################################################################')
        logger.error('#########################################################################################\n')
        raise SystemExit
    
    else:
        logger.info('All spectra avialable.')


    # Check if input x-axis units provided by the user
    if not input_x:
        logger.error('#####################################################################################')
        logger.error('#####################################################################################')
        logger.error('#                                                                                   #')
        logger.error('#          ERROR in selectSpectralRegion: Input x-axis units not detected.          #')
        logger.error('#                                  Exiting script.                                  #')
        logger.error('#                                                                                   #')
        logger.error('#####################################################################################')
        logger.error('#####################################################################################\n')
        raise SystemExit
    
    elif (input_x == 'w') or (input_x == 'p'):
        logger.info('Input x-axis units detected.')
        logger.info('X axis units: %s', input_x)
    
    else:
        logger.error('#########################################################################################')
        logger.error('#########################################################################################')
        logger.error('#                                                                                       #')
        logger.error('#          ERROR in selectSpectralRegion: Invalid input x-axis units detected.          #')
        logger.error('#                                    Exiting script.                                    #')
        logger.error('#                                                                                       #')
        logger.error('#########################################################################################')
        logger.error('#########################################################################################\n')
        raise SystemExit
    

    # Check if x quantity bounds provided by the user
    if (not x_region_start) and (not x_region_end):
        logger.error('#############################################################################################')
        logger.error('#############################################################################################')
        logger.error('#                                                                                           #')
        logger.error('#          ERROR in selectSpectralRegion: Both starting and ending x quantities of          #')
        logger.error('#                 the desired spectral region not detected.  Exiting script.                #')
        logger.error('#                                                                                           #')
        logger.error('#############################################################################################')
        logger.error('#############################################################################################\n')
        raise SystemExit
    
    elif (not x_region_start) and x_region_end:
        logger.error('#######################################################################################')
        logger.error('#######################################################################################')
        logger.error('#                                                                                     #')
        logger.error('#          ERROR in selectSpectralRegion: Starting x quantity of the desired          #')
        logger.error('#                    spectral region not detected.  Exiting script.                   #')
        logger.error('#                                                                                     #')
        logger.error('#######################################################################################')
        logger.error('#######################################################################################\n')
        raise SystemExit
    
    elif x_region_start and (not x_region_end):
        logger.error('#####################################################################################')
        logger.error('#####################################################################################')
        logger.error('#                                                                                   #')
        logger.error('#          ERROR in selectSpectralRegion: Ending x quantity of the desired          #')
        logger.error('#                   spectral region not detected.  Exiting script.                  #')
        logger.error('#                                                                                   #')
        logger.error('#####################################################################################')
        logger.error('#####################################################################################\n')
        raise SystemExit
    
    else:
        logger.info('Both starting and ending x quantities of the desired spectral region detected.')
        logger.info('Starting x quantity: %s', x_region_start)
        logger.info('Ending x quantity: %s', x_region_end)


    # If input x quantity is wavelength, check if valid input frames provided for spectra and wavelength bounds
    if input_x == 'w':
        logger.info('Checking if valid input frames provided for spectra and wavelength bounds.')

        if x_bounds_frame == 'observed':
            logger.info('Wavelength bounds to be applied for desired spectral region detected in observed frame.')
            
            if spectra_frame == 'observed':
                logger.info('Spectra detected in observed frame.')
                # No conversion between frames required for wavelength bounds
                logger.info('No conversion between frames required for wavelength bounds.')

            elif spectra_frame == 'restframe':
                logger.info('Spectra detected in restframe.')

                # Notify the user that the wavelength bounds will be converted from observed to restframe
                logger.info('Wavelength bounds in observed frame will be converted to restframe.')

                # Check if z values provided by the user
                if not object_z:
                    logger.error('#########################################################################')
                    logger.error('#########################################################################')
                    logger.error('#                                                                       #')
                    logger.error('#          ERROR in selectSpectralRegion: Object redshifts not          #')
                    logger.error('#                       detected.  Exiting script.                      #')
                    logger.error('#                                                                       #')
                    logger.error('#########################################################################')
                    logger.error('#########################################################################\n')
                    raise SystemExit
                else:
                    logger.info('Object redshifts detected.')
                    object_z = [float(x) for x in object_z.split(',')]  ## convert comma-separated string of z values
                                                                        ## into a list of floats
                    logger.info('Object redshifts: %s', object_z)
                
                # Check if number of z values match with the number of spectra
                if len(object_z) < len(spectra_filenames):
                    difference = len(spectra_filenames) - len(object_z)
                    logger.error('################################################################################')
                    logger.error('################################################################################')
                    logger.error('#                                                                              #')
                    logger.error('#          ERROR in selectSpectralRegion: %s redshifts not available.          #')
                    logger.error('#                               Exiting script.                                #')
                    logger.error('#                                                                              #')
                    logger.error('################################################################################')
                    logger.error('################################################################################\n')
                    raise SystemExit
                
                elif len(object_z) > len(spectra_filenames):
                    difference = len(object_z) - len(spectra_filenames)
                    logger.error('#######################################################################')
                    logger.error('#######################################################################')
                    logger.error('#                                                                     #')
                    logger.error('#          ERROR in selectSpectralRegion: %s extra redshifts          #')
                    logger.error('#                      available.  Exiting script.                    #')
                    logger.error('#                                                                     #')
                    logger.error('#######################################################################')
                    logger.error('#######################################################################\n')
                    raise SystemExit

                else:
                    logger.info('Number of redshifts match with the number of spectra.')


            else:
                logger.warning('###################################################################################')
                logger.warning('###################################################################################')
                logger.warning('#                                                                                 #')
                logger.warning('#          WARNING in selectSpectralRegion: Invalid input frame detected          #')
                logger.warning('#                                  for spectra.                                   #')
                logger.warning('#                                                                                 #')
                logger.warning('###################################################################################')
                logger.warning('###################################################################################\n')
                logger.info('Setting spectra frame to the default "restframe".')
                spectra_frame = 'restframe'
                
                # Notify the user that the wavelength bounds will be converted from observed to restframe
                logger.info('Wavelength bounds in observed frame will be converted to restframe.')

                # Check if z values provided by the user
                if not object_z:
                    logger.error('#########################################################################')
                    logger.error('#########################################################################')
                    logger.error('#                                                                       #')
                    logger.error('#          ERROR in selectSpectralRegion: Object redshifts not          #')
                    logger.error('#                       detected.  Exiting script.                      #')
                    logger.error('#                                                                       #')
                    logger.error('#########################################################################')
                    logger.error('#########################################################################\n')
                    raise SystemExit
                else:
                    logger.info('Object redshifts detected.')
                    object_z = [float(x) for x in object_z.split(',')]  ## convert comma-separated string of z values
                                                                        ## into a list of floats
                    logger.info('Object redshifts: %s', object_z)
                
                # Check if number of z values match with the number of spectra
                if len(object_z) < len(spectra_filenames):
                    difference = len(spectra_filenames) - len(object_z)
                    logger.error('################################################################################')
                    logger.error('################################################################################')
                    logger.error('#                                                                              #')
                    logger.error('#          ERROR in selectSpectralRegion: %s redshifts not available.          #')
                    logger.error('#                               Exiting script.                                #')
                    logger.error('#                                                                              #')
                    logger.error('################################################################################')
                    logger.error('################################################################################\n')
                    raise SystemExit
                
                elif len(object_z) > len(spectra_filenames):
                    difference = len(object_z) - len(spectra_filenames)
                    logger.error('#######################################################################')
                    logger.error('#######################################################################')
                    logger.error('#                                                                     #')
                    logger.error('#          ERROR in selectSpectralRegion: %s extra redshifts          #')
                    logger.error('#                      available.  Exiting script.                    #')
                    logger.error('#                                                                     #')
                    logger.error('#######################################################################')
                    logger.error('#######################################################################\n')
                    raise SystemExit

                else:
                    logger.info('Number of redshifts match with number of spectra.')
        
        elif x_bounds_frame == 'restframe':
            logger.info('Wavelength bounds to be applied for desired spectral region detected in restframe.')
            
            if spectra_frame == 'observed':
                logger.info('Spectra detected in observed frame.')
                
                # Notify the user that the wavelength bounds will be converted from restframe to observed frame
                logger.info('Wavelength bounds in restframe will be converted to observed frame.')

                # Check if z values provided by the user
                if not object_z:
                    logger.error('#########################################################################')
                    logger.error('#########################################################################')
                    logger.error('#                                                                       #')
                    logger.error('#          ERROR in selectSpectralRegion: Object redshifts not          #')
                    logger.error('#                       detected.  Exiting script.                      #')
                    logger.error('#                                                                       #')
                    logger.error('#########################################################################')
                    logger.error('#########################################################################\n')
                    raise SystemExit
                else:
                    logger.info('Object redshifts detected.')
                    object_z = [float(x) for x in object_z.split(',')]  ## convert comma-separated string of z values
                                                                        ## into a list of floats
                    logger.info('Object redshifts: %s', object_z)
                
                # Check if number of z values match with the number of spectra
                if len(object_z) < len(spectra_filenames):
                    difference = len(spectra_filenames) - len(object_z)
                    logger.error('################################################################################')
                    logger.error('################################################################################')
                    logger.error('#                                                                              #')
                    logger.error('#          ERROR in selectSpectralRegion: %s redshifts not available.          #')
                    logger.error('#                               Exiting script.                                #')
                    logger.error('#                                                                              #')
                    logger.error('################################################################################')
                    logger.error('################################################################################\n')
                    raise SystemExit
                
                elif len(object_z) > len(spectra_filenames):
                    difference = len(object_z) - len(spectra_filenames)
                    logger.error('#######################################################################')
                    logger.error('#######################################################################')
                    logger.error('#                                                                     #')
                    logger.error('#          ERROR in selectSpectralRegion: %s extra redshifts          #')
                    logger.error('#                      available.  Exiting script.                    #')
                    logger.error('#                                                                     #')
                    logger.error('#######################################################################')
                    logger.error('#######################################################################\n')
                    raise SystemExit

                else:
                    logger.info('Number of redshifts mattch with the number of spectra.')
            
            elif spectra_frame == 'restframe':
                logger.info('Spectra detected in restframe.')
                # No conversion between frames required for wavelength bounds
                logger.info('No conversion between frames required for wavelength bounds.')

            else:
                logger.warning('###################################################################################')
                logger.warning('###################################################################################')
                logger.warning('#                                                                                 #')
                logger.warning('#          WARNING in selectSpectralRegion: Invalid input frame detected          #')
                logger.warning('#                                  for spectra.                                   #')
                logger.warning('#                                                                                 #')
                logger.warning('###################################################################################')
                logger.warning('###################################################################################\n')
                logger.info('Setting spectra frame to the default "restframe".')
                spectra_frame = 'restframe'
                # No conversion between frames required for wavelength bounds
                logger.info('No conversion between frames required for wavelength bounds.')

        else:
            logger.warning('#######################################################################################')
            logger.warning('#######################################################################################')
            logger.warning('#                                                                                     #')
            logger.warning('#          WARNING in selectSpectralRegion: Invalid input frame detected for          #')
            logger.warning('#                                 wavelength bounds.                                  #')
            logger.warning('#                                                                                     #')
            logger.warning('#######################################################################################')
            logger.warning('#######################################################################################\n')
            logger.info('Setting wavelength bounds frame to the default "restframe".')
            x_bounds_frame = 'restframe'

            if spectra_frame == 'observed':
                logger.info('Spectra detected in observed frame.')
                
                # Notify the user that the wavelength bounds will be converted from restframe to observed frame
                logger.info('Wavelength bounds in restframe will be converted to observed frame.')

                # Check if z values provided by the user
                if not object_z:
                    logger.error('#########################################################################')
                    logger.error('#########################################################################')
                    logger.error('#                                                                       #')
                    logger.error('#          ERROR in selectSpectralRegion: Object redshifts not          #')
                    logger.error('#                       detected.  Exiting script.                      #')
                    logger.error('#                                                                       #')
                    logger.error('#########################################################################')
                    logger.error('#########################################################################\n')
                    raise SystemExit
                else:
                    logger.info('Object redshifts detected.')
                    object_z = [float(x) for x in object_z.split(',')]  ## convert comma-separated string of z values
                                                                        ## into a list of floats
                    logger.info('Object redshifts: %s', object_z)
                
                # Check if number of z values match with the number of spectra
                if len(object_z) < len(spectra_filenames):
                    difference = len(spectra_filenames) - len(object_z)
                    logger.error('################################################################################')
                    logger.error('################################################################################')
                    logger.error('#                                                                              #')
                    logger.error('#          ERROR in selectSpectralRegion: %s redshifts not available.          #')
                    logger.error('#                               Exiting script.                                #')
                    logger.error('#                                                                              #')
                    logger.error('################################################################################')
                    logger.error('################################################################################\n')
                    raise SystemExit
                
                elif len(object_z) > len(spectra_filenames):
                    difference = len(object_z) - len(spectra_filenames)
                    logger.error('#######################################################################')
                    logger.error('#######################################################################')
                    logger.error('#                                                                     #')
                    logger.error('#          ERROR in selectSpectralRegion: %s extra redshifts          #')
                    logger.error('#                      available.  Exiting script.                    #')
                    logger.error('#                                                                     #')
                    logger.error('#######################################################################')
                    logger.error('#######################################################################\n')
                    raise SystemExit

                else:
                    logger.info('Number of redshifts match with the number of spectra.')
            
            elif spectra_frame == 'restframe':
                logger.info('Spectra detected in restframe.')
                # No conversion between frames required for wavelength bounds
                logger.info('No conversion between frames required for wavelength bounds.')

            else:
                logger.warning('###################################################################################')
                logger.warning('###################################################################################')
                logger.warning('#                                                                                 #')
                logger.warning('#          WARNING in selectSpectralRegion: Invalid input frame detected          #')
                logger.warning('#                                  for spectra.                                   #')
                logger.warning('#                                                                                 #')
                logger.warning('###################################################################################')
                logger.warning('###################################################################################\n')
                logger.info('Setting spectra frame to the default "restframe".')
                spectra_frame = 'restframe'
                # No conversion between frames required for wavelength bounds
                logger.info('No conversion between frames required for wavelength bounds.')
    
    else:
        pass
    

    # Check if output filename suffix provided by the user
    if not output_spectra_filename_suffix:
        logger.warning('###########################################################################################')
        logger.warning('###########################################################################################')
        logger.warning('#                                                                                         #')
        logger.warning('#          WARNING in selectSpectralRegion: Output filename suffix not detected.          #')
        logger.warning('#                                                                                         #')
        logger.warning('###########################################################################################')
        logger.warning('###########################################################################################\n')
        logger.info('Setting the output filename suffix to the default "_s".\n')
        output_spectra_filename_suffix = '_s'
    else:
        logger.info('Output filename suffix detected.')
        logger.info('Output filename suffix: %s\n', output_spectra_filename_suffix)


    ######################################################################## 
    ##                                                                    ##
    ##          COMPLETE -- Select Spectral Region General Setup          ##
    ##                                                                    ##
    ########################################################################


    # Loop through all spectra
    for i in range(len(spectra_filenames)):
        spectrum_filename = data_path + '/' + spectra_filenames[i]
        logger.info('Working on spectrum %s: %s', i+1, spectrum_filename)


        # Load spectrum
        spectrum = ascii.read(spectrum_filename)
        x_qty = spectrum['col1']  ## x quantity
        y_qty = spectrum['col2']  ## y quantity
        error = spectrum['col3']  ## error in y quantity


        # Determine desired spectral region if input x quantity is wavelength
        if input_x == 'w':
            if x_bounds_frame == 'observed':
                if spectra_frame == 'observed':
                    # Skip conversion between frames
                    logger.info('Skipping conversion between frames.')
                    x_qty_region_start_calc = x_region_start  ## starting wavelength
                    x_qty_region_end_calc = x_region_end  ## ending wavelength
                
                elif spectra_frame == 'restframe':
                    # Calculate restframe wavelength bounds (in angstroms) of desired spectral region
                    logger.info('Calculating restframe wavelength bounds (in angstroms) of desired spectral region.')
                    x_qty_region_start_calc = x_region_start / (object_z[i] + 1)  ## calculate starting wavelength
                    x_qty_region_end_calc = x_region_end / (object_z[i] + 1)  ## calculate ending wavelength

                else:
                    pass
            
            elif x_bounds_frame == 'restframe':
                if spectra_frame == 'observed':
                    # Calculate observed wavelength bounds (in angstroms) of desired spectral region
                    logger.info('Calculating observed wavelength bounds (in angstroms) of desired spectral region.')
                    x_qty_region_start_calc = x_region_start * (object_z[i] + 1)
                    x_qty_region_end_calc = x_region_end * (object_z[i] + 1)
                
                elif spectra_frame == 'restframe':
                    # Skip conversion between frames
                    logger.info('Skipping conversion between frames.')
                    x_qty_region_start_calc = x_region_start
                    x_qty_region_end_calc = x_region_end

                else:
                    pass

            else:
                pass
        
        else:
            x_qty_region_start_calc = x_region_start  ## starting pixel
            x_qty_region_end_calc = x_region_end  ## ending pixel
        

        # Select desired spectral region
        if input_x == 'w':
            logger.info('Desired spectral region in %s frame: [%s:%s] (same units as input spectra)', spectra_frame, \
                x_qty_region_start_calc, x_qty_region_end_calc)
        else:
            logger.info('Desired spectral region: [%s:%s]', x_qty_region_start_calc, x_qty_region_end_calc)
        
        logger.info('Selecting desired spectral region.')
        x_qty_region_indices = np.where((x_qty_region_start_calc <= x_qty) & (x_qty <= x_qty_region_end_calc))  
                                              ## stores indices of the x quantities in desired spectral region
        x_qty_region = x_qty[x_qty_region_indices]  ## stores x quantities of selected spectral region
        y_qty_region = y_qty[x_qty_region_indices]  ## stores y quantities of selected spectral region
        error_region = error[x_qty_region_indices]  ## stores error in y quantities of selected spectral region


        # Save trimmed spectrum in an ASCII file
        spectrum_filename_trimmed = spectrum_filename[:spectrum_filename.rfind('.')] + \
            output_spectra_filename_suffix + '.dat'
        np.savetxt(spectrum_filename_trimmed, np.c_[x_qty_region, y_qty_region, error_region]); \
            logger.info('Saved an ASCII file named %s.\n', spectrum_filename_trimmed)
    

    logger.info('################################################################')
    logger.info('#                                                              #')
    logger.info('#          COMPLETE -- Select desired spectral region          #')
    logger.info('#                                                              #')
    logger.info('################################################################\n')

    return


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Set log file
    log.configure('selectSpectralRegion.log', filelevel='INFO', screenlevel='DEBUG')

    start('prepdataps.cfg')
