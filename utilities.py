#!/usr/bin/env python
# -*- coding: utf-8 -*-


###############################################################
##                                                           ##
##          Python utility functions for prepdataps          ##
##                                                           ##
###############################################################

# Import required modules
import numpy as n
from math import sqrt
import statistics as stats
from astropy import constants as const
import log


################################################################################################################
##                                                  ROUTINES                                                  ##
################################################################################################################

def error_add_sub(error):
  """
  Computes error propogation for values that are added or subtracted.

  INPUTS:
    error - errors of the values that are added or subtracted; list (of floats)

  OUTPUTS:
    qty_error - error corresponding to the calculated quantity; float
  """
  logger = log.getLogger('astroutilities.error_add_sub')

  qty_error = sqrt(sum([err**2 for err in error]))

  return qty_error


# ---------------------------------------------------------------------------------------------------------------------
def error_mul_div_var(qty_value, values, error):
  """
  Computes error propogation for variable values that are multiplied or divided.

  INPUTS:
    qty_value - calculated value of the quantity; float
    values - values that are multiplied or divided; list (of floats)
    error - errors of the values; list (of floats)

  OUTPUTS:
    qty_error - error corresponding to the calculated quantity
  """
  logger = log.getLogger('astroutilities.error_mul_div_var')

  qty_error = qty_value * sqrt(sum([(error[i] / values[i])**2 for i in range(len(values))]))

  return qty_error


# ---------------------------------------------------------------------------------------------------------------------
def error_mul_div_const(error, factor):
  """
  Computes error propogation for variable and constant values that are multiplied or divided.

  INPUTS:
    error - propagated error of the variable values that are multiplied or divided; float
    factor - multiplicative factor of the variable values; float

  OUTPUTS:
    qty_error - error corresponding to the calculated quantity; float
  """
  logger = log.getLogger('astroutilities.error_mul_div_const')

  qty_error = abs(factor) * error

  return qty_error


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
  # The following two lines will test all functions in the module.
  # Run "python utilities.py -v" to see verbose output.  It is probably best to rely on this kind of testing once you 
  # believe the code to already be functional since it is a great method for building but difficult to work with during
  # development.
  import doctest
  doctest.testmod()
