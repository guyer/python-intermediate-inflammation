"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: A 2D data array with inflammation data (each row
        contains measurements for a single patient across all days).
    :returns: An array of mean values of measurements for each day.
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: A 2D data array with inflammation data (each row
        contains measurements for a single patient across all days).
    :returns: An array of maximum values of measurements for each day.
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.

    :param data: A 2D data array with inflammation data (each row
        contains measurements for a single patient across all days).
    :returns: An array of minimum values of measurements for each day.
    """
    return np.min(data, axis=0)

def patient_normalize(data):
    """Normalize patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.

    Negative values are rounded to 0.
    """
    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')
    if not isinstance(data, np.ndarray):
        raise TypeError('Inflammation data must be a numpy ndarray')
    if len(data.shape) != 2:
        raise ValueError('Inflammation values must be a 2D array')
    max_data = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalized = data / max_data[:, np.newaxis]
    normalized[np.isnan(normalized)] = 0
    normalized[normalized < 0] = 0
    return normalized
