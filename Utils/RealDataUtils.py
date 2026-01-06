################## Initializations ##################

import numpy as np
from pandas import read_excel, read_csv

from Utils import SpectralDecompKernels as SDK
from Utils.Constants import Dirs

################## EEG Application ##################

def EEG_Band_Extractor(SignalTensor, Band = 'All', method = 'wavelet', Fs = 500):

    BE_Func = getattr(SDK, method)

    DecomposedData = BE_Func(SignalTensor, Fs, Band = Band)

    return DecomposedData

def ExperimentDataLoader():

    BehavioralData = read_excel(Dirs.EEGAUXDirs['BehavioralData'])
    Performance_data = read_csv(Dirs.EEGAUXDirs['PerformanceData'])

    return BehavioralData, Performance_data

def AvailableSubjects():

    SOI = np.load(Dirs.EEGAUXDirs['AvailableSubjects'])

    return SOI