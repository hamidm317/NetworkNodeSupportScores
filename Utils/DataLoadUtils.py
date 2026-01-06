import numpy as np
import scipy.stats as sps
import os
import mne

from Utils.Constants import Dirs

def ID_Ext(Root):

    """
    
    Input:
    - Root -> A directory in which all the numerical names of subdirs would be sorted and added to a list

    Output:
    - Sorted IDs

    """

    IDs = []
    for name in os.listdir(Root):

        full_path = os.path.join(Root, name)

        if os.path.isdir(full_path) and name.isdigit():

            IDs.append(int(name))

    return sorted(IDs)

def RootGen(Root, Template, ID):

    """
    Outputs: Root + '\' + ID + '\' + ID + Template
    """

    return Root + '\\' + ID + '\\' + ID + Template

def LoadByMNE(set_path, ValidCoordsPath = r'D:\AIRLab_Research\To Write the Paper\Paper D\ch_coords.npy'):

    raw = mne.io.read_epochs_eeglab(set_path)
    data = raw.get_data()

    FCoords = np.load(ValidCoordsPath, allow_pickle =True).item()
    ValidChannels = [key for key in FCoords.keys()]
    t_chans = Map_Indices(raw.ch_names, ValidChannels)

    Data2Test = sps.zscore(data[:, t_chans, :], axis = -1)

    return Data2Test

def Map_Indices(V, A):
    """
    For each element in A, return the index of that element in V.
    If not found, return np.nan.

    Parameters
    ----------
    V : list/array-like
        Reference list/array (where we search).
    A : list/array-like
        Query list/array (what to look for in V).

    Returns
    -------
    indices : np.ndarray of shape (len(A),)
        Indices of A elements in V (int if found, np.nan if not).
    """
    # make dict {value -> index}
    idx_map = {v: i for i, v in enumerate(V)}
    result = []
    for a in A:
        if a in idx_map:
            result.append(idx_map[a])
        else:
            result.append(np.nan)
    return np.array(result)

def RanSequenceGen(Req, Max):

    if Max < Req:

        return np.arange(Max)
    
    else:

        return np.random.permutation(Max)[:Req]
    
def TimeInq(Tis, Trs, Fs = 500):

    assert Tis[0] <= Trs[0] and Tis[0] >= Trs[1], "Invalid Time Interval"
    assert Tis[1] <= Trs[0] and Tis[1] >= Trs[1], "Invalid Time Interval"

    T1 = int((Tis[0] - Trs[1]) * Fs)
    T2 = int((Tis[1] - Trs[1]) * Fs)

    return T1, T2

def LoadEEGMat(Event = 'Stimulus'):

    Root = Dirs.EEGMatDirs[Event]

    Data = np.load(Root, allow_pickle = True).item()

    return Data

def LoadONIMat(Event = 'Stimulus', Novu = False, **kwargs):

    options = {
        'Band': 'Theta',
        'WinLen': 0.4,
        'OverLap': 0.5,
    }

    options.update(kwargs)

    Band = options['Band']
    WinLen = options['WinLen']
    OL = options['OverLap']

    if Novu:
        
        Root = NovuDataDirGen(Event, Band, WinLen = WinLen, OverLap = OL)

    else:

        Root = Dirs.ONIMatDirs[Event]

    Data = np.load(Root, allow_pickle = True).item()

    return Data

def NovuDataDirGen(Event, Band = 'All', WinLen = 0.4, OverLap = 0.5, OutPathMain = Dirs.ONIMatDirs['OutPath']):

    return OutPathMain + '\\' + Event + 'WinLen' + str(int(WinLen * 1000)) + 'OL' + str(int(OverLap * 100)) + '_Band' + Band + '.npy'