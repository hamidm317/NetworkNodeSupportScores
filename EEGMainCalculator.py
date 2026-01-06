import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import os

import Utils.GraphLearningUtils as GLU
from Utils import NoSCoUtils as NDS
from Utils import DataLoadUtils as DLU
from Utils.GSignalGen import normalize_shift
from Utils.RealDataUtils import EEG_Band_Extractor as EEG_BE

from Arguments.EEGCalcArgs import MainArgs as Args

_, Coords = GLU._load_coords()
W = GLU.DistWeightMat(Coords)
W = W / np.max(W)

_, A = GLU.SimpleSparsityGraphLearning(W, plotplot = True)

S = normalize_shift(A)

R = Args['R']
OutPathMain = Args['OutPath']
SAVE_EVERY = Args['SaveEvery']
Fs = Args['Fs']

Bands = Args['Band']
Events = Args['Event']

if Args['Dynamic']:

    WinLen = Args['WinLen']
    OverLap = Args['OverLap']

nosco_All = {}
Proc = 0

for Event in Events:

    MainData = DLU.LoadEEGMat(Event)
    IDs = [key for key in MainData.keys()]

    InTimes = np.arange(0, MainData[str(IDs[0])].shape[-1] - int(Fs * WinLen), int(Fs * WinLen * OverLap))

    for Band in Bands:

        print(f'Calculation of NSS locked to {Event} in {Band} Band begins ...')

        OutPath = OutPathMain + '\\' + Event + 'WinLen' + str(int(WinLen * 1000)) + 'OL' + str(int(OverLap * 100)) + '_Band' + Band + '.npy'

        if os.path.isfile(OutPath):

            print(f'NSS Data Locked to {Event} in {Band} Band with {OverLap} Overlap and {WinLen} Window Length Already Exists.')

        else:

            for ID in IDs:

                SID = str(ID)
                print(SID)

                Data2Calc = MainData[SID]
                
                if Args['ERP']:

                    Data2Calc = sps.zscore(np.mean(Data2Calc, axis = 0), axis = -1)

                    if Args['Dynamic']:

                        nosco_Score = []

                        for InTime in InTimes:

                            Data2W = np.mean(EEG_BE(Data2Calc[:, InTime : InTime + int(Fs * WinLen)], Band)[0], axis = 1)

                            nosco_Score.append(NDS.nosco_topological(Data2W, S = S, R = R, reduced_error_node = True)[0])

                    else:

                        Data2W = np.mean(EEG_BE(Data2Calc, Band)[0], axis = 1)

                        nosco_Score, _ = NDS.nosco_topological(Data2W, S = S, R = R, reduced_error_node = True)

                else:

                    print('Under Construction!')

                nosco_All[SID] = nosco_Score
                Proc = Proc + 1

                # print(f"[ok] {SID} → shape={np.array(nosco_Score).shape}")

                if Proc % SAVE_EVERY == 0:

                    # Save partial snapshot; single-file .npy with a Python dict (pickle-based)
                    np.save(OutPath, nosco_All, allow_pickle=True)
                    print(f"[checkpoint] saved {len(nosco_All)} subjects (up to {SID}) → {OutPath}")

            # Final save
            np.save(OutPath, nosco_All, allow_pickle=True)
            print(f"[done] saved {len(nosco_All)} subjects → {OutPath}")