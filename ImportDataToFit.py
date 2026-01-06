import mne
import os
import numpy as np
# import scipy.stats as sps

from Utils import DataLoadUtils as DLU
from Arguments import DataLoadArgs as Args

# IDs = Args.Mat2Py['IDs']
Root = Args.Mat2Py['Root']
Template = Args.Mat2Py['DirTmp']

Save_Path = Args.Mat2Py['SavePath']

SAVE_EVERY = Args.Mat2Py['SaveEvery']

Ti_max = Args.Mat2Py['TiM']
Ti_min = Args.Mat2Py['Tim']

Tr_max = Args.Mat2Py['TrM']
Tr_min = Args.Mat2Py['Trm']

NumTrials2Keep = Args.Mat2Py['NumTrials2Keep']

Channels = Args.Mat2Py['Channels']

IDs = DLU.ID_Ext(Root)

processed = 0

BigData = {}

for i, ID in enumerate(IDs):

    SID = str(ID)

    set_path = DLU.RootGen(Root, Template, SID)

    Raw_Data = DLU.LoadByMNE(set_path)

    TrialsToKeep = DLU.RanSequenceGen(NumTrials2Keep, Raw_Data.shape[0])

    T2, T1 = DLU.TimeInq((Ti_max, Ti_min), (Tr_max, Tr_min))

    Data2Save = Raw_Data[TrialsToKeep, :, T1 : T2]

    processed += 1

    BigData[SID] = Data2Save

    print(f"[ok] {SID} → shape={Data2Save.shape}")

    if processed % SAVE_EVERY == 0:
        # Save partial snapshot; single-file .npy with a Python dict (pickle-based)
        np.save(Save_Path, BigData, allow_pickle=True)
        print(f"[checkpoint] saved {len(BigData)} subjects → {Save_Path}")

# Final save
np.save(Save_Path, BigData, allow_pickle=True)
print(f"[done] saved {len(BigData)} subjects → {Save_Path}")