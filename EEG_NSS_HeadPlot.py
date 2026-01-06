import numpy as np
import matplotlib.pyplot as plt

from Utils import NoSCoUtils as NDS
import Utils.DataLoadUtils as DLU
import Utils.PlotUtils as PU
from Utils import RealDataUtils as RDU
import Utils.GraphLearningUtils as GLU

plt.style.use('seaborn-v0_8-paper')
# plt.style.use('default')

plt.rcParams["font.family"] = "Times New Roman"

############################################## Extracting Labels of Subjects (DEP/CTRL) ########################################

BehavioralData, Performance_data = RDU.ExperimentDataLoader()
SOI = RDU.AvailableSubjects()

Sub_G = [[], []] # first element is CTRL Group Members and the Second one the DEP Group

for i, sub_i in enumerate(SOI[0]):

    if BehavioralData['BDI'][sub_i] < 10:

        Sub_G[0].append([i, sub_i])

    # else:

    #     Sub_G[1].append([i, sub_i])

    elif BehavioralData['BDI'][sub_i] > 10:

        Sub_G[1].append([i, sub_i])

CTRL = np.array(Sub_G[0])[:, 0]
DEP = np.array(Sub_G[1])[:, 0]

GroupIDs = {'CTRL': CTRL,
            'DEP': DEP}
############################################### Extracting EEG Electrode Labels ###############################################

AE, Coords = GLU._load_coords()
W = GLU.DistWeightMat(Coords)
W = W / np.max(W)

###################################################### Assign Paramteres ######################################################

Event = 'Reward'
Band = 'Beta'
Group = 'CTRL'

######################################################## Load NSS Data ######################################################## 

Data_Ps = []
Data = DLU.LoadONIMat(Event, Novu = True, Band = Band)
Subs = [Sub for Sub in Data.keys()]
MeanData = np.mean(np.array([NDS.ADer(np.array(Data[Subs[Sub]])) for Sub in GroupIDs[Group]]), axis = 0)
Data_Ps.append(MeanData)
MeanData = np.mean(np.array([np.array(Data[Subs[Sub]]) for Sub in CTRL]), axis = 0)
Data_Ps.append(MeanData)

########################################################### PLOT IT! ##########################################################

for Data_P in Data_Ps: 
    
    # First Plot is Standard NSS and Second is LOG(Reconstruction ERROR) which seems to be more suitable for EEG Data
        
    fig, axs = plt.subplots(3, 4, figsize = (8, 6), dpi = 1200, layout = 'constrained')

    vmin = np.min(Data_P)
    vmax = np.max(Data_P)

    times = np.linspace(-1, 2, 1500)

    w_len = 200
    w_starts = np.arange(0, 1300, 100)

    colorbar = False

    for t in range(Data_P.shape[0] - 1):

        i = int(t / 4)
        j = np.mod(t, 4)

        ax = axs[i, j]

        if t == Data_P.shape[0] - 2:

            colorbar = True

        PU.plot_scalp_topography(ax = ax, ch_xy = Coords[:, :2], ch_values = Data_P[t], show_colorbar = colorbar, head_radius = 100, ears = True, n_grid = 200, vmin = vmin, vmax = vmax)

        ax.set_title(f'{np.round(times[int(w_starts[t])], 3)} - {np.round(times[int(w_starts[t] + w_len)], 3)}')

    fig.suptitle(f'NSS following {Event} - {Group} Group, {Band} Band', fontsize = 15)

    plt.show()