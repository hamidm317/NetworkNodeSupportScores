import numpy as np
import matplotlib.pyplot as plt


from Utils import NoSCoUtils as NDS
import Utils.DataLoadUtils as DLU
import Utils.PlotUtils as PU
from Utils import RealDataUtils as RDU
import Utils.GraphLearningUtils as GLU

plt.style.use('seaborn-v0_8-paper')
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

############################################### Extracting EEG Electrode Labels ###############################################

AE, Coords = GLU._load_coords()
W = GLU.DistWeightMat(Coords)
W = W / np.max(W)

#################################################### Assign Band Parameter ####################################################

Band = 'Alpha'

######################################################## Load NSS Data ######################################################## 

Event = 'Punishment'
Data = DLU.LoadONIMat(Event, Novu = True, Band = Band)
Subs = [Sub for Sub in Data.keys()]

Data_CTRL_P = np.array([NDS.ADer(np.array(Data[Subs[Sub]])) for Sub in CTRL])
Data_DEP_P = np.array([NDS.ADer(np.array(Data[Subs[Sub]])) for Sub in DEP])

Data_CTRL_P_Raw = np.array([np.array(Data[Subs[Sub]]) for Sub in CTRL])
Data_DEP_P_Raw = np.array([np.array(Data[Subs[Sub]]) for Sub in DEP])

Event = 'Reward'
Data = DLU.LoadONIMat(Event, Novu = True, Band = Band)
Subs = [Sub for Sub in Data.keys()]

Data_CTRL_R = np.array([NDS.ADer(np.array(Data[Subs[Sub]])) for Sub in CTRL])
Data_DEP_R = np.array([NDS.ADer(np.array(Data[Subs[Sub]])) for Sub in DEP])

Data_CTRL_R_Raw = np.array([np.array(Data[Subs[Sub]]) for Sub in CTRL])
Data_DEP_R_Raw = np.array([np.array(Data[Subs[Sub]]) for Sub in DEP])

time_nss = np.linspace(-0.6, 2, 13)

Chan = 9
CL = 0.63

########################################################### PLOT IT! ##########################################################
# # # # First Plot is Standard NSS and Second is LOG(Reconstruction ERROR) which seems to be more suitable for EEG Data
######################################################### First Plot ##########################################################

fig, axs = plt.subplots(1, 2, dpi = 1200, figsize = (6, 2), sharey = True, sharex = True, layout = 'constrained')

ax = axs[0]

TD = PU.mean_confidence_interval(Data_CTRL_R[:, :, Chan], CoLe = CL)
ax.plot(time_nss, TD[0], label = 'Reward')
ax.fill_between(time_nss, TD[1], TD[2], alpha = 0.4)

TD = PU.mean_confidence_interval(Data_CTRL_P[:, :, Chan], CoLe = CL)
ax.plot(time_nss, TD[0], label = 'Punishment')
ax.fill_between(time_nss, TD[1], TD[2], alpha = 0.4)

# ax.legend(frameon = False)
ax.autoscale(axis = 'x', tight = True)
ax.set_title('CTRL')

ax = axs[1]

TD = PU.mean_confidence_interval(Data_DEP_R[:, :, Chan], CoLe = CL)
ax.plot(time_nss, TD[0], label = 'Reward')
ax.fill_between(time_nss, TD[1], TD[2], alpha = 0.4)

TD = PU.mean_confidence_interval(Data_DEP_P[:, :, Chan], CoLe = CL)
ax.plot(time_nss, TD[0], label = 'Punishment')
ax.fill_between(time_nss, TD[1], TD[2], alpha = 0.4)

ax.legend(frameon = False)
ax.autoscale(axis = 'x', tight = True)
ax.set_title('DEP')

for ax in axs:

    ax.axvline(0, color = 'k', ls = '--', lw = 0.5)

fig.suptitle(f'NSS Values of Channel {AE[Chan]}')
plt.show()

######################################################### Second Plot ##########################################################

fig, axs = plt.subplots(1, 2, dpi = 1200, figsize = (6, 2), sharey = True, sharex = True, layout = 'constrained')

ax = axs[0]

TD = PU.mean_confidence_interval(Data_CTRL_R_Raw[:, :, Chan], CoLe = CL)
ax.plot(time_nss, TD[0], label = 'Reward')
ax.fill_between(time_nss, TD[1], TD[2], alpha = 0.4)

TD = PU.mean_confidence_interval(Data_CTRL_P_Raw[:, :, Chan], CoLe = CL)
ax.plot(time_nss, TD[0], label = 'Punishment')
ax.fill_between(time_nss, TD[1], TD[2], alpha = 0.4)

# ax.legend(frameon = False)
ax.autoscale(axis = 'x', tight = True)
ax.set_title('CTRL')

ax = axs[1]

TD = PU.mean_confidence_interval(Data_DEP_R_Raw[:, :, Chan], CoLe = CL)
ax.plot(time_nss, TD[0], label = 'Reward')
ax.fill_between(time_nss, TD[1], TD[2], alpha = 0.4)

TD = PU.mean_confidence_interval(Data_DEP_P_Raw[:, :, Chan], CoLe = CL)
ax.plot(time_nss, TD[0], label = 'Punishment')
ax.fill_between(time_nss, TD[1], TD[2], alpha = 0.4)

ax.legend(frameon = False)
ax.autoscale(axis = 'x', tight = True)
ax.set_title('DEP')

for ax in axs:

    ax.axvline(0, color = 'k', ls = '--', lw = 0.5)

fig.suptitle(f'NSS Values of Channel {AE[Chan]}')