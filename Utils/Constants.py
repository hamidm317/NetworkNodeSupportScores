class Dirs:

    EEGMatDirs = {

        'Stimulus': r'D:\AIRLab_Research\To Write the Paper\Paper D\Data\EEG_Data\TrStimulus\DenseEEG60NTrStim_20T_87S.npy',
        'Reward': r'D:\AIRLab_Research\To Write the Paper\Paper D\Data\EEG_Data\Reward\DenseEEG60NReward_20T_87S.npy',
        'Punishment': r'D:\AIRLab_Research\To Write the Paper\Paper D\Data\EEG_Data\Punishment\DenseEEG60NPunishment_20T_87S.npy',

    }

    ONIMatDirs = {

        'Stimulus': r'D:\AIRLab_Research\To Write the Paper\Paper D\Data\EEG_Data\TrStimulus\DenseEEG60NTrStim_20T_87S.npy',
        'Reward': r'D:\AIRLab_Research\To Write the Paper\Paper D\Data\ONIData\Reward60Trials87SubsWinLen400OL50RNE1.npy',
        'Punishment': r'D:\AIRLab_Research\To Write the Paper\Paper D\Data\ONIData\Punishment60Trials87SubsWinLen400OL50RNE1.npy',
        'OutPath': r'D:\AIRLab_Research\To Write the Paper\Paper D\Data\ONIData'
    }

    EEGAUXDirs = {

        'BehavioralData': r'E:\HWs\Msc\Research\Research\Depression Dataset\depression_rl_eeg\Depression PS Task\Scripts from Manuscript\Data_4_Import.xlsx',
        'PerformanceData': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Subjects_Behavioral_datas.csv',
        'AvailableSubjects': r'D:\AIRLab_Research\Data\BehavioralData\AvailableSubjects.npy',
    }

    

class SpectralConstants():

    BandsBounds = {

        'Delta': [0.5, 4],
        'Theta': [4, 8],
        'Alpha': [8, 12],
        'LowF': [4, 12],
        'Beta': [12, 30],
        'Gamma': [30, 50],
        'LowBeta': [12, 20],
        'HighBeta': [20, 30],
        'LowGamma': [30, 38],
        'MidGamma': [38, 44],
        'HighGamma': [44, 50],
        'All': [0.5, 50] # Keep 'All' the last key!

    }

    WaveletParams = {

        'wavelet': 'morl',
        'widths_param':{

            'morl': {

                '500': {

                    'All': [8, 1024],
                    'Delta': [128, 1024],
                    'Theta': [54, 128],
                    'NomTheta': [51, 80],
                    'Alpha': [32, 54],
                    'Beta': [13, 32],
                    'Gamma': [8, 14],
                    'LowBeta': [20, 32],
                    'HighBeta': [12, 20],
                    'LowGamma': [11, 14],
                    'MidGamma': [9.2, 11],
                    'HighGamma': [8, 9.2],
                    'LowF': [32, 128]
                },

                '1000': {

                    'All': [16, 2048],
                    'Delta': [256, 2048],
                    'Theta': [108, 256],
                    'Alpha': [64, 108],
                    'Beta': [26, 64],
                    'Gamma': [16, 28],
                    'LowBeta': [40, 64],
                    'HighBeta': [24, 40],
                    'LowGamma': [22, 28],
                    'MidGamma': [18.4, 22],
                    'HighGamma': [16, 18.4],
                    'LowF': [64, 256]
                },
            },

            'cmor': {

                '500': {

                    'All': [8, 1024],
                    'Delta': [128, 1024],
                    'Theta': [54, 128],
                    'Alpha': [32, 54],
                    'Beta': [13, 32],
                    'Gamma': [8, 14],
                    'LowBeta': [20, 32],
                    'HighBeta': [12, 20],
                    'LowGamma': [11, 14],
                    'MidGamma': [9.2, 11],
                    'HighGamma': [8, 9.2],
                    'LowF': [32, 128]
                },
            }

        },

        # 'time_lims':{

        #     '100': [0, 0.2],
        #     '200': [0, 0.4],
        #     '400': [-0.2, 0.6],
        #     '500': [-0.4, 0.6],
        #     '800': [-0.4, 1.2],
        #     'Default': 'No'

        # },

        'Spectral_Res': 20
    }