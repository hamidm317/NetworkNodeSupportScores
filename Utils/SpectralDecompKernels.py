import numpy as np
import pywt

from Utils.Constants import SpectralConstants
from Utils import KernelUtils as KU

def wavelet(data: np.ndarray, Fs, Band = 'All', wavelet = 'morl', return_freqs = False, **kwargs):

    # if str(data.shape[-1]) in SpectralConstants.WaveletParams['time_lims'].keys():

    #     TimeLims = SpectralConstants.WaveletParams['time_lims'][str(data.shape[-1])]

    # else:

    #     TimeLims = [0, data.shape[-1] / Fs]

    if 'Spectral_Res' in kwargs.keys():

        SpecRes = kwargs['Spectral_Res']

    else:

        SpecRes = SpectralConstants.WaveletParams['Spectral_Res']

    if type(Band) == str:

        WiPa = SpectralConstants.WaveletParams['widths_param'][wavelet][str(Fs)][Band]

    else:

        WiPa = KU.AssignWidthsParams(BandRanges = Band, Fs = Fs, Spectral_Res = SpecRes)
    
    options = {

        'widths_param': WiPa,
        # 'time_lims': TimeLims,
        'Spectral_Res': SpecRes

    }

    options.update(kwargs)

    widths_param = options['widths_param']
    widths = np.geomspace(widths_param[0], widths_param[1], num = options['Spectral_Res'])

    # time = np.linspace(options['time_lims'][0], options['time_lims'][1], data.shape[-1])

    assert data.ndim < 4 and data.ndim > 0, "Invalid data shape"

    CWTMat_Conn = []

    for _ in range(3 - data.ndim):

        data = np.reshape(data, (1,) + data.shape)

    [Trials, Channels, Length] = data.shape

    for Channel_Num in range(Channels):

        CWTMat = np.zeros((len(widths), Length))

        for i in range(Trials):

            cwtmatr, freqs = pywt.cwt(data[i, Channel_Num, :], widths, wavelet, sampling_period = 1 / Fs)
            
            CWTMat = CWTMat + 1 / Trials * cwtmatr

        CWTMat_Conn.append(CWTMat)

    outputs = []
    outputs.append(np.squeeze(np.array(CWTMat_Conn)))

    if return_freqs:

        outputs.append(freqs)

    return tuple(output for output in outputs)