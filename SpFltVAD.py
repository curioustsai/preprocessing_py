"""
Ceptstrum VAD

2020 - Richard Tsai

Detect human by Cepstrum vad
search frequency between CepFreqDn ~ CepFreqUp,
where human voice fundamental frequency (F0) are.

Default value:
CepFreqDn: 50Hz
CepFreqUp: 400Hz
"""

import numpy as np
import matplotlib.pyplot as plt


class SpFltVAD:
    def __init__(self, fftLen, sample_rate):
        self.fftlen = fftLen
        self.half_fftlen = fftLen // 2 + 1
        self.sample_rate = sample_rate
        self.spflt = 0
        self.spflt_all = 0
        self.fn = 0
        self.vad = 0

    def SpFltVADCal(self, X):
        self.fn = self.fn + 1
        alpha = 0.9

        bin_start = 1
        bin_1k = int(1000/self.sample_rate*self.fftlen)
        Xpow = np.real(X*np.conj(X))
        spflt_num = np.exp(sum(np.log(Xpow[bin_start:bin_1k]))/(bin_1k-bin_start+1))
        spflt_den = sum(Xpow[bin_start:bin_1k])/(bin_1k-bin_start+1)
        spflt = spflt_num / (spflt_den + 1e-12)
        self.spflt = alpha * self.spflt + (1-alpha)*spflt

        bin_4k = int(4000/self.sample_rate*self.fftlen)
        spflt_num = np.exp(sum(np.log(Xpow[bin_start:bin_4k]))/(bin_4k-bin_start+1))
        spflt_den = sum(Xpow[bin_start:bin_4k])/(bin_4k-bin_start+1)
        spflt_all = spflt_num / (spflt_den + 1e-12)
        self.spflt_all = alpha * self.spflt_all + (1-alpha)*spflt_all

        self.vad = 0
        if self.spflt > 0.10:
            self.vad = 1
        elif self.spflt_all > 0.02:
            self.vad = 1

        return self.vad

        # # debug
        # plt.clf()
        # plt.xlim([CepFreqDn, CepFreqUp])
        # plt.ylim([0, 0.2])
        # # plt.plot(range(0, self.fftlen), CepDataAll)
        # plt.plot(range(CepFreqDn, CepFreqUp+1), CepData_sm_max)
        # plt.title(self.fn)
        # plt.pause(0.01)

