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


class CepstrumVAD:
    def __init__(self, fftLen, CepFreqUp, CepFreqDn):
        self.fftlen = fftLen
        self.half_fftlen = fftLen // 2 + 1
        self.CepFreqUp = CepFreqUp
        self.CepFreqDn = CepFreqDn
        self.pitchBufLen = 5
        self.CepIdxLen = 5
        self.CepData_sm = np.zeros(CepFreqUp - CepFreqDn + 1, dtype=float)
        self.CepIdxBuf = np.zeros(self.CepIdxLen, dtype=int)
        self.pitchBuf = np.zeros(self.pitchBufLen, dtype=int)
        self.pitch = 0
        self.fn = 0

        # final result
        self.vad = 0

    def CepstrumCal(self, X):
        self.fn = self.fn + 1
        # parameters
        step = 7
        alpha = 0.5
        CepFreqUp = self.CepFreqUp
        CepFreqDn = self.CepFreqDn
        CepMax_Thrd = 0.125  # 0.1
        CepMax_ThrdLow = 0.08  # 0.07

        Xpow = np.log10(X*np.conj(X) + 1e-12)
        powTmp = np.flipud(np.conjugate(Xpow[0:255]))
        Xpow_ext = np.concatenate([Xpow, powTmp])
        CepDataAll = np.fft.irfft(Xpow_ext, self.fftlen)
        # CepDataAll = np.fft.irfft(Xpow, self.fftlen)

        CepData = CepDataAll[CepFreqDn-1-step: CepFreqUp+step]  # Only search this range
        CepData_sm_max = np.zeros(CepFreqUp-CepFreqDn+1, dtype=float)

        # smooth cepstrum by shift 2*step
        for i in range(0, 2*step):
            CepData_sm = alpha*self.CepData_sm[0: CepFreqUp-CepFreqDn+1] + (1-alpha)*CepData[i: i+CepFreqUp-CepFreqDn+1]
            CepData_sm_max = np.maximum(CepData_sm_max, CepData_sm)

        # # debug
        # plt.clf()
        # plt.xlim([CepFreqDn, CepFreqUp])
        # plt.ylim([0, 0.2])
        # # plt.plot(range(0, self.fftlen), CepDataAll)
        # plt.plot(range(CepFreqDn, CepFreqUp+1), CepData_sm_max)
        # plt.title(self.fn)
        # plt.pause(0.01)

        self.CepData_sm = CepData_sm_max
        CepData_max = max(self.CepData_sm)
        max_idx = np.argmax(self.CepData_sm)

        idx_dist = np.zeros(self.CepIdxLen, dtype=int)
        for i in range(0, self.CepIdxLen):
            idx_dist[i] = abs(max_idx - self.CepIdxBuf[i]) < 5

        idx_dist_num = sum(idx_dist)

        self.CepIdxBuf[1:] = self.CepIdxBuf[0:self.CepIdxLen - 1]
        self.CepIdxBuf[0] = max_idx

        vad = 0
        pitch_dist = abs(max_idx - self.pitch)

        if CepData_max > CepMax_Thrd:  # by current frame
            vad = 1
            self.pitch = max_idx
        elif (pitch_dist < 15) and (self.pitch != 0):  # by dist from previous F0
            vad = 1
            if CepData_max > CepMax_ThrdLow:
                self.pitch = max_idx
        elif (idx_dist_num > 4) and (CepData_max > CepMax_Thrd):  # by CepIdxBuf
            vad = 1
            self.pitch = max_idx

        self.pitchBuf[1:] = self.pitchBuf[0:self.pitchBufLen - 1]
        if vad == 1:
            self.pitchBuf[0] = self.pitch
        else:
            self.pitchBuf[0] = 0

        pitch_available = sum(self.pitchBuf)
        if pitch_available == 0:
            self.pitch = 0

        self.vad = vad

        return CepData_max, max_idx, self.pitch, self.vad
