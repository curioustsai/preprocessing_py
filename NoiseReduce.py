"""
Noise Estimation

2020 - Richard

Implement Unbiased MMSE-Based Noise Power Estimation with Low Complexity and Low Delay

"""

import numpy as np

# define constant
ax = 0.8            # noise output smoothing time constant(8)
axc = 1 - ax
ap = 0.9            # speech prob smoothing time constant (23)
apc = 1 - ap
psthr = 0.99        # threshold for smoothed speech probability [0.99] (24)
pnsaf = 0.01        # noise probability [0.01] (24)
pspri = 0.5         # prior speech probability [0.5] (18)
asnr = 15           # active SNR in dB [15] (18)
psini = 0.5         # initial speech probabilty [0.5] (23)
tavini = 0.064 * 3  # assumed speech absent time at start [64 ms]

xih1 = 31.6227766017  # speech-present SNR at asnr=15dB i.e. 10^(asnr/10)
xih1r = xih1/(1.0+xih1)
pfac = ((1.0-pspri) / pspri)*(1.0+xih1)  # p(noise)/p(speech) (18)


class NoiseReduce:
    def __init__(self, fftlen, half_fftlen):
        self.fftlen = fftlen
        self.Ypw = np.zeros(half_fftlen, dtype=float)
        self.Npw = np.zeros(half_fftlen, dtype=float)
        self.spp = 0.5 * np.ones(half_fftlen, dtype=float)
        self.noisyFrmCnt = 0

        # post filter
        self.last_Ypw = np.zeros(half_fftlen, dtype=float)
        self.last_Npw = np.zeros(half_fftlen, dtype=float)
        self.prior_snr = np.zeros(half_fftlen, dtype=float)
        self.post_snr = np.zeros(half_fftlen, dtype=float)

        # parameters
        self.snr_thrd_H = 7.0  # 6.0
        self.snr_thrd_L = 4.0  # 3.0

        self.g_min_db = -6
        self.g_min = np.power(10, (self.g_min_db/20.0))

        # final results
        self.speech_frame = 0
        self.noise_frame = 0
        self.speech_bin = np.zeros(half_fftlen)
        self.noise_bin = np.zeros(half_fftlen)

    def EstimateNoise(self, Power, FrameCnt, CepstrumVad):
        self.Ypw = 0.6 * self.Ypw + 0.4 * Power

        self.last_Npw = self.Npw

        if sum(Power) > 1e-5:
            self.noisyFrmCnt = self.noisyFrmCnt + 1

        if self.noisyFrmCnt < 16 or FrameCnt < 16:
            self.Npw = 0.7 * self.Npw + 0.3 * Power
        else:
            Npw = self.Npw

            # a-posterior speech presence prob (18)
            ph1y = 1.0 / (1.0 + pfac * np.exp(-1.0 * xih1r * (Power / Npw + 1e-16)))

            # smoothed speech presence prob (23)
            self.spp = ap * self.spp + apc * ph1y

            # limit ph1y (24)
            ph1y = np.minimum(ph1y, 1 - pnsaf * (self.spp > psthr))

            # estimated raw noise spectrum (22)
            noise_r = (1.0 - ph1y) * Power + ph1y * Npw

            # smooth the noise estimate (8)
            # noiseMag = ax * noiseMag + axc * noise_r
            if CepstrumVad == 0:
                Npw = ax * Npw + axc * noise_r
            else:
                ax_array = Npw < Power
                ax_array = np.maximum(ax_array, 0.8)
                ax_array = np.minimum(ax_array, 0.9)

                Npw = ax_array * Npw + (1 - ax_array) * noise_r

            self.Npw = Npw

        return self.Npw

    def SnrVAD(self):
        f1 = int(50/16000*self.fftlen)
        f2 = int(1000/16000*self.fftlen)
        f3 = int(2000/16000*self.fftlen)
        snr_thrd_H = self.snr_thrd_H
        snr_thrd_L = self.snr_thrd_L

        mag_band1 = np.real(10 * np.log10(sum(self.Ypw[f1:f3]) + 1.0))
        noise_band1 = np.real(10 * np.log10(sum(self.Npw[f1:f3]) + 1.0))
        snr_band1 = mag_band1 - noise_band1

        mag_band2 = np.real(10 * np.log10(sum(self.Ypw[f2:]) + 1.0))
        noise_band2 = np.real(10 * np.log10(sum(self.Npw[f2:]) + 1.0))
        snr_band2 = mag_band2 - noise_band2

        mag_all = np.real(10 * np.log10(sum(self.Ypw) + 1.0))
        noise_all = np.real(10 * np.log10(sum(self.Npw) + 1.0))
        snr_all = mag_all - noise_all

        snr_max = max(snr_band2, snr_all)

        speech_frame = 0
        if (snr_band1 > snr_thrd_H) or (snr_band2 > snr_thrd_H) or (snr_all > snr_thrd_H):
            speech_frame = 1

        noise_frame = 1
        if (snr_band1 > snr_thrd_L) or (snr_band2 > snr_thrd_L) or (snr_all > snr_thrd_L):
            noise_frame = 0

        mag_db = np.real(10 * np.log10(self.Ypw + 1.0))
        noise_db = np.real(10 * np.log10(self.Npw + 1.0))

        # threshold for post snr
        speech_bin = (mag_db - noise_db) > snr_thrd_H
        noise_bin = (mag_db - noise_db) < snr_thrd_L

        self.speech_bin = speech_bin
        self.noise_bin = noise_bin
        self.post_snr = mag_db - noise_db

        self.speech_frame = speech_frame
        self.noise_frame = noise_frame

        return speech_frame, noise_frame, speech_bin, noise_bin, snr_max

    # alpha*|Y(l-1)|^2 ./ |N(l-1)|^2 + (1-alpha)*|X(l)|^2./|N(l)|^2
    # alpha*prior_snr + (1-alpha)*post_snr
    def WienerFilter(self, X):
        alpha = 0.9  # alpha range [0.9, 0.98]
        g_min = self.g_min

        prior_snr = self.last_Ypw / (self.last_Npw + 1e-12)
        snr = alpha * prior_snr + (1 - alpha) * np.maximum(self.post_snr - 1.0, 0.0)
        gain = snr / (snr + 1.0)
        gain = np.clip(gain, g_min, 1.0)
        Y = gain * X
        self.last_Ypw = np.real(Y * np.conj(Y))

        return Y


