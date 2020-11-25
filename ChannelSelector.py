"""
ChannelSelector
2020 - Richard Tsai

Select channel with higher SNR as reference channel
"""

import numpy as np


class ChannelSelector:
    def __init__(self, fftlen, nchannel):
        self.fn = 0
        self.refChan = 0
        self.fftlen = fftlen
        self.half_fftlen = fftlen // 2 + 1
        self.nchannel = nchannel
        self.nband = 2
        self.histogram = np.zeros(nchannel)
        self.initialized = 0

        self.MagSqsSmooth = np.zeros((nchannel, self.half_fftlen))

    def ChanSelHisto(self, SnrBand, RefSnrBand0, RefSnrBand1, histogram):
        if (SnrBand[0] > RefSnrBand0 + 3) or (SnrBand[1] > RefSnrBand1 + 3):
            histogram = histogram + 1
        if (SnrBand[0] < RefSnrBand0 - 3) or (SnrBand[1] < RefSnrBand1 - 3):
            histogram = histogram - 1
            histogram = max(histogram, 0)

        return histogram

    def SelectChannel(self, MagSqs, Noises):
        self.fn = self.fn + 1
        fftlen = self.fftlen
        nchannel = self.nchannel
        self.MagSqsSmooth = 0.6 * self.MagSqsSmooth + 0.4 * MagSqs

        f1 = int(50/16000 * fftlen)
        f2 = int(2000/16000 * fftlen)
        f3 = int(4000/16000 * fftlen)

        nband = self.nband
        MagBand = np.zeros((nchannel, nband), dtype=np.float)
        NoiseBand = np.zeros((nchannel, nband), dtype=np.float)
        SnrBand = np.zeros((nchannel, nband), dtype=np.float)

        MagSqsSmooth = self.MagSqsSmooth

        for i in range(0, nchannel):
            MagBand[i][0] = np.real(10*np.log10(sum(MagSqsSmooth[i][f1:f2]) + 1.0))
            MagBand[i][1] = np.real(10*np.log10(sum(MagSqsSmooth[i][f1:f3]) + 1.0))
            NoiseBand[i][0] = np.real(10*np.log10(sum(Noises[i][f1:f2]) + 1.0))
            NoiseBand[i][1] = np.real(10*np.log10(sum(Noises[i][f1:f3]) + 1.0))
            SnrBand[i][0] = MagBand[i][0] - NoiseBand[i][0]
            SnrBand[i][1] = MagBand[i][1] - NoiseBand[i][1]

        refChan = self.refChan
        RefSnrBand0 = SnrBand[refChan][0]
        RefSnrBand1 = SnrBand[refChan][1]

        maxBand0 = max(SnrBand[:][0])
        maxBand1 = max(SnrBand[:][1])

        if self.initialized == 0:
            snr_threshold = 6.0
            frame_threshold = 10
        else:
            snr_threshold = 10.0
            frame_threshold = 62

        refresh_channel = 0
        if maxBand0 > snr_threshold or maxBand1 > snr_threshold:
            for i in range(0, nchannel):
                self.histogram[i] = self.ChanSelHisto(SnrBand[i][:], RefSnrBand0, RefSnrBand1, self.histogram[i])

                if self.histogram[i] > frame_threshold:
                    refChan = i
                    refresh_channel = 1
                    self.initialized = 1
                    print("fn: {}, Change reference microphone to {}".format(self.fn, refChan))

            if refresh_channel == 1:
                self.histogram[:] = 0

        return refChan


