"""
Speech Enhancement class includes
1. Cepstrum VAD
2. Beamformer
3. Noise Reduction
4. Sound Location
5. Auto Gain Control

"""
import numpy as np
from CepstrumVAD import *
from ChannelSelector import *
from NoiseReduce import *
from Beamformer import *
from SoundLocate import *
from AutoGainControl import *

func_switch = {
    'CepstrumVAD': 1,
    'ChannelSelector': 0,
    'SnrEst': 1,
    'DOA': 1,
    'Beamformer': 1,
    'PostFilt': 1,
}


class SpeechEnhance:
    def __init__(self, fs, nframe, fftLen, nchannel):
        self.frame_cnt = 0
        self.fs = fs
        self.nframe = nframe
        self.nchannel = nchannel
        self.ref_channel = 0

        # 512-point fft
        self.fftLen = fftLen
        self.half_fftLen = self.fftLen // 2 + 1
        self.fftwin = np.hanning(self.fftLen+2)
        self.fftwin = np.sqrt(self.fftwin[1:fftLen+1])
        self.MagSqs = np.zeros((self.nchannel, self.half_fftLen), dtype=float)
        self.Noises = np.zeros((self.nchannel, self.half_fftLen), dtype=float)
        self.bfloss = 0

        # Cepstrum VAD
        CepFreqUp = int(fs/40)
        CepFreqDn = int(fs/400)
        self.Cepstrum = CepstrumVAD(self.fftLen, CepFreqUp, CepFreqDn)

        # ChannelSelector
        self.ChanSelector = ChannelSelector(self.fftLen, self.nchannel)

        # Noise Estimate
        self.SnrEst = NoiseReduce(self.fftLen, self.half_fftLen)
        self.PostFilt = NoiseReduce(self.fftLen, self.half_fftLen)

        # Beamformer
        self.Beamformer = Beamformer(self.half_fftLen, self.nchannel)
        self.SoundLocate = SoundLocate(self.fftLen, self.fs, self.nchannel)

        # AGC
        self.AutoGainControl = AutoGainControl(tpka_fp=32768/2, g_min=1/8, g_max=8)

        self.dataLast = np.zeros((self.nframe, self.nchannel), dtype=int)
        self.overlap = np.zeros(self.nframe, dtype=int)
        self.overlap_mvdr = np.zeros(self.nframe, dtype=int)
        self.overlap_adaptive_mvdr = np.zeros(self.nframe, dtype=int)
        self.overlap_gev = np.zeros(self.nframe, dtype=int)
        self.overlap_mwf = np.zeros(self.nframe, dtype=int)
        self.overlap_y_diff = np.zeros(self.nframe, dtype=int)

        # DC removal
        self.state_y = np.zeros(nchannel, dtype=np.float)
        self.state_x = np.zeros(nchannel, dtype=np.float)

    def process(self, data_in):
        self.frame_cnt = self.frame_cnt + 1
        nchannel = self.nchannel

        # 50% shift
        nshift = int(0.5*self.fftLen)

        # time domain dc removal for data_in of all channels
        dc_remove = np.zeros((nshift, nchannel), dtype=np.float)
        a = 0.98
        # H(z) = 1-z^-1/(1-az^-1)
        for i in range(0, nshift):
            dc_remove[i, :] = a * self.state_y + data_in[i, :] - self.state_x
            self.state_x = data_in[i, :]
            self.state_y = dc_remove[i, :]

        fft_data = np.concatenate((self.dataLast, dc_remove))
        self.dataLast = dc_remove

        # generate fft windows
        fftwin = self.fftwin
        fftwins = np.tile(fftwin, (nchannel, 1))
        fftwins = np.transpose(fftwins)

        fft_data = np.multiply(fft_data, fftwins)
        Zxx = np.zeros([nchannel, self.half_fftLen], dtype=complex)

        for i in range(0, nchannel):
            Zxx[i, :] = np.fft.rfft(fft_data[:, i], self.fftLen)
            Zxx[i, 0] = 1e-36
            Zxx[i, 1] = 1e-36

        # Cepstrum VAD, a rough VAD used for noise estimation
        Zxx_ref = Zxx[self.ref_channel, :]
        CepData_max, max_idx, pitch, cep_vad = self.Cepstrum.CepstrumCal(Zxx_ref)

        # # Dynamic update reference channel
        # for i in range(0, nchannel):
        #     self.MagSqs[i, :] = np.square(np.abs(Zxx[i, :]))
        #     # FIXME: SnrEst for respective channels
        #     self.Noises[i, :] = self.SnrEst.EstimateNoise(self.MagSqs[i, :], self.frame_cnt, cep_vad)
        #
        # self.ref_channel = self.ChanSelector.SelectChannel(self.MagSqs, self.Noises)

        self.MagSqs[0, :] = np.square(np.abs(Zxx[self.ref_channel, :]))
        self.Noises[0, :] = self.SnrEst.EstimateNoise(self.MagSqs[0, :], self.frame_cnt, cep_vad)

        # SNR VAD
        speech_frame, noise_frame, speech_bin, noise_bin, snr_max = self.SnrEst.SnrVAD()

        """
        Directional of Arrival, and beamforming
        """
        if nchannel >= 2:
            angle, energy = self.SoundLocate.FindDOA(Zxx)
            max_weight, angle_num, vad_num, angle_cluster, inbeam, outbeam, angleRetain = \
                self.SoundLocate.Cluster(angle, energy, speech_frame)

            # Beamformer
            noise_status = int((noise_frame == 1) and (cep_vad == 0))
            # update_noise = self.Beamformer.UpdateNoiseMatrix(Zxx, noise_status, noise_bin)
            update_noise = self.Beamformer.UpdateNoiseMatrix(Zxx, noise_status, self.SnrEst.spp)

            if snr_max > 15:
                speech_status = int((speech_frame == 1) and (cep_vad == 1)) and inbeam
            else:
                speech_status = int((speech_frame == 1)) and inbeam
                # speech_status = int((speech_frame == 1))

            update_speech = self.Beamformer.UpdateSpeechMatrix(Zxx, speech_status, speech_bin)
            self.Beamformer.UpdateSteeringVector(update_speech, update_noise)

            # self.Beamformer.UpdateAdaptiveMvdrFilter(update_speech, update_noise)
            # output_adaptive_mvdr = self.Beamformer.DoFilter(Zxx, 'adaptive_mvdr')

            self.Beamformer.UpdateMvdrFilter(update_speech, update_noise)
            output_mvdr = self.Beamformer.DoFilter(Zxx, 'mvdr')

            power_bf = 10*np.log10(np.sum(np.square(np.abs(Zxx[self.ref_channel, :]))))
            power_af = 10*np.log10(np.sum(np.square(np.abs(output_mvdr))))
            self.bfloss = power_af - power_bf

            # self.Beamformer.UpdateGevFilter(update_speech, update_noise)
            # output_gev = self.Beamformer.DoFilter(Zxx, 'gev')

            # self.Beamformer.UpdateMwfFilter(update_speech, update_noise)
            # output_mwf = self.Beamformer.DoFilter(Zxx, 'mwf')

            y_mvdr = np.fft.irfft(output_mvdr)
            y_mvdr = np.multiply(y_mvdr, fftwin)
            data_out_mvdr = y_mvdr[0:nshift]
            data_out_mvdr = data_out_mvdr + self.overlap_mvdr
            self.overlap_mvdr = y_mvdr[nshift:]
            data_out_mvdr = np.int16(data_out_mvdr)

            # y_adaptive_mvdr = np.fft.irfft(output_adaptive_mvdr)
            # y_adaptive_mvdr = np.multiply(y_adaptive_mvdr, fftwin)
            # data_out_adaptive_mvdr = y_adaptive_mvdr[0:nshift]
            # data_out_adaptive_mvdr = data_out_adaptive_mvdr + self.overlap_adaptive_mvdr
            # self.overlap_adaptive_mvdr = y_adaptive_mvdr[nshift:]
            # # normalized and format to int16
            # data_out_adaptive_mvdr = np.int16(data_out_adaptive_mvdr)

            # y_gev = np.fft.irfft(output_gev)
            # y_gev = np.multiply(y_gev, fftwin)
            # data_out_gev = y_gev[0:nshift]
            # data_out_gev = data_out_gev + self.overlap_gev
            # self.overlap_gev = y_gev[nshift:]
            # # normalized and format to int16
            # data_out_gev = np.int16(data_out_gev)
            #
            # y_mwf = np.fft.irfft(output_mwf)
            # y_mwf = np.multiply(y_mwf, fftwin)
            # data_out_mwf = y_mwf[0:nshift]
            # data_out_mwf = data_out_mwf + self.overlap_mwf
            # self.overlap_mwf = y_mwf[nshift:]
            # data_out_mwf = np.int16(data_out_mwf)

            # debug
            bfout = output_mvdr
            # bfout = output_adaptive_mvdr
            # difference = Zxx_ref - bfout
            # y_diff = np.fft.irfft(difference)
            # y_diff = np.multiply(y_diff, fftwin)
            # bf_difference = y_diff[0:nshift]
            # bf_difference = bf_difference + self.overlap_y_diff
            # self.overlap_y_diff = y_diff[nshift:]
            # bf_difference = np.int16(bf_difference)

            ns2_input = np.square(np.abs(bfout))
            ns_input = bfout
        else:
            ns2_input = np.square(np.abs(Zxx_ref))
            ns_input = Zxx_ref

        # Noise Reduction
        if func_switch['PostFilt']:
            self.PostFilt.EstimateNoise(ns2_input, self.frame_cnt, speech_status)
            self.PostFilt.SnrVAD()
            post_filter = self.PostFilt.WienerFilter(ns_input)
            irfft_input = post_filter
        else:
            if func_switch['Beamformer']:
                irfft_input = bfout
            else:
                irfft_input = Zxx_ref

        # irfft & overlap
        y = np.fft.irfft(irfft_input)
        y = np.multiply(y, fftwin)
        data_out_ns = y[0:nshift]
        data_out_ns = data_out_ns + self.overlap
        self.overlap = y[nshift:]
        # normalized and format to int16
        data_out_ns = np.int16(data_out_ns)

        spp_mean = np.mean(self.PostFilt.spp)
        data_agc, pka_fp, g_fp, max_abs_y = self.AutoGainControl.agc(data_out_ns, speech_status, spp_mean)

        if nchannel >= 2:
            # return data_out_mvdr, data_out_adaptive_mvdr, data_out_ns, data_agc, inbeam, outbeam
            return data_out_mvdr, data_out_ns, data_agc, inbeam, outbeam
        else:
            return data_out_ns, data_agc

