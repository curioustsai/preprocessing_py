#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Main function of Speech Enhancement
2020 - Richard Tsai

read data from input, enhance speech and write to output
"""
from scipy.io import wavfile
from SpeechEnhance import *
import numpy as np
import glob
import matplotlib.pyplot as plt
import argparse

debugger = {
    "cepstrumVAD": 1,
    "doa": 1,
    "noiseEst": 1,
    "beamformer": 1,
    "agc": 0,
}


def drawMask(mask, output, fftLen, frame_rate, outfile, color='gray'):
    plt.figure()
    plt.subplot(211)
    plt.imshow(np.flipud(np.transpose(mask)), cmap=color, aspect='auto')
    plt.subplot(212)
    plt.specgram(output, NFFT=fftLen, Fs=frame_rate, window=np.sqrt(np.hanning(fftLen)), noverlap=fftLen//2,
                 cmap='magma', scale='dB')
    plt.savefig(outfile)
    # plt.show()
    # plt.pause(0.01)
    plt.close('all')


def main(input_file, output_file):
    # initialize
    # TODO: only support 16kHz now, how about 48k configuration

    frame_length = 0.016  # reading file, every 16 ms
    [frame_rate, data] = wavfile.read(input_file)
    data_len, nchannel = data.shape
    sample_per_frame = int(frame_rate*frame_length)
    fftLen = 512
    half_fftLen = fftLen // 2 + 1

    # frame_stop = 12000
    # data_len = min(data_len, frame_stop * sample_per_frame)

    frame_cnt = 0
    total_frame = data_len // sample_per_frame
    output_buf = np.zeros([data_len, 3], dtype=np.int16)

    if debugger["cepstrumVAD"]:
        vad_buf = np.zeros(data_len, dtype=np.int16)

    if debugger["noiseEst"]:
        noise_est_buf = np.zeros([data_len, 2], dtype=np.int16)
        speech_bin_2d = np.zeros((total_frame, half_fftLen), dtype=np.uint8)
        noise_bin_2d = np.zeros((total_frame, half_fftLen), dtype=np.uint8)
        mag_2d = np.zeros((total_frame, half_fftLen), dtype=np.uint8)
        noiseMag_2d = np.zeros((total_frame, half_fftLen), dtype=np.uint8)

    if debugger["doa"]:
        doa_buf = np.zeros([data_len, 5], dtype=np.int16)
        polar_buf = np.zeros([total_frame, 2], dtype=np.float)

    if debugger["agc"]:
        agc_buf = np.zeros([data_len, 3], dtype=np.int16)

    spxEn = SpeechEnhance(frame_rate, sample_per_frame, fftLen, nchannel)

    while (frame_cnt+1)*sample_per_frame < data_len:
        data_in = data[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame]

        # (processed_mvdr, processed_adaptive_mvdr, processed_ns, processed_agc, inbeam, outbeam) = spxEn.process(data_in)
        (processed_mvdr, processed_ns, processed_agc, inbeam, outbeam) = spxEn.process(data_in)

        output_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 0] = data_in[:, spxEn.ref_channel]
        output_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 1] = processed_mvdr
        # output_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 2] = processed_adaptive_mvdr
        output_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 2] = processed_ns
        # output_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 4] = processed_agc
        # output_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 2] = processed_gev
        # output_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 3] = processed_mwf

        if debugger["doa"]:
            # doa_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 0] = \
            #     int(spxEn.SoundLocate.theta_pair[0] / 180 * 32767)
            # doa_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 1] = \
            #     int(spxEn.SoundLocate.theta_pair[1] / 180 * 32767)
            doa_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 0] = \
                int(inbeam * 16384)
            doa_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 1] = \
                int(outbeam * 16384)
            doa_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 2] = \
                int(spxEn.SoundLocate.theta)
            doa_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 3] = \
                int(spxEn.SoundLocate.angleCluster)
            doa_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 4] = \
                int(spxEn.SoundLocate.angleRetain[spxEn.SoundLocate.curBeamIdx])

            polar_buf[frame_cnt, 0] = spxEn.SoundLocate.angleRetain[spxEn.SoundLocate.curBeamIdx] / 180 * np.pi
            polar_buf[frame_cnt, 1] = spxEn.bfloss

        if debugger["cepstrumVAD"]:
            vad_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame] = spxEn.Cepstrum.vad * 16384

        if debugger["noiseEst"]:
            noise_est_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 0] = spxEn.SnrEst.speech_frame*16384
            noise_est_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 1] = spxEn.SnrEst.noise_frame*16384
            speech_bin_2d[frame_cnt, :] = spxEn.SnrEst.speech_bin*255
            noise_bin_2d[frame_cnt, :] = spxEn.SnrEst.noise_bin*255
            mag_2d[frame_cnt, :] = spxEn.MagSqs[0, :]
            noiseMag_2d[frame_cnt, :] = spxEn.Noises[0, :]

        # if debugger["agc"]:
        #     agc_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 0] = pka_fp
        #     agc_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 1] = g_fp*1000
        #     agc_buf[frame_cnt*sample_per_frame:(frame_cnt+1)*sample_per_frame, 2] = max_abs_y

        frame_cnt = frame_cnt + 1
        # print(frame_cnt)

    wavfile.write(output_file.replace('.wav', '_processed.wav'), frame_rate, output_buf)
    wavfile.write(output_file.replace('.wav', '_cep_vad.wav'), frame_rate, vad_buf)
    wavfile.write(output_file.replace('.wav', '_noise_est.wav'), frame_rate, noise_est_buf)
    wavfile.write(output_file.replace('.wav', '_doa.wav'), frame_rate, doa_buf)
    # wavfile.write(output_file.replace('.wav', '_agc.wav'), frame_rate, agc_buf)

    speech_outfile = output_file.replace('.wav', '_speech_bin.jpg')
    noise_outfile = output_file.replace('.wav', '_noise_bin.jpg')
    mag_outfile = output_file.replace('.wav', '_mag.jpg')
    noiseMag_outfile = output_file.replace('.wav', '_noiseMag.jpg')
    drawMask(speech_bin_2d, output_buf[:, 0], fftLen, frame_rate, speech_outfile)
    drawMask(noise_bin_2d, output_buf[:, 0], fftLen, frame_rate, noise_outfile)
    drawMask(mag_2d, output_buf[:, 0], fftLen, frame_rate, mag_outfile, color='magma')
    drawMask(noiseMag_2d, output_buf[:, 0], fftLen, frame_rate, noiseMag_outfile, color='magma')

    # plt.axes(projection='polar')
    # for i in range(0, total_frame):
    #     rad = polar_buf[i, 0]
    #     loss = polar_buf[i, 1]
    #     plt.polar(rad, loss, 'g.')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate tdoa")
    parser.add_argument("--input", default="./data/test.wav", help="input file")
    parser.add_argument("--output", default="test.wav", help="output file")

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    main(input_file, output_file)

