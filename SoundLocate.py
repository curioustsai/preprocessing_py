"""
Sound Locate module

2020 - Richard Tsai

GccPhat and Cluster

"""

import numpy as np
import matplotlib.pyplot as plt

# define constant
sound_speed = 343  # 343m/s
ANGLE_UNVAL = 1000


def save_data(data_buf, new_data, buf_len):
    """
    data buffer
    index 0, 1, 2, 3, ...
    newest              oldest
    """
    data_buf[1:] = data_buf[0:buf_len-1]
    data_buf[0] = new_data


class SoundLocate:
    def __init__(self, fftlen, fs, nchannel):
        # basic information
        c = sound_speed
        self.fn = 0
        self.fs = fs
        self.nchannel = nchannel
        self.fftlen = fftlen
        self.half_fftlen = fftlen // 2 + 1
        self.target_angle = 252

        self.mic0_location = np.array([2, 0, 0]) / 100
        self.mic1_location = np.array([-2, 0, 0]) / 100
        self.mic2_location = np.array([0, 5.23, 0]) / 100
        # self.mic0_location = np.array([4.5, 0, 0]) / 100
        # self.mic1_location = np.array([0, 4.5, 0]) / 100
        # self.mic2_location = np.array([-4.5, 0, 0]) / 100

        # DOA parameters initialization
        if self.nchannel == 3:
            self.num_mic_pair = 3
            self.theta = 0  # range from [0, 180] in degree
            self.phi = 0  # range from [-90, 90] in degree
            self.energy = 0
            self.basis = np.zeros([3, self.num_mic_pair])
            self.basis[:, 0] = self.mic1_location - self.mic0_location  # pair01
            self.basis[:, 1] = self.mic2_location - self.mic0_location  # pair02
            self.basis[:, 2] = self.mic2_location - self.mic1_location  # pair12

            # x, y, z
            basis01, basis02, basis12 = self.basis[:, 0], self.basis[:, 1], self.basis[:, 2]
            self.projection_mat = np.matmul(basis01[:, np.newaxis], basis01[np.newaxis, :]) + \
                                  np.matmul(basis02[:, np.newaxis], basis02[np.newaxis, :]) + \
                                  np.matmul(basis12[:, np.newaxis], basis12[np.newaxis, :])
            self.trace_mat = np.trace(self.projection_mat)
        elif self.nchannel == 2:
            self.num_mic_pair = 1
            self.theta = 0  # range from [0, 180] in degree
            self.energy = 0
            self.basis = np.zeros([3, self.num_mic_pair], dtype=np.int)
            self.basis[:, 0] = self.mic1_location - self.mic0_location  # pair01

        # initialization for gccphat
        self.theta_pair = np.zeros([self.num_mic_pair])
        self.num_interpl = 1
        self.num_angle_samples = 181  # 7
        self.cand_angle = np.zeros([self.num_angle_samples])
        self.mapper = np.zeros([self.num_angle_samples, self.num_mic_pair], dtype=int)
        self.xcorr = np.zeros([self.half_fftlen, self.num_mic_pair], dtype=np.complex)

        for i in range(0, self.num_mic_pair):
            mic_dist = np.sqrt(np.sum(self.basis[:, i]**2))

            for q in range(0, self.num_angle_samples):
                self.cand_angle[q] = q / (self.num_angle_samples - 1) * np.pi
                tdoa = np.round(self.num_interpl * mic_dist * fs / c * np.cos(self.cand_angle[q])) / self.num_interpl
                self.mapper[q, i] = int(self.num_interpl * tdoa + (tdoa < 0) * self.num_interpl * self.fftlen)

        # cluster parameters
        self.angleBufLen = 62  # 1s buffer for frame size 0.016s
        self.angleStep = 5
        self.vadBuf = np.zeros(self.angleBufLen, dtype=np.int)

        # pair TL
        self.angleCluster = 0
        self.angleBuf = np.zeros(self.angleBufLen, dtype=np.int)
        self.gccValueBuf = np.zeros(self.angleBufLen, dtype=np.float)

        # Remain angle
        self.curBeamIdx = 0
        self.angleRetainNum = 3
        self.angleReset = np.zeros(self.angleRetainNum, dtype=np.int)
        self.angleRetain = ANGLE_UNVAL*np.ones(self.angleRetainNum, dtype=np.int)
        self.sameSourceCnt = np.zeros(self.angleRetainNum, dtype=np.int)
        self.diffSourceCnt = np.zeros(self.angleRetainNum, dtype=np.int)
        self.angleRetainMaxTime = np.int(10/0.016)

        # Cluster parameters
        self.gccVal_thrd = 0.1  # 0.35
        self.angle_num_thrd = 20  # 10
        self.vad_num_thrd = 10
        self.weight_thrd = 0.1  # 0.5

    def GccPhat(self, X, Y, pair_id=0):
        # discrete implementation w/ interpolation points
        fftlen = self.fftlen
        half_fftlen = self.half_fftlen
        num_interpl = self.num_interpl
        mapper = self.mapper[:, pair_id]

        fftlen_interpl = fftlen * num_interpl
        half_fftlen_interpl = fftlen_interpl // 2 + 1

        xcorr = X*np.conj(Y)
        xcorr_mod = abs(xcorr) + 1e-12
        xcorr = xcorr / xcorr_mod
        self.xcorr[:, pair_id] = 0.9 * self.xcorr[:, pair_id] + 0.1 * xcorr

        xcorr_interpl = np.zeros(half_fftlen_interpl, dtype=np.complex)
        xcorr_interpl[0:half_fftlen] = self.xcorr[:, pair_id]

        ifft = np.fft.irfft(xcorr_interpl, fftlen_interpl)
        gphat = np.zeros(self.num_angle_samples)
        for q in range(0, self.num_angle_samples):
            idx = mapper[q]
            gphat[q] = ifft[idx]  # be aware of 1/N scale of IRFFT
        max_idx = np.argmax(gphat)

        source_angle_rad = self.cand_angle[max_idx]
        source_angle = source_angle_rad * 180 / np.pi
        energy = gphat[max_idx]

        return source_angle, energy

    def Projection(self, theta01, theta02, theta12):
        b = np.cos(theta01/180*np.pi) * self.basis[:, 0] + np.cos(theta02/180*np.pi) * self.basis[:, 1] + \
            np.cos(theta12/180*np.pi) * self.basis[:, 2]

        # iterate 20 times for convergence
        n = np.array([0, 0, 0.5])
        for i in range(0, 20):
            n = n - (np.matmul(n, self.projection_mat) - b) / self.trace_mat

        theta = np.arctan2(n[1], n[0]) * 180 / np.pi
        # map theta from [-pi, pi] -> [0, 2*pi]
        if theta < 0:
            theta = theta + 360
        phi = np.arctan2(n[2], np.sqrt(n[0]**2 + n[1]**2)) * 180 / np.pi

        return theta,  phi

    def FindDOA(self, X):
        self.fn = self.fn + 1

        if self.nchannel == 3:
            (angle1, gccVal1) = self.GccPhat(X[0, :], X[1, :], 0)
            (angle2, gccVal2) = self.GccPhat(X[0, :], X[2, :], 1)
            (angle3, gccVal3) = self.GccPhat(X[1, :], X[2, :], 2)

            self.theta_pair[0] = angle1
            self.theta_pair[1] = angle2

            self.theta, self.phi = self.Projection(angle1, angle2, angle3)
            self.energy = gccVal1
            # print("fn: {}, theta: {:3.2f}, phi: {:3.2f}, energy: {:3.4f}, energy: {:3.4f}".format(
            #     self.fn, self.theta, self.phi, gccVal1, gccVal2))

        elif self.nchannel == 2:
            (angle1, gccVal1) = self.GccPhat(X[0, :], X[1, :], 0)
            self.theta = angle1
            self.energy = gccVal1
            # print("fn: {}, theta: {:3.2f}, energy: {:3.4f}".format(self.fn, self.theta, gccVal1))

        return self.theta, self.energy

    def Cluster(self, angle, gccVal, vad):
        gccVal_thrd = self.gccVal_thrd
        angle_num_thrd = self.angle_num_thrd
        vad_num_thrd = self.vad_num_thrd
        weight_thrd = self.weight_thrd
        buflen = self.angleBufLen

        save_data(self.angleBuf, angle, buflen)
        save_data(self.gccValueBuf, gccVal, buflen)
        save_data(self.vadBuf, vad, buflen)

        # 1s
        (angle_cluster, max_weight, angle_num, vad_num, weightBuf) = self.ClusterAngle(gccVal_thrd)
        self.angleCluster = angle_cluster
        # print("fn: {}, angle cluster: {}".format(self.fn, angle_cluster))

        if max_weight > weight_thrd and angle_num > angle_num_thrd and vad_num > vad_num_thrd:
            self.ReplaceAngle(angle_cluster, angleDistThrd=10)

        if max_weight > 0.25:
            self.RetainAngle(weightBuf, weightThrd=0.25)

        curBeamIdx = self.curBeamIdx
        angleRetain = self.angleRetain[curBeamIdx]
        angleBuf = self.angleBuf

        # inbeam = self.InbeamDet(angleBuf, angleRetain)
        # outbeam = self.OutbeamDet(angleBuf, self.gccValueBuf, angleRetain, FrameDelay=0)

        inbeam = self.InbeamDet(angleBuf, self.target_angle)
        outbeam = self.OutbeamDet(angleBuf, self.gccValueBuf, self.target_angle, FrameDelay=0)

        return max_weight, angle_num, vad_num, angle_cluster, inbeam, outbeam, angleRetain

    """
    Cluster Angle

    out: 
        angle_cluster: angle with maximum weighting within observed buffer
    """
    def ClusterAngle(self, gccVal_Thrd):
        angleBuf = self.angleBuf
        gccValBuf = self.gccValueBuf
        vadBuf = self.vadBuf

        step = self.angleStep
        bufLen = self.angleBufLen  # 1 second buffer

        if self.nchannel > 2:
            angle_range = 360
        else:
            angle_range = 180

        weightBuf = np.zeros(angle_range + 2 * step, dtype=np.float)
        numBuf = np.zeros(angle_range + 2 * step, dtype=np.int)
        numBuf_vad = np.zeros(angle_range + 2 * step, dtype=np.int)

        for i in range(0, bufLen):
            angle = angleBuf[i]
            value = max(gccValBuf[i] - gccVal_Thrd, 0)
            weightBuf[angle:angle+2*step] = weightBuf[angle:angle+2*step] + value*value
            numBuf[angle:angle+2*step] = numBuf[angle:angle+2*step] + 1
            numBuf_vad[angle:angle+2*step] = numBuf_vad[angle:angle+2*step] + 1*(vadBuf[i] == 1)

        # circular buffer
        if 360 == angle_range:
            weightBuf[angle_range:angle_range+step] = weightBuf[angle_range:angle_range+step] + weightBuf[0:step]
            weightBuf[step:2*step] = weightBuf[step:2*step] + weightBuf[angle_range+step:]

            numBuf[angle_range:angle_range+step] = numBuf[angle_range:angle_range+step] + numBuf[0:step]
            numBuf[step:2*step] = numBuf[step:2*step] + numBuf[angle_range+step:]

            numBuf_vad[angle_range:angle_range+step] = numBuf_vad[angle_range:angle_range+step] + numBuf_vad[0:step]
            numBuf_vad[step:2*step] = numBuf_vad[step:2*step] + numBuf_vad[angle_range+step:]

        # plt.clf()
        # plt.subplot(311)
        # plt.ylim([0, 0.1])
        # plt.plot(weightBuf)
        #
        # plt.subplot(312)
        # plt.ylim([0, 50])
        # plt.plot(numBuf)
        #
        # plt.subplot(313)
        # plt.ylim([0, 30])
        # plt.plot(numBuf_vad)
        # plt.title("fn: "+str(self.fn))
        # plt.pause(0.001)

        max_weight = max(weightBuf[step: angle_range + step])
        max_index = weightBuf.tolist().index(max_weight)
        angle_cluster = max_index - step
        angle_num = numBuf[max_index]
        vad_num = numBuf_vad[max_index]

        if max_weight == 0:
            angle_cluster = ANGLE_UNVAL

        return angle_cluster, max_weight, angle_num, vad_num, weightBuf[step: angle_range + step]

    def ReplaceAngle(self, angleClustNew, angleDistThrd):
        angleDiff = np.zeros(self.angleRetainNum, dtype=np.int)
        for i in range(0, self.angleRetainNum):
            angleDiff[i] = abs(angleClustNew - self.angleRetain[i])

        angledist = min(angleDiff)
        minIndex = angleDiff.tolist().index(angledist)

        # update angle with the smallest difference
        if angledist < angleDistThrd:
            self.curBeamIdx = minIndex
            self.angleRetain[minIndex] = angleClustNew
        else:
            # replace angle with the minimum retain time
            minRetainTime = self.angleRetainMaxTime
            angleReplace = 0
            for i in range(0, self.angleRetainNum):
                if minRetainTime > self.sameSourceCnt[i] - self.diffSourceCnt[i]:
                    minRetainTime = self.sameSourceCnt[i] - self.diffSourceCnt[i]
                    angleReplace = i

            self.curBeamIdx = angleReplace
            self.angleRetain[angleReplace] = angleClustNew
            self.sameSourceCnt[angleReplace] = 0
            self.diffSourceCnt[angleReplace] = 0
            self.angleReset[angleReplace] = 0

        return

    def RetainAngle(self, weightBuf, weightThrd):
        for i in range(0, self.angleRetainNum):
            DestAngle = self.angleRetain[i]
            DiffSourceCnt = self.diffSourceCnt[i]
            SameSourceCnt = self.sameSourceCnt[i]

            if DestAngle != ANGLE_UNVAL:
                if self.nchannel == 3:
                    if DestAngle - 20 < 0:
                        weight = max(weightBuf[360+DestAngle-20:360])
                        weight = max(weight, max(weightBuf[0:DestAngle+20]))
                    elif DestAngle + 20 > 360:
                        weight = max(weightBuf[DestAngle - 20:360])
                        weight = max(weight, max(weightBuf[0:DestAngle+20-360]))
                    else:
                        weight = max(weightBuf[DestAngle-20:DestAngle+20])
                else:
                    AngleMin = max(DestAngle - 10, 0)
                    AngleMax = min(DestAngle + 10, 180)
                    weight = max(weightBuf[AngleMin:AngleMax])

                if weight > weightThrd:
                    DiffSourceCnt = 0
                    SameSourceCnt = SameSourceCnt + 1
                else:
                    DiffSourceCnt = DiffSourceCnt + 1

                threshold_min = np.int(0.160 / 0.016)  # 0.5s in frame
                SameSourceCnt = min(SameSourceCnt, self.angleRetainMaxTime)
                threshold = max(SameSourceCnt, threshold_min)

                if DiffSourceCnt > threshold:
                    DestAngle = ANGLE_UNVAL
                    DiffSourceCnt = 0
                    SameSourceCnt = 0

                self.angleRetain[i] = DestAngle
                self.diffSourceCnt[i] = DiffSourceCnt
                self.sameSourceCnt[i] = SameSourceCnt
        return

    def OutbeamDet(self, angleBuf, gccValueBuf, angleRetain, FrameDelay=0):
        angle = angleBuf[FrameDelay - 1]
        # angle = angleBuf[0]
        DeltAngle = abs(angle - angleRetain)
        # if DeltAngle > 180:
        #     DeltAngle = 360 - DeltAngle
        if DeltAngle < 0:
            DeltAngle = 360 + DeltAngle

        if DeltAngle > 60:
            outbeam = 1

            valid_count = 0
            valid_count2 = 0
            valid_count3 = 0
            for i in range(0, FrameDelay + 20):
                anglediff = min(abs(angleBuf[i] - angleRetain), 360 - abs(angleBuf[i] - angleRetain))
                if anglediff < 10:
                    if gccValueBuf[i] > 0.3:
                        valid_count = valid_count + 1
                    valid_count2 = valid_count2 + 1

                if gccValueBuf[i] > 0.3:
                    valid_count3 = valid_count3 + 1

            if (valid_count > 1) or (valid_count2 > 3) or (valid_count3 > 4):
                outbeam = 0
        else:
            outbeam = 0

        return outbeam

    def InbeamDet(self, angleBuf, angleRetain):
        if angleRetain == ANGLE_UNVAL:
            return 0

        delta_angle = min(abs(angleBuf[0] - angleRetain), 360 - abs(angleBuf[0] - angleRetain))

        inbeam = 0
        if delta_angle < 15:
            inbeam = 1

        return inbeam

