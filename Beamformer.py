"""
Beamformer

2020 - Richard Tsai

1) MVDR
2) GEV
3) MCWF
4) Auxiliary Vector MVDR

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.linalg import eig


def ldlh_update(L, D, x, n):
    """
    A = (1-alpha)*A + alpha * x * x^H
    """
    # a = 0.1
    a = 1
    for k in range(0, n):
        # d = (1 - a) * D[k, k] + a * x[k, 0] * x[k, 0].conj()
        d = D[k, k] + a * x[k, 0] * x[k, 0].conj()
        b = x[k, 0] * a / d
        # a = (1 - a) * D[k, k] * a / d
        a = D[k, k] * a / d
        D[k, k] = d

        for r in range(k+1, n):
            x[r, 0] = x[r, 0] - x[k, 0] * L[r, k]
            L[r, k] = L[r, k] + b.conj() * x[r, 0]

    return L, D


def forward_substitution(L, b, n):
    """
    Lx = b, solve x
    """
    x = np.zeros((n, n), dtype=complex)
    for j in range(0, n):
        for i in range(j, n):
            x[i, j] = b[i, j] / L[i, i]
            # b[i+1:n, j] = b[i+1:n, j] - L[i+1:n, i] * x[i, j]
            for k in range(i+1, n):
                b[k, j] = b[k, j] - L[k, i] * x[i, j]

    return x


class Beamformer:
    def __init__(self, half_fftlen, nchannel):
        self.fn = 0
        self.half_fftlen = half_fftlen
        self.nchannel = nchannel
        self.speechRyy = np.zeros((half_fftlen, nchannel, nchannel), dtype=np.complex)
        self.noiseRvv = np.zeros((half_fftlen, nchannel, nchannel), dtype=np.complex)

        for f in range(0, half_fftlen):
            self.speechRyy[f][:][:] = np.eye(nchannel, nchannel) * 4096
            self.noiseRvv[f][:][:] = np.eye(nchannel, nchannel) * 4096

        self.speechRyy_cnt = 0
        self.noiseRvv_cnt = 0
        self.steering = np.ones((half_fftlen, nchannel), dtype=np.complex) / np.sqrt(nchannel)

        # MVDR
        self.mvdr_coef = np.ones((half_fftlen, nchannel), dtype=np.complex) / np.sqrt(nchannel)

        # GEV
        self.gev_coef = np.ones((half_fftlen, nchannel), dtype=np.complex) / np.sqrt(nchannel)

        # PMWF
        self.mwf_coef = np.ones((half_fftlen, nchannel), dtype=np.complex) / np.sqrt(nchannel)

        # Adaptive MVDR
        self.adaptive_mvdr_coef = np.ones((half_fftlen, nchannel), dtype=np.complex) / np.sqrt(nchannel)

    def UpdateSpeechMatrix(self, X, speech_status, speech_bin):
        pow1 = sum(np.abs(X[0, :]))
        update_speech = 0
        self.fn = self.fn + 1

        # if (speech_status == 1) and (pow1 > 0):
        if ((speech_status == 1) and (pow1 > 0)) or self.fn <= 20:
            self.speechRyy_cnt = self.speechRyy_cnt + 1
            speechRyy_cnt = self.speechRyy_cnt

            if speechRyy_cnt < 100:
                alpha = 0.5
            else:
                alpha = 0.95

            X_freq = np.zeros([self.nchannel, 1], dtype=np.complex)
            for i in range(0, self.half_fftlen):
                X_freq[:, 0] = X[:, i]
                Rxx = np.dot(X_freq, X_freq.conj().T)
                self.speechRyy[i, :, :] = alpha * self.speechRyy[i, :, :] + (1 - alpha) * Rxx

            if (speechRyy_cnt < 100) or (speechRyy_cnt % 10 == 0):
                update_speech = 1

        return update_speech

    def UpdateNoiseMatrix(self, X, noise_status, noise_bin):
        pow1 = sum(np.abs(X[0, :]))
        update_noise = 0

        if (noise_status == 1) and (pow1 > 0):
            self.noiseRvv_cnt = self.noiseRvv_cnt + 1
            if self.noiseRvv_cnt == 1:
                alpha = 0.5
            else:
                alpha = 0.95

            X_freq = np.zeros([self.nchannel, 1], dtype=np.complex)
            for i in range(0, self.half_fftlen):
                if 1 == noise_bin[i]:
                    X_freq[:, 0] = X[:, i]
                    Rxx = np.dot(X_freq, X_freq.conj().T)
                    self.noiseRvv[i, :, :] = alpha * self.noiseRvv[i, :, :] + (1 - alpha) * Rxx

            update_noise = 1

        return update_noise

    def UpdateSteeringVector(self, update_speech, update_noise):
        if update_speech or update_noise:
            Ryy = self.speechRyy
            Rvv = self.noiseRvv
            for i in range(0, self.half_fftlen):
                # Rxx = Ryy[i, :, :]

                if np.trace(Ryy[i, :, :]) > 2 * np.trace(Rvv[i, :, :]):
                    Rxx = Ryy[i, :, :] - Rvv[i, :, :]
                else:
                    Rxx = Ryy[i, :, :]

                v = np.zeros((self.nchannel, 1), dtype=complex)
                v[:, 0] = self.steering[i, :]
                Rxx = Rxx / np.trace(Rxx)
                eig_vect = np.matmul(Rxx, v)
                eig_norm = np.sqrt(np.sum(np.abs(eig_vect) ** 2))
                v = eig_vect / eig_norm
                self.steering[i, :] = v[:, 0]

    def UpdateMvdrFilter(self, update_speech, update_noise):
        """
        Replace inverse matrix with rank-one Cholesky decomposition
        :param update_speech:
        :param update_noise:
        :return:
        """
        if (1 == update_speech) or (1 == update_noise):
            Ryy = self.speechRyy
            Rvv = self.noiseRvv
            steering = np.zeros([self.nchannel, 1], dtype=np.complex)

            for i in range(0, self.half_fftlen):
                steering[:, 0] = self.steering[i, :]

                # FIXME: remove it, if we have proper initialization
                # if np.linalg.det(Rvv[i, :, :]) == 0:
                #     for n in range(0, self.nchannel):
                #         Rvv[i, n, n] = Rvv[i, n, n] * 1.01

                R_inv = np.linalg.inv(Rvv[i, :, :])
                num = np.matmul(R_inv, steering)
                denum = np.matmul(steering.conj().T, num)
                W = num / denum

                self.mvdr_coef[i, :] = W.T
        return

    # def UpdateAdaptiveMvdrFilter(self):
    #     """
    #     update adaptive mvdr coefficient
    #     : param v, column vector, (nchannel, 1)
    #     : param Rxx, covarince matrix (nchannel, nchannel)
    #     : param w, coefficient vector, column vector (nchannel, 1)
    #     g_n = (I - (v * v^H / ||v||^2) ) * R * w_(n-1)
    #     mu = g_n^H * R * w_(n-1) / g_n^H * R * g_n
    #     w_n = w_n-1 - mu * g_n
    #     """
    #     if self.fn == 1:
    #         self.adaptive_mvdr_coef = self.steering.copy()
    #         return
    #
    #     Ryy = self.speechRyy
    #     Rvv = self.noiseRvv
    #     w = np.zeros((self.nchannel, 1), dtype=complex)
    #     v = np.zeros((self.nchannel, 1), dtype=complex)
    #
    #     for i in range(0, self.half_fftlen):
    #
    #         w[:, 0] = self.adaptive_mvdr_coef[i, :]
    #         v[:, 0] = self.steering[i, :]
    #
    #         Rxx = Ryy[i, :, :]
    #         # Rxx = Rxx / np.trace(Rxx)
    #         Rxx_w = np.matmul(Rxx, w)
    #         V = np.matmul(v, v.conj().T)
    #         p = (np.eye(self.nchannel)) - (V / np.trace(V))
    #         g = np.matmul(p, Rxx_w)
    #
    #         num = np.matmul(g.conj().T, Rxx_w)
    #         den = np.matmul(Rxx, g)
    #         den = np.matmul(g.conj().T, den)
    #         mu = num / den
    #         w = w - mu * g
    #
    #         self.adaptive_mvdr_coef[i, :] = w[:, 0]
    #
    #     return

    # FIXME: compared to closed-form solution, this results are not good
    # need to come up with an interpretaion
    def UpdateAdaptiveMvdrFilter(self, update_speech, update_noise):
        if (1 == update_speech) or (1 == update_noise):
            Ryy = self.speechRyy
            Rvv = self.noiseRvv
            w = np.zeros((self.nchannel, 1), dtype=complex)
            v = np.zeros((self.nchannel, 1), dtype=complex)
            mu = 1

            for i in range(0, self.half_fftlen):
                w[:, 0] = self.adaptive_mvdr_coef[i, :]
                v[:, 0] = self.steering[i, :]

                Rxx = Ryy[i, :, :]
                Rxx = Rxx / np.trace(Rxx)
                Rxx_w = np.matmul(Rxx, w)
                w = w - mu * Rxx_w
                g = np.real(np.matmul(w.conj().T, v))
                if g < 1:
                    w = w + (1 - g) * v

                # self.adaptive_mvdr_coef[i, :] = 0.9 * self.adaptive_mvdr_coef[i, :] + 0.5 * w[:, 0]
                self.adaptive_mvdr_coef[i, :] = w[:, 0]

        return

    def UpdateGevFilter(self, update_speech, update_noise):
        if (1 == update_speech) or (1 == update_noise):
            Ryy = self.speechRyy
            Rvv = self.noiseRvv

            for i in range(0, self.half_fftlen):
                if np.linalg.det(Rvv[i, :, :]) == 0:
                    for n in range(0, self.nchannel):
                        Rvv[i, n, n] = Rvv[i, n, n] * 1.01
                Rvv_inv = np.linalg.inv(Rvv[i, :, :])
                numerator = np.matmul(Rvv_inv, Ryy[i, :, :])
                eigvalue, eigvect = np.linalg.eig(numerator)
                eigvalue_abs = abs(eigvalue)
                index = np.argmax(eigvalue_abs)
                W = eigvect[:, index]
                W = W / np.sqrt(self.nchannel)

                self.gev_coef[i, :] = W.T
        return

    def UpdateMwfFilter(self, update_speech, update_noise):
        mu = 1.0
        ref_ch = 0
        if (1 == update_speech) or (1 == update_noise):
            Ryy = self.speechRyy
            Rvv = self.noiseRvv

            for i in range(0, self.half_fftlen):
                Rxx = Ryy[i, :, :] - Rvv[i, :, :]
                if np.trace(Rxx) < 0:
                    Rxx = Ryy[i, :, :]

                if np.linalg.det(Rxx) == 0:
                    for n in range(0, self.nchannel):
                        Rxx[n, n] = Rxx[n, n] * 1.01
                Rvv_inv = np.linalg.inv(Rvv[i, :, :])
                numerator = np.matmul(Rvv_inv, Rxx)
                lambda_mu = mu + np.trace(numerator)
                numerator = numerator / (lambda_mu + 1e-12)
                W = numerator[ref_ch, :]

                self.mwf_coef[i, :] = W.T
        return

    def DoFilter(self, X, filter_type):
        coef_table = {
            'mvdr': self.mvdr_coef,
            'gev': self.gev_coef,
            'mwf': self.mwf_coef,
            'adaptive_mvdr': self.adaptive_mvdr_coef
        }
        coef = coef_table[filter_type]

        X = np.transpose(X)
        Y_out_tmp = X[0:self.half_fftlen, :] * coef.conj()
        Y_out = Y_out_tmp.sum(axis=1) / np.sqrt(self.nchannel)

        return Y_out
