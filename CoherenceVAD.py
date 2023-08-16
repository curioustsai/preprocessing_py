"""
Coherence VAD

Detect human by Coherence vad
search frequency between CohFreqDn ~ CohFreqUp.

Default value:
CohFreqDn: 50Hz
CohFreqUp: 4000Hz
"""

import numpy as np
import matplotlib.pyplot as plt

#define constant
#alpha = 0.75;
alpha_init = 0.75#0.75
alpha_N1 = 0.8 #0.8
alpha_N2 = 0.9 #0.85
N1_hold = 5     #frame 
N2_hold = 12    #frame
thr = 30 #40
thr1 = 25 #30
noise_bin_std_thr = 2  

class CoherenceVAD:
    def __init__(self, fftLen, CohFreqUp, CohFreqDn):
        self.fftlen = fftLen
        self.half_fftlen = fftLen // 2 + 1
        self.CohFreqUp = round(CohFreqUp*fftLen/16000)+1
        self.CohFreqDn = round(CohFreqDn*fftLen/16000)+1
        self.Pxii = np.zeros(self.half_fftlen, dtype=float)
        self.last_Pxii = np.zeros(self.half_fftlen, dtype=float)
        self.Pxjj = np.zeros(self.half_fftlen, dtype=float)
        self.last_Pxjj = np.zeros(self.half_fftlen, dtype=float)
        self.Pxij = np.zeros(self.half_fftlen, dtype=float)
        self.last_Pxij = np.zeros(self.half_fftlen, dtype=float)
        self.Coh = np.zeros(self.half_fftlen, dtype=float)
        self.km = np.zeros(self.half_fftlen, dtype=float) # remove k buffer
        self.vad_hold = 0
        
    def CoherenceCal(self, X):
        half_fftlen = self.half_fftlen
        vad_hold = self.vad_hold
        
        if vad_hold >= N1_hold:
            alpha = alpha_N1
            if vad_hold >= N2_hold:
                alpha = alpha_N2
        else:
            alpha = alpha_init
            
        #alpha = alpha_init
            
        
        self.Pxii = X[0, 0:half_fftlen]*np.conj(X[0, 0:half_fftlen])
        self.Pxii = alpha*self.Pxii + (1-alpha)*self.last_Pxii
        self.Pxii = self.Pxii.real
        
        self.Pxjj = X[1, 0:half_fftlen]*np.conj(X[1, 0:half_fftlen])
        self.Pxjj = alpha*self.Pxjj + (1-alpha)*self.last_Pxjj
        self.Pxjj = self.Pxjj.real   
        
        self.Pxij = X[0, 0:half_fftlen]*np.conj(X[1, 0:half_fftlen])   
        self.Pxij = self.Pxij*np.conj(self.Pxij) # optional
        self.Pxij = alpha*self.Pxij + (1-alpha)*self.last_Pxij   
        
        self.Coh =  self.Pxij / np.sqrt(self.Pxii,self.Pxjj)
        self.Coh =  self.Coh.real #/(32768*32768)
        
        CohSTD = np.std(self.Coh[self.CohFreqUp:self.CohFreqDn], ddof=1)
        #CohSTD = CohSTD/(32768*32768*256)
        #CohSTD = np.var(self.Coh[self.CohFreqUp:self.CohFreqDn])
        #tmp = self.Pxii*self.Pxjj
        #self.Coh =  self.Pxij / (self.Pxii*self.Pxjj)
        #self.Coh =  self.Pxij / tmp
        #for i in range(10):
        #    print('tmp = \n',i,tmp[i])
        #    print('Pxij = \n',i,self.Pxij[i])
        
        #if CohSTD < 0:
        #    tmp = 1
            
        vad = 0
        if CohSTD > thr :
           #vad = 1 
           vad_hold += 1
        else:
           vad_hold = max(0, vad_hold-1)
           
        self.vad_hold = vad_hold
        
        if CohSTD > thr1 :
            vad = 1 
            # set km buffer 
            for i in range(half_fftlen):
                if self.Coh[i] > noise_bin_std_thr:
                    self.km[i] = 0
                else:
                    self.km[i] = 1
            
        return CohSTD, vad, self.km

