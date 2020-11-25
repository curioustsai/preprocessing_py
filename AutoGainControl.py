"""
Automatic Gain Control
(AGC module)

input: time domain signal y
output: agc result y_out
"""

class AutoGainControl:
    def __init__(self, tpka_fp, g_min, g_max):
        # parameters
        self.tpka_fp = tpka_fp  # target peak amp.
        self.g_min_fp = g_min
        self.g_max_fp = g_max
        self.lambda_pka_fp = 31/32  # smooth factor

        self.eps = 64   # 64/32768
        self.last_g_fp = 1.0
        self.pka_fp = tpka_fp  # init envelope peak amp.

    def agc(self, y, speech_frame, spp):
        max_abs_y = max(abs(y))
        if max_abs_y > self.pka_fp:
            self.pka_fp = 0.75*self.pka_fp + 0.25*max_abs_y
        else:
            # if it is noisy, maintain the same level to avoid fluctuation
            if speech_frame == 1 and spp > 0.5 and max_abs_y > self.eps:
                mu = spp * (1 - self.lambda_pka_fp)
                self.pka_fp = (1-mu)*self.pka_fp + mu*max_abs_y

        g = self.tpka_fp/(self.pka_fp + 1e-12)
        g = max(g, self.g_min_fp)
        g = min(g, self.g_max_fp)

        # g_fp smooth, and prevent from clipping
        g_fp = 15 / 16 * self.last_g_fp + 1 / 16 * g
        if g_fp * max_abs_y > 32767:
            g_fp = 32767 / (max_abs_y + 1e-12)

        y_out = g_fp * y
        self.last_g_fp = g_fp

        return y_out, self.pka_fp, g_fp, max_abs_y




