"""
pwc_opt.py:  Methods for power control optimization
"""
import numpy as np


class UtilMax(object):
    """
    Uplink power control optimizer.

    The problem is to select a vector of power values, p,
    where p[i] is the TX power on link i relative to maximum TX power.
    Hence, p[i] \in [0,1].  We assume the rate on the links are
    given by

        snr = g0*p/q,   q = G*p + 1

    where q is the interference + thermal noise.

    """

    def __init__(self, g0, G, max_se=4.5, bw_loss=0.6, bw=400e6):
        """
        Constructor:

        Parameters
        ----------
        max_se float:
            Max spectral efficiency
        bw_loss, float:
            Fractional loss in bandwidth
        bw, flaat:
            Total bandwidth in Hz
        snr_min, float:
            Minimum SNR in dB on any link
        """
        self.max_se = max_se
        self.bw_loss = bw_loss
        self.bw = bw
        self.G = G
        self.g0 = g0

    def rate_fn(self, snr):
        """
        Computes rate in bps/Hz and its gradient wrt to the SNR
        """
        se = self.bw_loss * np.log2(1 + snr)
        I = np.where(se >= self.max_se)
        se = np.minimum(se, self.max_se)

        se_grad = self.bw_loss / np.log(2) / (1 + snr)
        se_grad[I] = 0
        return se, se_grad

    def util_eval(self, p, return_snr_rate=False):

        # Compute the SNR
        q = self.G.dot(p) + 1
        snr = self.g0 * p / q

        # Compute the rate and its gradient
        rate, rate_grad = self.rate_fn(snr)

        # Compute the utility
        util = np.sum(np.log(rate))

        # Compute the gradient with p
        ugrad_snr = 1 / rate * rate_grad
        ugrad = ugrad_snr * self.g0 / q - self.G.T.dot(ugrad_snr * snr / q)

        if return_snr_rate:
            return snr, rate, util, ugrad
        else:
            return util, ugrad

    def test_gradient(self, step=1e-6):
        """
        Test the gradients

        Parameters
        ----------
        step : TYPE, optional
            DESCRIPTION. The default is 1e-6.

        Returns
        -------
        None.

        """
        nlinks = len(self.g0)
        p0 = np.random.uniform(0, 1, nlinks)
        p0 = np.minimum(1, np.maximum(1e-6, p0))
        p1 = p0 + np.random.normal(0, step, nlinks)
        p1 = np.minimum(1, np.maximum(1e-6, p1))

        util0, ugrad0 = self.util_eval(p0)
        util1, ugrad1 = self.util_eval(p1)

        dutil = util1 - util0
        dutil_exp = ugrad0.dot(p1 - p0)
        print('dutil:  Act %12.4e Exp %12.4e' % (dutil, dutil_exp))

    def grad_des_opt(self, nit=100, step_init=1e-6):
        """
        Armijo-Rule based projected gradient descent
        """

        nlinks = len(self.g0)

        p0 = np.ones(nlinks)
        util0, ugrad0 = self.util_eval(p0)
        step = step_init

        # Keep history for debugging
        self.hist = dict()
        self.hist['util'] = []
        self.hist['step'] = []

        for it in range(nit):

            # Get candidate point
            p1 = p0 + step * ugrad0
            p1 = np.minimum(1, np.maximum(1e-6, p1))

            # Get utility and gradient
            util1, ugrad1 = self.util_eval(p1)

            # Check if successful
            dutil = util1 - util0
            dutil_exp = ugrad0.dot(p1 - p0)
            if (dutil > 0) and (dutil > 0.5 * dutil_exp):
                step = step * 2
                util0 = util1
                p0 = p1
                ugrad0 = ugrad1
            else:
                step = step * 0.5
                step = np.maximum(1e-6, step)

            self.hist['util'].append(util0)
            self.hist['step'].append(step)

        return p0