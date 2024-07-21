from bbhx.utils.constants import YRSID_SI, PC_SI
import numpy as np

def wrap_BBHx_likelihood(bbhx_instance, f_ref=0):
    """
    Wraps the BBHx waveform class to be in a more convenient form for the likelihood function.
    """
    return lambda x, **kwargs: np.swapaxes(np.asarray(bbhx_instance(
        x['Mt']*(x['q']/(1+x['q'])),
        x['Mt']/(1 + x['q']),
        x['a1'],
        x['a2'],
        x['dist'] * PC_SI * 1e9,
        x['phi_ref'],
        f_ref,
        x['iota'],
        x['lam'],
        x['beta'],
        x['psi'],
        x['t_ref'],
        **kwargs
    )),1, 2)

def wrap_BBHx_normal(bbhx_instance, f_ref=0):
    """
    Wraps the BBHx waveform class to be in a more convenient form for normal use.
    """
    return lambda x, **kwargs: np.swapaxes(np.asarray(bbhx_instance(
        x[0]*(x[1]/(1+x[1])),
        x[0]/(1 + x[1]),
        x[2],
        x[3],
        x[4] * PC_SI * 1e9,
        x[5],
        f_ref,
        x[6],
        x[7],
        x[8],
        x[9],
        x[10],
        **kwargs
    )),1,2)
