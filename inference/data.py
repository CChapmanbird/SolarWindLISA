import numpy as np

def inject_signal(waveform_model, parameters, data=None, waveform_kwargs=None):
    """
    Injects a signal in the frequency domain. 
    """
    if waveform_kwargs is None:
        waveform_kwargs = {}
    
    wf_out = np.squeeze(waveform_model(parameters, **waveform_kwargs))  # (N_f, 3)

    if data is None:
        return wf_out
    else:
        return data + wf_out