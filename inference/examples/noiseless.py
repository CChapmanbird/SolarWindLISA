import sys
import os
sys.path.append('../')  # so we can see the packages in the above directory
from model import XYZTDILikelihood, estimate_parameter_uncertainties, set_bounds_from_errors, SNR
from waveform import wrap_BBHx_likelihood, wrap_BBHx_normal
from data import inject_signal
from noise import get_noise_covariance_matrix

import numpy as np
from nessai.flowsampler import FlowSampler
from nessai.utils import setup_logger
from nessai.plot import corner_plot

from bbhx.waveformbuild import BBHWaveformFD

### DATA PARAMETERS

duration = 86400 * 14  # two weeks
dt = 1.  # sampling cadence (s)
df = 1/duration

fmin = 1e-3
fmax = 1e-1
num_freqs = int((fmax - fmin) / df) + 1

freqs = np.arange(num_freqs) * df + fmin
 
### WAVEFORM WRAPPER

wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), response_kwargs=dict(TDItag="XYZ"))
wave_wrap_like = wrap_BBHx_likelihood(wave_gen, f_ref=0.)
wave_wrap_norm = wrap_BBHx_normal(wave_gen, f_ref=0.)

### SOURCE PARAMETERS

phi_ref = np.pi/4 # phase at f_ref
m1 = 3e6
m2 = 1e6
a1 = 0.2
a2 = 0.4
dist = 10. # Gpc
inc = np.pi/3.
beta = np.pi/4.  # ecliptic latitude
lam = np.pi/5.  # ecliptic longitude
psi = np.pi/6.  # polarization angle
t_ref = duration / 2  # t_ref (seconds)

Mt = m1 + m2
q = m1/m2

params_in = np.array([
    Mt,
    q,
    a1,
    a2,
    dist,
    phi_ref,
    inc,
    lam,
    beta,
    psi,
    t_ref
])

params_in_dict = {nm: params_in[i] for i, nm in enumerate(["Mt", "q", "a1", "a2", "dist", "phi_ref","iota","lam","beta","psi","t_ref"])}

waveform_kwargs = dict(squeeze=True, freqs=freqs, direct=False, length=1024, fill=True, combine=False)

# inject signal (data = None for noiseless)  ASSUMES DATA IS IN FREQUENCY DOMAIN!

data_with_signal = inject_signal(wave_wrap_norm, params_in, data=None, waveform_kwargs=waveform_kwargs)

##### configure likelihood

# noise assumptions: covariance matrix (has shape (3,3,N_f))

covariance_matrix = get_noise_covariance_matrix(freqs)

# parameters to sample over
parameters_to_sample = ["Mt", "q", "a1", "a2", "dist", "phi_ref","iota","lam","beta","psi","t_ref"]

# parameters_to_sample = ["Mt", "q", "a1", "a2", "dist"]

# smartly estimate the prior bounds of the analysis from the Fisher information matrix
marginal_widths = estimate_parameter_uncertainties(wave_wrap_norm, params_in, df, covariance_matrix, waveform_kwargs=waveform_kwargs)
prior_bounds = set_bounds_from_errors(params_in, marginal_widths, names=parameters_to_sample, scale=5)

# instantiate likelihood object
likelihood_model = XYZTDILikelihood(parameters_to_sample, prior_bounds, wave_wrap_like, data_with_signal, df, covariance_matrix, params_in_dict, waveform_kwargs=waveform_kwargs)

# get the SNR for information
snr_inj = SNR(params_in, wave_wrap_norm, df, covariance_matrix, waveform_kwargs=waveform_kwargs)

outdir = "all_params_example"  # output directory name
ncores = 4  # number of CPU cores to use
logger = setup_logger(output=outdir)

logger.info(f"Optimal SNR of injection is {snr_inj:.2f}")

fs = FlowSampler(
    likelihood_model,
    output=outdir,
    stopping=0.1,
    resume=True,
    seed=42,
    nlive=1000,
    proposal_plots=False,
    plot=True,
    likelihood_chunksize=100,
    n_pool=ncores,
)
fs.run()

# plot the results

true_values = [params_in_dict[nm] for nm in parameters_to_sample]
corner_plot(
    fs.posterior_samples, 
    exclude=["logL","logP","it"], 
    truths=true_values,
    filename=os.path.join(outdir, "corner_plot.pdf"),
)


