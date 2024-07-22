import numpy as np
from numpy.lib.recfunctions import merge_arrays
from nessai.model import Model

def fast_mult(vectors1, vectors2, matrices):
    part = (vectors1[:,:,None,:].conj() * matrices).sum(axis=-1)
    part2 = (part * vectors2).real.sum(axis=-1)
    return part2

def matrix_inner(h1, h2, df, invC):
    # ensure inputs are of shape (N_t, N_f, 3,)
    if h1.ndim == 2:
        h1 = h1[None, ...]
    if h2.ndim == 2:
        h2 = h2[None, ...]
    # invC has shape (N_f, 3, 3)
    temp = 4 * df * fast_mult(h1, h2, invC[None, ...]).sum(axis=-1)  # fast_mult spits out something of shape (N_t, N_f)
    return temp

class XYZTDILikelihood(Model):
    """Generic likelihood assuming XYZ TDI variables"""

    def __init__(self, names, bounds, waveform_model, data, df, covariance_matrix, injection_parameter_dict, waveform_kwargs = None):
        self.names = names
        self.bounds = bounds

        self.data = data  # shape (N_f, 3)
        self.invC = np.linalg.inv(covariance_matrix)

        self.df = df

        self.waveform_model = waveform_model
        if waveform_kwargs is None:
            waveform_kwargs = {}
        self.waveform_kwargs = waveform_kwargs

        self.injection_parameter_dict = injection_parameter_dict
        inj_par_keys = list(self.injection_parameter_dict.keys())
        if inj_par_keys != names:
            self.fill_params = True

            fixed_params = []
            for nm in inj_par_keys:
                if nm not in names:
                    fixed_params.append(nm)
            fixed_param_values = tuple(injection_parameter_dict[nm] for nm in fixed_params)
            custom_dtype = [(nm, '<f8') for nm in fixed_params]
            self.merger_array = np.array([fixed_param_values,], custom_dtype)

        else:
            self.fill_params = False

        self._vectorised_likelihood = False#True

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniform
        priors on each parameter.

        Assume it's a flat prior for now
        """
        # Check if values are in bounds, returns True/False
        # Then take the log to get 0/-inf and make sure the dtype is float
        log_p = np.log(self.in_bounds(x), dtype="float")
        # Iterate through each parameter (x and y)
        # since the live points are a structured array we can
        # get each value using just the name
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, parameters):
        """
        Returns log likelihood of given live point.

        Performs the XYZ matrix inner product in vectorised form.
        """

        if self.fill_params: 
            merger_tile = np.repeat(self.merger_array, parameters.size)

            parameters = merge_arrays((parameters, merger_tile), flatten=True)

        templates = self.waveform_model(parameters, **self.waveform_kwargs)  # (N_t, N_f, 3)

        # if self.convolution_kernel is not None:
            # tdiX = np.convolve(np.squeeze(tdiX), self.convolution_kernel, mode='valid')
            # tdiY = np.convolve(np.squeeze(tdiY), self.convolution_kernel, mode='valid')
            # tdiZ = np.convolve(np.squeeze(tdiZ), self.convolution_kernel, mode='valid')

        # template_in = np.squeeze(np.array([tdiX, tdiY, tdiZ]))
        # data_in = np.array([sliced_data['X'], sliced_data['Y'], sliced_data['Z']])

        d_h = matrix_inner(templates, self.data, self.df,self.invC)  # (N_t,)
        h_h = matrix_inner(templates, templates, self.df,self.invC)  # (N_t,)

        ll = d_h - 0.5*h_h  # (N_t,)

        return ll


def estimate_parameter_uncertainties(waveform_model, parameters, df, covariance_matrix, waveform_kwargs=None):
    if waveform_kwargs is None:
        waveform_kwargs = {}

    invC = np.linalg.inv(covariance_matrix)

    names = ["Mt", "q", "a1", "a2", "dist", "phi_ref","iota","lam","beta","psi","t_ref"]

    steps = np.array([1, 1e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]) # Steps for numerical derivative. Seem "fine"
    N_params = len(steps) 
    deriv_vec = []
    parameters = parameters.copy()
    params_copy = parameters.copy()
    for j in range(N_params):
        parameters[j] = parameters[j] + 2*steps[j] # this is the f(x + h) step
        h_f_2p = waveform_model(parameters, **waveform_kwargs)
        parameters[j] = parameters[j] - steps[j] # this is the f(x + h) step
        h_f_p = waveform_model(parameters, **waveform_kwargs)

        parameters[j] = parameters[j] - 2 * steps[j] # this is the f(x - h) step
        h_f_m = waveform_model(parameters, **waveform_kwargs)

        parameters[j] = parameters[j] - steps[j] # this is the f(x - h) step
        h_f_2m = waveform_model(parameters, **waveform_kwargs)

        deriv_h_f = (-h_f_2p + 8*h_f_p - 8*h_f_m + h_f_2m) / (12*steps[j]) # compute derivative
        deriv_vec.append(deriv_h_f)
        parameters = params_copy # reset parameters

    gamma_XYZ = np.zeros((N_params, N_params))

    for i in range(N_params):
        for j in range(i, N_params):
            gamma_XYZ[i,j] = matrix_inner(deriv_vec[i], deriv_vec[j], df, invC)

    # for i in range(N_params):
    #     gamma_diag[i] = np.squeeze(matrix_inner(deriv_vec[i], deriv_vec[i], df, invC))
    # TODO replace with fisher matrix
    # covariances = 1/gamma_diag

    gamma_XYZ_diag = np.diag(np.diag(gamma_XYZ))
    gamma_XYZ -= 0.5 * gamma_XYZ_diag
    gamma_XYZ += gamma_XYZ.T

    cov_out = np.linalg.inv(gamma_XYZ)
    covariances = np.diag(cov_out)

    outp = dict()
    for i, nm in enumerate(names):
        outp[nm] = covariances[i]**0.5
    return outp

def set_bounds_from_errors(parameters, errors, names=None, scale=5):
    error_keys = list(errors.keys())
    if names is None:
        names = error_keys
    
    bounds = dict()

    for nm in names:
        for i in range(len(parameters)):
            if nm == error_keys[i]:
                indhere = i
        bounds[nm] = [parameters[indhere] - errors[nm]*scale, parameters[indhere] + errors[nm] * scale]
        if bounds[nm][0] < 0 and nm != "beta":
            bounds[nm][0] = 0.    
    
    return bounds


def SNR(parameters, waveform_model, df, covariance_matrix, waveform_kwargs=None):
    if waveform_kwargs is None:
        waveform_kwargs = {}

    invC = np.linalg.inv(covariance_matrix)
    wave_out = waveform_model(parameters, **waveform_kwargs)

    hh = matrix_inner(wave_out, wave_out, df, invC)

    return np.squeeze(hh)**0.5
