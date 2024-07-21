import numpy as np
from bbhx.utils.constants import C_SI

def get_noise_covariance_matrix(f):
    noisemat_out = np.zeros((f.size, 3,3,))
    psdX = noisepsd_X(f)
    psdXY = noisepsd_XY(f)
    noisemat_out[:,:,:] += psdXY[:,None,None]
    for i in range(3):
        noisemat_out[:,i,i] = psdX

    return noisemat_out

lisaL = 2.5e9  # LISA's arm meters
lisaLT = lisaL / C_SI  # LISA's armn in sec

Sloc = (1.7e-12) ** 2  # m^2/Hz
Ssci = (8.9e-12) ** 2  # m^2/Hz
Soth = (2.0e-12) ** 2  # m^2/Hz
## Global
Soms_d_all = {
    "Proposal": (10.0e-12) ** 2,
    "SciRDv1": (15.0e-12) ** 2,
    "MRDv1": (10.0e-12) ** 2,
    "sangria": (7.9e-12) ** 2
}  # m^2/Hz

### Acceleration
Sa_a_all = {
    "Proposal": (3.0e-15) ** 2,
    "SciRDv1": (3.0e-15) ** 2,
    "MRDv1": (2.4e-15) ** 2,
    "sangria": (2.4e-15) ** 2,
}  # m^2/sec^4/Hz

def lisanoises(f, model="SciRDv1", unit = "relativeFrequency"):

    if isinstance(model, str):
        Soms_d_in = Soms_d_all[model]
        Sa_a_in = Sa_a_all[model]

    else:
        # square root of the actual value
        Soms_d_in = model[0] ** 2
        Sa_a_in = model[1] ** 2

    frq = f
    ### Acceleration noise
    ## In acceleration
    Sa_a = Sa_a_in * (1.0 + (0.4e-3 / frq) ** 2) * (1.0 + (frq / 8e-3) ** 4)
    ## In displacement
    Sa_d = Sa_a * (2.0 * np.pi * frq) ** (-4.0)
    ## In relative frequency unit
    Sa_nu = Sa_d * (2.0 * np.pi * frq / C_SI) ** 2
    Spm = Sa_nu

    ### Optical Metrology System
    ## In displacement
    Soms_d = Soms_d_in * (1.0 + (2.0e-3 / f) ** 4)
    ## In relative frequency unit
    Soms_nu = Soms_d * (2.0 * np.pi * frq / C_SI) ** 2
    Sop = Soms_nu

    if unit == "displacement":
        return Sa_d, Soms_d
    elif unit == "relativeFrequency":
        return Spm, Sop


def noisepsd_X(f, model="SciRDv1"):
    x = 2.0 * np.pi * lisaLT * f

    Spm, Sop = lisanoises(f, model)

    Sx = 16.0 * np.sin(x) ** 2 * (2.0 * (1.0 + np.cos(x) ** 2) * Spm + Sop)

    return Sx

def noisepsd_XY(f, model="SciRDv1"):
    x = 2.0 * np.pi * lisaLT * f

    Spm, Sop = lisanoises(f, model)

    Sxy = -4.0 * np.sin(2 * x) * np.sin(x) * (Sop + 4.0 * Spm)

    return Sxy