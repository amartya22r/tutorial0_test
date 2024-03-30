import numpy as np

def awgn(samps, snr_db, symb_pow):
    snr_lin = 10.0**(snr_db/10.0)
    noise_std = (symb_pow/(2.0*snr_lin))**(1.0/2.0)
    noise_samps = (np.random.normal(0.0, noise_std, len(samps)) + 1.0j*(np.random.normal(0.0, noise_std, len(samps))))
    return (samps + noise_samps)

def freq_off(samps, freq):
    return samps*[np.exp(2.0j*np.pi*freq*x) for x in range(0,len(samps))]
