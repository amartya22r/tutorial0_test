import numpy as np
import matplotlib.pyplot as pyplot
import src.modems as modems
import src.channels as channels
from scipy.stats import norm

mod_type = 'psk'
mod_order = 2
total_symbs = 2
sps = 10
snrdb = 100
span = 6

modem = modems.ldapm(sps=sps, mod_type=mod_type, mod_order=mod_order, filt_type='rrc', beta=0.35, span=span, trim=1)

tx_symbs = modem.gen_symbs(total_symbs)

tx_samps = modem.gen_samps(102)

pyplot.plot(np.real(tx_samps))
pyplot.show()

print(len(np.real(tx_samps)))
