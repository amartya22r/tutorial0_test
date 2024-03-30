import numpy
import matplotlib.pyplot as pyplot
from scipy.stats import norm
from scipy.special import erfc
import src.modems as modems
import src.channels as channels

# User Defined Parameters
total_symbs = 100000
sps = numpy.random.randint(2,20)

mod_type = 'psk'
mod_order = 4

filt_type = 'rrc'
beta = numpy.random.uniform(0,1)
span = numpy.random.randint(2,100)
trim = 0

snr_db_min = 0
snr_db_max = 8

# Symbol Error Rate (SER) Calculations
modem = modems.ldapm(sps=sps, mod_type=mod_type, mod_order=mod_order, filt_type=filt_type, beta=beta, span=span, trim=trim)
ser_sim = []
ser_theory = []
for x in range(snr_db_min, snr_db_max+1):
    tx_symbs = modem.gen_symbs(total_symbs)
    samps = modem.get_samps(tx_symbs)

    noisy_samps = channels.awgn(samps, x, 1.0)
    rx_symbs = modem.get_symbs(noisy_samps)

    symb_errors = 0
    for y in range(0,len(rx_symbs)):
        idx = numpy.abs(rx_symbs[y]-modem.symb_table).argmin()
        symb_errors = symb_errors + (1-numpy.equal(tx_symbs[y], modem.symb_table[idx]))
    ser_sim.append(symb_errors / len(rx_symbs))

    if mod_type == 'psk':
        if mod_order == 2:
            ser_theory.append(norm.sf((2.0*10.0**(x/10.0))**(1.0/2.0)))
        elif mod_order == 4:
            ser_theory.append(1-(1-norm.sf((10.0**(x/10.0))**(1.0/2.0)))**(2.0))
        else:
            ser_theory.append(2.0*norm.sf((2.0*10.0**(x/10.0))**(1.0/2.0)*numpy.sin(numpy.pi/mod_order)))
    elif mod_type == 'qam':
        k = ((1.0)/(((2.0)/(3.0))*(mod_order-1.0)))**(1.0/2.0)
        ser_theory.append(2.0*(1.0-(1.0)/(mod_order**(1.0/2.0)))*erfc(k*(10.0**(x/10.0))**(1.0/2.0))-(1.0-(2.0)/(mod_order**(1.0/2.0))+(1.0)/(mod_order))*erfc(k*(10.0**(x/10.0))**(1.0/2.0))**2.0)

# Plotting SER Simulation and Theory Curves
pyplot.semilogy(range(snr_db_min, snr_db_max+1), ser_sim, 'b-*', label='Simulated')
pyplot.semilogy(range(snr_db_min, snr_db_max+1), ser_theory, 'r-*', label='Theory')
pyplot.xlabel('SNR (dB)')
pyplot.ylabel('Symbol Error Rate (SER)')
pyplot.legend()
pyplot.show()
