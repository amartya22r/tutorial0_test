import numpy as np
import matplotlib.pyplot as pyplot

from src import modems
from src import channels

def awgn_symbol_alrt(rx_symbs, snr_db, symb_tables):
    snr_lin = 10.0**(snr_db/10.0)
    noise_std = (1.0/(2.0*snr_lin))**(1.0/2.0)
    
    max_prob = -np.inf
    for mod_num, symbols in enumerate(symb_tables):
        prob = 0
        sum_symbs = np.zeros((1,len(rx_symbs)))

        for symb in symbols:
            sum_symbs = sum_symbs + np.exp(-(1.0/((np.sqrt(2)*noise_std)**2))*np.abs(rx_symbs-symb)**2)

        prob = np.sum(np.log((1.0/len(symbols))*sum_symbs))

        if prob > max_prob:
            max_prob = prob
            classification = mod_num

    return classification

def awgn_symbol_alrt_sweep(snr_db_list=list(range(0,21)), mod_set = [['psk', 2], ['psk', 4], ['psk', 8], ['qam', 16], ['qam', 64]]
, total_symbs = 1024, trials = 1000, plot = 1):

    # Creation of the Modems and Symbol Tables
    modem_set = []
    symb_tables = []
    for mod,order in mod_set:
        modem = modems.ldapm(mod_type=mod,mod_order=order)

        modem_set.append(modem)
        symb_tables.append(modem.symb_table)

    # ALRT Sweep
    probs = np.zeros((len(mod_set),len(snr_db_list)))
    for snr_num, snr_db in enumerate(snr_db_list):
        for trial_num in range(trials):
            for mod_num, modem in enumerate(modem_set):
                symbs = modem.gen_symbs(total_symbs)
                noisy_symbs = channels.awgn(symbs, snr_db, 1.0)

                classification = awgn_symbol_alrt(noisy_symbs, snr_db, symb_tables)
                if classification == mod_num:
                    probs[mod_num][snr_num] = probs[mod_num][snr_num] + 1.0/trials

            print('Completed Trial ' + str(trial_num+1) + ' of SNR ' + str(snr_db))
    avg_alrt = np.mean(probs,0)

    # Performance Plotting
    if plot == 1:
        for mod_num, modem in enumerate(modem_set):
            pyplot.plot(snr_db_list, probs[mod_num][:], label=str(mod_set[mod_num][1]) + '-' + mod_set[mod_num][0],marker='*')
        pyplot.plot(snr_db_list, avg_alrt, label='Average', marker='*')    
        pyplot.xlabel('SNR (dB)')
        pyplot.ylabel('Average Probability of Correct Classification')
        pyplot.legend()
        pyplot.show(block=True)

    return avg_alrt

if __name__=="__main__":
    awgn_symbol_alrt_sweep()
