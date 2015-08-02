
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt

import pickle

import plotting_helper
#colors = plotting_helper.MplColorHelper('YlOrBr', 0, 20)
colors = plotting_helper.MplColorHelper('gist_heat', -5, 20)


import extract_cumulative_curves

def run():

    L_desc = []


    desc300 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-3/experiment_00/log_00_obs_log",
                'nbr_workers':1,
                'endo_dropout':0.0}

    desc301 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-3/experiment_01/log_00_obs_log",
                'nbr_workers':3,
                'endo_dropout':0.0}

    desc302 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-3/experiment_02/log_00_obs_log",
                'nbr_workers':5,
                'endo_dropout':0.0}

    desc303 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-3/experiment_03/log_00_obs_log",
                'nbr_workers':11,
                'endo_dropout':0.0}
    L_desc = L_desc + [desc300, desc301, desc302, desc303]

    desc304 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-3/experiment_04/log_00_obs_log",
                'nbr_workers':1,
                'endo_dropout':0.5}

    desc305 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-3/experiment_05/log_00_obs_log",
                'nbr_workers':3,
                'endo_dropout':0.5}

    desc306 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-3/experiment_06/log_00_obs_log",
                'nbr_workers':5,
                'endo_dropout':0.5}

    desc307 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-3/experiment_07/log_00_obs_log",
                'nbr_workers':11,
                'endo_dropout':0.5}
    L_desc = L_desc + [desc304, desc305, desc306, desc307]

    desc204 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-2/experiment_04/log_00_obs_log",
                'nbr_workers':7,
                'endo_dropout':0.0}

    desc207 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-2/experiment_07/log_00_obs_log",
                'nbr_workers':15,
                'endo_dropout':0.0}

    desc212 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-2/experiment_12/log_00_obs_log",
                'nbr_workers':7,
                'endo_dropout':0.5}

    desc215 = { 'log_file':"/home/dpln/NIPS/experiments-svhn-2/experiment_15/log_00_obs_log",
                'nbr_workers':15,
                'endo_dropout':0.5}
    L_desc = L_desc + [desc204, desc207, desc212, desc215]


    for endo_dropout in [0.0, 0.5]:

        output_path = "six_workers_numbers_endo_dropout_%0.2f.png" % endo_dropout
        plot_smoothed_original_curves_log_scale([expand_desc(e) for e in L_desc if e['endo_dropout'] == endo_dropout],
                                                output_path,
                                                endo_dropout)
        print "Wrote %s." % output_path


def expand_desc(desc):
    # loads the files and the data and puts everything in a nice format

    assert os.path.exists(desc['log_file']), desc['log_file']
    E = pickle.load(open(desc['log_file'], "r"))

    # TO DO : convert this into hitting times for a given list of target values (in log)

    #import pdb; pdb.set_trace()

    A = process_D_log(E)

    #A = extract_cumulative_curves.process_D_log(E)

    #import pdb; pdb.set_trace()

    # pick the color and label based on the number of workers

    A['color'] = colors.get_rgb(desc['nbr_workers'])
    A['label'] = "%d" % desc['nbr_workers']
    A['nbr_workers'] = desc['nbr_workers']

    return A

def plot_smoothed_original_curves_log_scale(L_desc, output_path, endo_dropout):

    dpi = 150

    pylab.hold(True)

    smoothing_window_size = 50
    for desc in sorted(L_desc, key=lambda e: e['nbr_workers']):

        domain = extract_cumulative_curves.smoothe(desc['domain'], N=smoothing_window_size)
        image = extract_cumulative_curves.smoothe(desc['image'], N=smoothing_window_size)
        mask = np.isfinite(image)

        pylab.plot( domain[mask],
                    image[mask],
                    c=desc['color'], label=desc['label'])

    pylab.gca().set_xscale('linear')
    pylab.gca().set_yscale('log')

    pylab.ylim([0.0, 0.2])
    pylab.xlim([0.0, 10500.0])

    pylab.title("SVHN, training error rate (log scale), dropout %0.2f" % endo_dropout)
    pylab.xlabel("duration of training")
    pylab.ylabel("training error rate (log scale)")

    pylab.legend(loc=3)
    pylab.draw()
    pylab.savefig(output_path, dpi=dpi)
    pylab.close()



def plot(L_desc, output_path):

    dpi = 150

    pylab.hold(True)

    for desc in L_desc:
        pylab.plot(desc['domain'], desc['image'], c=desc['color'], label=desc['label'])

    pylab.legend()
    pylab.draw()
    pylab.savefig(output_path, dpi=dpi)
    pylab.close()




def process_D_log(D_log):

    # TO DO : change this to what you really want

    domain = []
    image = []

    for k in sorted(D_log.keys()):
        if D_log[k].has_key('train_error_rate'):
            domain.append(k)
            image.append(D_log[k]['train_error_rate'])

    return {'domain':domain, 'image':image}



if __name__ == "__main__":
    run()















