
import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt

import pickle

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
        plot(   [expand_desc(e) for e in L_desc if e['endo_dropout'] == endo_dropout,
                output_path)
        print "Wrote %s." % output_path


def expand_desc(desc):
    # loads the files and the data and puts everything in a nice format

    assert os.path.exists(desc['log_file'])
    E = pickle.load(open(desc['log_file'], "r"))

    # TO DO : convert this into hitting times for a given list of target values (in log)

    #import pdb; pdb.set_trace()

    E.merge(process_D_log(E))

    # pick the color and label based on the number of workers
    E['color'] = "#555555"
    E['label'] = "patate"
    
    return E

def plot(L_desc, output_path):

    dpi = 150

    pylab.hold(True)

    for desc in L_desc:
        pylab.plot(desc['domain'], desc['image'], c=desc['color'], label=desc['label'])

    pylab.draw()
    pylab.savefig(output_path, dpi=dpi)
    pylab.close()



def process_D_log(D_log):

    # TO DO : change this to what you really want

    domain = []
    image = []

    for k in sorted(D_log.keys()):
        if D_log[k].has_key['train_error_rate']:
            domain.append(k)
            image.append(D_log[k]('train_error_rate'))

    return {'domain':domain, 'image':image}



def __name__ == "__main__":
    run()















