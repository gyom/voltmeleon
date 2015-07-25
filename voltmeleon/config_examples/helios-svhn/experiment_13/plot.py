
import os
import subprocess
import pickle
import re

import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt


# BUG : The observer uses a different schedule.
#       You can't just put in there, side-by-side.
#       Something else has to be done.
#       Besides, the observer config is not read properly,
#       so that needs to be debugged first.


def run():

    id = "13"

    root_dir = os.path.join(os.environ['HOME'], "NIPS")
    #experiment_dir = os.path.join(root_dir, "voltmeleon/voltmeleon/config_examples/helios-svhn/experiment_%s" % id)
    experiment_dir = os.path.join(root_dir, "experiments/experiment_%s" % id)


    read_results_and_plot(experiment_dir, 'cost')
    read_results_and_plot(experiment_dir, 'error_rate')


def read_results_and_plot(experiment_dir, criterion):

    assert criterion in ['cost', 'error_rate']

    if criterion == 'cost':
        D_colors = {'train_cost' : ('#00cc66', '#006600'),
                    'valid_cost' : ('#ff9933', '#cc6600')}
        outputfile = os.path.join(experiment_dir, "cost.png")
    elif criterion == 'error_rate':
        D_colors = {'train_error_rate' : ('#9999ff', '#0000ff'),
                    'valid_error_rate' : ('#ff66b2', '#cc0066')}
        outputfile = os.path.join(experiment_dir, "error_rate.png")



    L_result_files = find_result_files(experiment_dir)

    L_legend_handles = []
    L_legend_tags = []

    pylab.hold(True)

    max_L_step = 0

    for result_file in reversed(sorted(L_result_files, key=lambda e:e['worker_id'])):
        # The strange sorting thing is so that we'll print the worker_id==0
        # at the very end so it will be at the top of the graph, above the others.

        path = result_file['path']
        worker_id = result_file['worker_id']
        observer_mode = result_file['observer_mode']

        A = pickle.load(open(path, "r"))

        L_step = []

        D_logged = dict(train_cost = [],
                        train_error_rate = [],
                        valid_cost = [],
                        valid_error_rate = [])

        for (step,v) in A.items():

            if all([v.has_key(k) for k in D_logged.keys()]):
                L_step.append(step)
                for k in D_logged.keys():
                    D_logged[k].append(v[k])

        # save this for the time when we plot the observer steps
        # (that we have to fake)
        M = np.array(L_step).max()
        if max_L_step < M:
            max_L_step = M


        #import pdb; pdb.set_trace()

        for (k, c2) in D_colors.items():
            if D_logged.has_key(k):
                if observer_mode:
                    color = c2[0]
                    fake_L_step = np.linspace(0.0, max_L_step, len(L_step))
                    h = pylab.plot(fake_L_step, D_logged[k], c=color, label=k)
                else:
                    color = c2[1]
                    h = pylab.plot(L_step, D_logged[k], c=color)
    
    plt.legend()

    if criterion == 'error_rate':
        pylab.ylim(ymin=0.0)

    pylab.draw()
    pylab.savefig(outputfile, dpi=150)
    pylab.close()
    print "Wrote %s." % outputfile

    #import pdb; pdb.set_trace()


def find_result_files(dir):
    """
    Find all the files called "log_\d\d" in the directory.
    """

    L_files = sorted([e for e in subprocess.check_output("find %s -name 'log*'" % (dir,), shell=True).split("\n") if len(e)>0])

    L_result_files = []
    for path in L_files:
        m = re.match(r".*/log_(\d\d)_log", path)
        if m:
            L_result_files.append({'path' : path, 'worker_id' : int(m.group(1)), 'observer_mode' : False})

        # yeah, the observer uses a slightly different name for the log
        m = re.match(r".*/log_(\d\d)_obs_log", path)
        if m:
            L_result_files.append({'path' : path, 'worker_id' : int(m.group(1)), 'observer_mode' : True})


    return L_result_files




def plot(L_step, D_logged):

    print "Generating plot."

    pylab.hold(True)
    pylab.scatter(samples[:,0], samples[:,1], c='#f9a21d')
   
    arrows_scaling = 1.0
    pylab.quiver(plotgrid[:,0],
                 plotgrid[:,1],
                 arrows_scaling * (grid_pred[:,0] - plotgrid[:,0]),
                 arrows_scaling * (grid_pred[:,1] - plotgrid[:,1]))
    pylab.draw()
    pylab.axis([center[0] - window_width*1.0, center[0] + window_width*1.0,
                center[1] - window_width*1.0, center[1] + window_width*1.0])
    pylab.savefig(outputfile, dpi=dpi)
    pylab.close()


if __name__ == "__main__":
    run()