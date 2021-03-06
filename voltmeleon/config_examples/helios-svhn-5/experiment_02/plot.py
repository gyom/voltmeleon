
import os
import subprocess
import pickle
import re

import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt


def run():

    #id = "02"
    #root_dir = os.path.join(os.environ['HOME'], "NIPS")
    #experiment_dir = os.path.join(root_dir, "experiments/experiment_%s" % id)

    experiment_dir = os.getcwd()

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

    L_results = []
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

                if not v.has_key('timestamp'):
                    print "You are probably not using the `Timestamp` blocks extension because there is no 'timestamp' entry in the logs."
                    quit()

                L_step.append(v['timestamp'])
                for k in D_logged.keys():
                    D_logged[k].append(v[k])

        L_results.append({'A_step':np.array(L_step), 'D_logged':D_logged, 'observer_mode':observer_mode})

    # This is the smallest value encountered for all the steps.
    # We shouldn't plot anything before we come up with this value.
    step_min = None
    for res in L_results:
        if step_min is not None:
            step_min = np.min([step_min, res['A_step'].min()])
        else:
            step_min = res['A_step'].min()

    for res in L_results:

        domain = res['A_step']
        D_logged = res['D_logged']
        observer_mode = res['observer_mode']

        #import pdb; pdb.set_trace()

        for (k, c2) in D_colors.items():
            if D_logged.has_key(k):
                if observer_mode:
                    color = c2[0]
                    h = pylab.plot(domain - step_min, D_logged[k], c=color, label=k)
                else:
                    color = c2[1]
                    h = pylab.plot(domain - step_min, D_logged[k], c=color)
    
    plt.legend()

    if criterion == 'error_rate':
        pylab.ylim(ymin=0.0, ymax=1.0)
    elif criterion == 'cost':
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