

import os

assert os.environ.has_key('MOAB_JOBARRAYINDEX')

def get_properties():

    props = {}

    # see https://wiki.calculquebec.ca/w/Moab#Particularit.C3.A9s_de_chaque_serveur
    for k in ['MOAB_JOBNAME', 'MOAB_USER', 'MOAB_TASKMAP', 'MOAB_CLASS', 'MOAB_PROCCOUNT', 'MOAB_GROUP', 'MOAB_NODELIST', 'MOAB_ACCOUNT', 'MOAB_NODECOUNT', 'MOAB_JOBID', 'MOAB_JOBARRAYINDEX', 'MOAB_QOS']:
        props[k] = os.environ[k]

    return props

def is_job_zero():
    return os.environ['MOAB_JOBARRAYINDEX'] == '0'

def print_properties():

    print "=== helios moab properties ==="
    props = get_properties()
    for (k,v) in props.items():
        print "%s : %s" % (k, v)
    print "=============================="
