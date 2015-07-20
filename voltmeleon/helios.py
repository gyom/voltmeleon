

import os

def get_properties():

    props = {}

    # see https://wiki.calculquebec.ca/w/Moab#Particularit.C3.A9s_de_chaque_serveur
    for k in ['MOAB_JOBNAME', 'MOAB_USER', 'MOAB_TASKMAP', 'MOAB_CLASS', 'MOAB_PROCCOUNT', 'MOAB_GROUP', 'MOAB_NODELIST', 'MOAB_ACCOUNT', 'MOAB_NODECOUNT', 'MOAB_JOBID', 'MOAB_JOBARRAYINDEX', 'MOAB_QOS']:
        if os.environ.has_key(k):
            props[k] = os.environ[k]

    return props

def is_job_zero():
    return get_id() == 0

def get_id():
    assert os.environ.has_key('MOAB_JOBARRAYINDEX')
    id = int(os.environ['MOAB_JOBARRAYINDEX'])

    import theano
    import re
    m = re.match("gpu(\d+)", theano.config.device)
    if m:
        offset = int(m.group(1))
        print "Device GPU offset %d." % offset
    else:
        print "Failed to read the offset based on theano.config.device : %s" % theano.config.device
        assert m

    return 2 * base_id + offset

def print_properties():

    print "=== helios moab properties ==="
    props = get_properties()
    for (k,v) in props.items():
        print "%s : %s" % (k, v)
    print "=============================="
