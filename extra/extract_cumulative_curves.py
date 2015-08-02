
import numpy as np


def smoothe(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def process_D_log(D_log):

    domain = []
    image = []

    for k in sorted(D_log.keys()):
        if D_log[k].has_key('train_error_rate'):
            domain.append(k)
            image.append(D_log[k]['train_error_rate'])

    domain = np.array(domain)
    image = np.array(image)

    return process(domain, image, N=25)

def process(domain, image, N=25):

    image = smoothe(image, N=N)
    domain = domain[0:image.shape[0]]

    cumulative_min_image = np.minimum.accumulate( image )

    #print "== smoothed image =="
    #print image
    #print "== cumulative_min_image =="
    #print cumulative_min_image

    levels = np.exp( np.linspace(np.log(0.2), np.log(1.0e-3), 100 ) )

    ht = hitting_times_descending(domain, cumulative_min_image, levels)

    #return {'domain':levels, 'image':ht}
    return {'domain':ht, 'image':levels}


def hitting_times_descending(domain, cumulative_min_image, levels):

    L_times = []
    for level in levels:
        I = np.where(cumulative_min_image <= level)[0]
        #print level
        #print I
        if 0 < len(I):
            t = domain[I[0]]
            L_times.append(t)
        else:
            # this level has never been achieved
            L_times.append(np.nan)
            #pass

    return np.array(L_times)


def test():

    T = 10
    domain = np.arange(0, T)
    image = np.random.rand(T)
    image = np.sort(image)[::-1]

    print "== domain =="
    print domain
    print "== image =="
    print image

    E = process(domain, image, N=2)

    print "== levels =="
    print E['domain']
    print "== hitting times =="
    print E['image']


if __name__ == "__main__":
    test()