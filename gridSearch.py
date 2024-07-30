import main


def gridsearch_Layer():
    best_lay = 0
    best_neuron = 0
    best_NMSE = 10e3
    lays = [2]
    neurons = [200, 256, 300, 400]
    for lay in lays:
        for neuron in neurons:
            mean_NMSE = main.main(lay, neuron)
            if mean_NMSE < best_NMSE:
                best_NMSE = mean_NMSE
                best_neuron = neuron
    print('Best NMSE: ', best_NMSE, '\nBest neuron: ', best_neuron)

def gridsearch_Lambdas():
    best_NMSE = 10e3
    best_lam_f = 0
    best_lam_t = 0
    lams_f = [1, 5, 10, 50]
    lams_t = [1, 5, 10, 50]
    for lam_f in lams_f:
        for lam_t in lams_t:
            NMSE = main.main(lam_f, lam_f)
            if NMSE < best_NMSE:
                best_NMSE = NMSE
                best_lam_f = lam_f
                best_lam_t = lam_t
    print('NMSE: ', best_NMSE, 'best_lam_f: ', best_lam_f, 'best_lam_t: ', best_lam_t)


def gridSearch_Relobralo():
    tmps = [0.1, 0.01, 0.001, 10e-05]  # , 10e-05
    rhos = [0.99, 0.999]
    alphas = [0.9, 0.99]
    best_NMSE = 10e5
    best_tmp = 0
    best_rho = 0
    best_alpha = 0

    for tmp in tmps:
        for rho in rhos:
            for alpha in alphas:
                NMSE = main.main(tmp, rho, alpha)
                if NMSE < best_NMSE:
                    best_NMSE = NMSE
                    best_tmp = tmp
                    best_rho = rho
                    best_alpha = alpha

    print('NMSE: ', best_NMSE, 'best_rho: ', best_rho, 'best_alpha: ', best_alpha, 'best_tmp: ', best_tmp)


def gridSearch_modes():
    # 7: 0.0094905 / 8: 0.012216 / 9: 0.016826 / 11: 0.03027 / 13: 0.030792 / 14: 0.033438 / 18: 0.057536
    # /20: 0.06447 / 22: 0.074002 / 23: 0.078311 / 24: 0.082339 / 25: 0.0975 / 27: 0.099372 / 30: 0.10527 / 32: 0.13573 / 33: 0.13573
    # /36: 0.14482 / 39: 0.14971
    mode = [22, 23, 26]
    n_p = [10, 15]
    for m in mode:
        for n in n_p:
            NMSE = main.main(m, n)
            print('NMSE: ', NMSE, 'mode: ', m, 'n_p: ', n)


def gridSearch_epochs():
    epochs = [150, 300, 500]
    steps = [10, 50]
    best_NMSE = 10e5
    best_epochs = 0
    best_steps = 0
    for epoch in epochs:
        for step in steps:

            NMSE = main.main(epoch, step)
            print('NMSE: ', NMSE, 'epochs: ', epoch, 'step: ', step, '')
            if NMSE < best_NMSE:
                best_NMSE = NMSE
                best_epochs = epoch
                best_steps = step

    print('best NMSE:', best_NMSE)
    print('best_epochs: ', best_epochs)
    print('best_steps: ', best_steps)


gridSearch_Relobralo()
