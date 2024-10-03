import main
import config

def gridSearch_Relobralo():
    tmps = [1, 10e-02, 10e-05]  # , 10e-05
    rhos = [0.1, 0.99]
    alphas = [0.1, 0.99]
    best_NMSE = 10e5
    best_tmp = 0
    best_rho = 0
    best_alpha = 0

    for tmp in tmps:
        for rho in rhos:
            for alpha in alphas:

                NMSE = main.main(rho, alpha, tmp)
                print('NMSE: ', NMSE, 'rho: ', rho, 'alpha: ', alpha, 'tmp: ', tmp)
                if NMSE < best_NMSE:
                    best_NMSE = NMSE
                    best_rho = rho
                    best_alpha = alpha
                    best_tmp = tmp

    print('best NMSE:', best_NMSE)
    print('best_temperature: ', best_tmp)
    print('best_rho: ', best_rho)
    print('best_alpha: ', best_alpha)
    print('best_tmp: ', best_tmp)


def gridSearch_modes():
    # 7: 0.0094905 / 8: 0.012216 / 9: 0.016826 / 11: 0.03027 / 13: 0.030792 / 14: 0.033438 / 18: 0.057536
    # /20: 0.06447 / 22: 0.074002 / 23: 0.078311 / 24: 0.082339 / 25: 0.0975 / 27: 0.099372 / 30: 0.10527 / 32: 0.13573 / 33: 0.13573
    # /36: 0.14482 / 39: 0.14971
    modes = [6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
    nkp = [6, 7, 8, 9, 10]
    for m in modes:
        NMSE = main.main(m)
        print('NMSE: ', NMSE)


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


def gridSearch_neurons():
    results = []
    neurons = [100, 128, 150, 180]
    layers = [2]

    for n in neurons:
        for l in layers:
            NMSE = main.main(n, l)
            results.append((NMSE, n, l))

    results.sort(key=lambda x: x[0])

    print("Classifica NMSE:")
    for i, (NMSE, neuron, layer) in enumerate(results):
        print(f"{i + 1}. NMSE: {NMSE}, Neurons: {neuron}, Layers: {layer}")


def gridSearch_lambda():
    results = []
    lam_f = [1, 0.1, 0.01]
    lam_t = [1, 0.1, 0.01]
    lam_o = [0, 1, 0.1, 0.01]

    for f in lam_f:
        for t in lam_t:
            for o in lam_o:
                if f == t == o:
                    continue
                NMSE = main.main(f, t, o)
                results.append((NMSE, f, t, o))

    print("Classifica NMSE:")
    results.sort(key=lambda x: x[0])
    for i, (NMSE, f, t, o) in enumerate(results):
        print(f"{i + 1}. NMSE: {NMSE}, Lam_f: {f}, Lam_t: {t}, Lam_o: {o}")


def gridSearch_nkp():
    results = []
    nkp = [6, 8, 10, 12, 14]
    for n in nkp:
        config.num_known_points = n
        NMSE, NMSE_concatenate = main.main(n)
        results.append((NMSE, NMSE_concatenate, n))

    print("Classifica NMSE:")
    results.sort(key=lambda x: x[0])
    for i, (NMSE, nkp, NMSE_concatenate) in enumerate(results):
        print(f"{i + 1}. mean_NMSE: {NMSE}, NMSE_concatenate: {NMSE_concatenate}, Nkp: {nkp}")


gridSearch_nkp()
