import main


def gridSearch_Relobralo():
    # tmps = [10, 0.1, 0.00001]
    rhos = [0.1, 0.5, 0.99]
    alphas = [0.1, 0.5, 0.99]
    best_NMSE = 10e5
    best_tmp = 0
    best_rho = 0
    best_alpha = 0

    for rho in rhos:
        for alpha in alphas:

            NMSE = main.main(rho, alpha)
            print('NMSE: ', NMSE, 'rho: ', rho, 'alpha: ', alpha, '')
            if NMSE < best_NMSE:
                best_NMSE = NMSE
                best_rho = rho
                best_alpha = alpha

    print('best NMSE:', best_NMSE)
    print('best_temperature: ', best_tmp)
    print('best_rho: ', best_rho)
    print('best_alpha: ', best_alpha)


def gridSearch_modes():
    # 7: 0.0094905 / 8: 0.012216 / 9: 0.016826 / 11: 0.03027 / 13: 0.030792 / 14: 0.033438 / 18: 0.057536
    # /20: 0.06447 / 22: 0.074002 / 23: 0.078311 / 24: 0.082339 / 25: 0.0975 / 27: 0.099372 / 30: 0.10527 / 32: 0.13573 / 33: 0.13573
    # /36: 0.14482 / 39: 0.14971
    mode = [22, 23, 25, 27]
    freq = [15.709, 16.599, 20.323, 20.972]
    for m, f in zip(mode, freq):
        NMSE = main.main(m, f)
        print('NMSE: ', NMSE, 'mode: ', m, 'freq: ', f)


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
