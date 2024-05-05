import main


def gridSearch_Relobralo():
    # tmps = [10, 0.1, 0.00001]
    rhos = [0.8, 0.9, 0.99, ]
    alphas = [0.8, 0.9, 0.99]
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
    mode = [11, 18, 25, 36]
    freq = [0.03027, 0.057536, 0.0975, 0.14482]
    for m, f in zip(mode, freq):
        NMSE = main.main(m, f)


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


gridSearch_epochs()
