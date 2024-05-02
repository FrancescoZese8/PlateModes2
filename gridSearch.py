import main

tmps = [10, 0.1, 0.00001]
rhos = [0.5, 0.99]
alphas = [0.5, 0.9, 0.9999]
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

print('best NMSE:', best_NMSE)
print('best_temperature: ', best_tmp)
print('best_rho: ', best_rho)
print('best_alpha: ', best_alpha)
