import pandas as pd
import numpy as np

def montecarlo_simulation(S, sigma, r, T, n_step, path):
    """
    Monte-Carlo simulation
    :param S: Spot price of the underlying asset
    :param sigma: volatility of the underlying asset
    :param r: risk-free interest rate
    :param T: Expiration date
    :param n_step: step size for Monte-Carlo simulation
    :param path: number of paths to simulate
    :return:
    """
    np.random.seed(42)
    dt = T/n_step

    sec_path = np.zeros((n_step + 1, path))
    sec_path[0,:] = S

    for p in range(path):
        for n in range(1, n_step + 1):
            z = np.random.randn()
            sec_path[n,p] = sec_path[n - 1,p] * np.exp((r - 0.5 * sigma ** 2)*dt + sigma * np.sqrt(dt) * z)

    return sec_path

def EU_vanilla_option_pricing(prices, K, r, T, call=True):
    path = prices.shape[1]
    payoff = np.zeros(path)
    for p in range(path):
        if call:
            payoff[p] = max(prices[-1, p] - K, 0)
        else:
            payoff[p] = max(K - prices[-1, p], 0)

    price = np.exp(-r * T) * np.mean(payoff)

    return price

def delta_gamma_vega_finitediff(S, K, sigma, r, T, n_step, path, epsilon, call=True):
    """
    Compute delta, gamma and vega with Finite Difference method (Bump and Reval)
    :param vanilla_price: Vanilla option price
    :param epsilon: deviation parameter
    :return: delta,gamma and vega of the option
    """

    mc_price = montecarlo_simulation(S, sigma, r, T, n_step, path)
    mc_price_up = montecarlo_simulation(S + epsilon, sigma, r, T, n_step, path)
    mc_price_down = montecarlo_simulation(S - epsilon, sigma, r, T, n_step, path)

    V = EU_vanilla_option_pricing(mc_price, K, r, T, call=call)
    V_up = EU_vanilla_option_pricing(mc_price_up, K, r, T, call=call)
    V_down = EU_vanilla_option_pricing(mc_price_down, K, r, T, call=call)

    delta = (V_up - V_down) / epsilon
    gamma = (V_up - 2 * V + V_down) / np.square(epsilon)

    mc_price_sigma_up = montecarlo_simulation(S, sigma + epsilon, r, T, n_step, path)

    V_sigma_up = EU_vanilla_option_pricing(mc_price_sigma_up, K, r, T, call=call)

    vega = (V_sigma_up - V) / epsilon

    greeks = pd.DataFrame({'Option Price': V,'Delta': delta, 'Gamma': gamma, 'Vega': vega}, index = [f"Option_{epsilon}"]).T
    return greeks

S = 100
K = 90
sigma = 0.25
r = 0.01
T = 1
call = True
n_step = 252
path = 10_000
epsilons = [0.1, 0.01, 0.001, 0.0001]

greeks = []
for epsilon in epsilons:
    greeks.append(delta_gamma_vega_finitediff(S, K, sigma, r, T, n_step, path, epsilon, call=call))

greeks_df = pd.concat(greeks, axis = 1)
print(greeks_df)

# Computation of Delta using Pathwise method

def delta_pathwise(S, price_mc, K, r, T, call):
    """
    Compute the delta of an option usign Pathwise method
    :param S: Spot price
    :param price_mc: montecarlo price
    :param K: Strike Price
    :param r: risk-free interest rate
    :param T: Expiration date
    :param call: If the option is call or put (True or False)
    :return: Delta of option
    """
    ST = price_mc[-1,:]
    if call:
        payoff = np.where(ST > K, ST / S, 0.0)
    else:
        payoff = np.where(ST < K, -ST / S, 0.0)

    delta = np.exp(-r*T) * np.mean(payoff)

    return delta

montecarlo_price = montecarlo_simulation(S, sigma, r, T, n_step, path)
pathwise_delta = delta_pathwise(S, montecarlo_price, K, r, T, call)
print(f"Pathwise Delta: {pathwise_delta:.4f}")

# Computation of Vega using LRM (Likelihood Ratio Method)

def vega_LRM(S, K, sigma, r, T, n_step, path, call=True):
    dt = T / n_step
    ST = np.zeros(path)
    payoff = np.zeros(path)
    Z_all = np.zeros(path)

    for p in range(path):
        S_t = S
        Z_total = 0

        for _ in range(n_step):
            z = np.random.randn()
            S_t *= np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
            Z_total += z

        ST[p] = S_t
        Z_all[p] = Z_total / np.sqrt(n_step)  # average for central limit approx

        if call:
            payoff[p] = max(ST[p] - K, 0)
        else:
            payoff[p] = max(K - ST[p], 0)

    likelihood_term = (Z_all**2 - 1) / sigma
    vega = np.exp(-r * T) * np.mean(payoff * likelihood_term)

    return vega

vega_lrm = vega_LRM(S, K, sigma, r, T, n_step, path, call=True)
print(f"LRM Vega: {vega_lrm:.4f}")

