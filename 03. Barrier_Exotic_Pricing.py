import pandas as pd
import numpy as np
import scipy.stats as stats

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

def EU_downandout_pricing(prices, B, K, r, T, call=True):
    """
    EU down-and-out exotic option pricing
    :param prices: montecarlo simulated prices
    :param B: barrier price
    :param K: strike price
    :param r: risk-free interest rate
    :param T: expiration date
    :param call: boolean indicating whether it is a call option or not
    :return: Down-and-out exotic option price
    """
    path = prices.shape[1]
    payoff = np.zeros(path)

    for p in range(path):
        if np.min(prices[:, p]) <= B:
            payoff[p] = 0  # Knocked out
        else:
            if call:
                payoff[p] = max(prices[-1, p] - K, 0)
            else:
                payoff[p] = max(K - prices[-1, p], 0)

    price = np.exp(-r * T) * np.mean(payoff)
    return price

def EU_downandin_pricing(prices, B, K, r, T, call=True):
    """
    EU down-and-in exotic option pricing
    :param prices: Monte Carlo simulated price paths (array shape: [steps+1, paths])
    :param B: barrier level (activation threshold)
    :param K: strike price
    :param r: risk-free interest rate
    :param T: maturity (in years)
    :param call: True for Call, False for Put
    :return: Present value of the Down-and-In European option
    """
    path = prices.shape[1]
    payoff = np.zeros(path)

    for p in range(path):
        if np.min(prices[:, p]) <= B:
            if call:
                payoff[p] = max(prices[-1, p] - K, 0)
            else:
                payoff[p] = max(K - prices[-1, p], 0)
        else:
            payoff[p] = 0

    price = np.exp(-r * T) * np.mean(payoff)
    return price

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

S = 100
B = 80
K = 90
sigma = 0.25
r = 0.01
T = 1
call = True
n_step = 252
path = 10_000

montecarlo_price = montecarlo_simulation(S, sigma, r, T, n_step, path)

# Pricing Down-and-Out Asian option
option_price_dao = EU_downandout_pricing(montecarlo_price, B, K, r, T, call)
print(f"Down-and-Out Option Price: {option_price_dao:.4f}")

# Pricing Down-and-In Asian option
option_price_dai = EU_downandin_pricing(montecarlo_price, B, K, r, T, call)
print(f"Down-and-Out Option Price: {option_price_dai:.4f}")

# Vanilla option pricing and check
vanilla_price = EU_vanilla_option_pricing(montecarlo_price, K, r, T, call)
print(f"Vanilla Option Price: {vanilla_price:.4f}")
print("Check -> Vanilla option Price = Down-and-In + Down-and-Out")
print(f"Check Value: {option_price_dao+option_price_dai:.4f}")

# For Up-and-In/Out Asian option mechanism is the same, just change the logic of the threshold barrier

#======================================================================================================================#

# Lookback Option - Fixed Strike
# Fixed Strike Lookback Call Payoff = max(S_max - K, 0)

def EU_lookbackfixed_pricing(prices,  K, r, T, call=True):
    """
    EU lookback fixed strike pricing
    :param prices: montecarlo simulated prices
    :param K: strike price
    :param r: risk-free interest rate
    :param T: expiration date
    :param call: boolean indicating whether it is a call option or not
    :return: lookback fixed strike option price
    """
    max_S = np.max(prices, axis=0)
    if call:
        payoff = np.maximum(max_S - K, 0)
    else:
        payoff = np.maximum(K - max_S, 0)

    price = np.exp(-r * T) * np.mean(payoff)
    return price

lookback_fixed_price = EU_lookbackfixed_pricing(montecarlo_price, K, r, T, call)
print(f"\nLookback Fixed Option Price: {lookback_fixed_price:.4f}")

# Lookback Floating Strike -> payoff_call = (S_T - min(S_t)), payoff_put = (max(S_t) - S_T)

def EU_lookbackfloating_pricing(prices, r, T, call=True):
    """
    EU Lookback floating strike option pricing.
    :param prices: Monte Carlo simulated prices
    :param r: risk-free rate
    :param T: maturity
    :param call: True if call, False if put
    :return: option price
    """
    final_price = prices[-1, :]
    if call:
        strike = np.min(prices, axis=0)
        payoff = np.maximum(final_price - strike, 0)
    else:
        strike = np.max(prices, axis=0)
        payoff = np.maximum(strike - final_price, 0)

    price = np.exp(-r * T) * np.mean(payoff)

    return price

lookback_floating_price = EU_lookbackfloating_pricing(montecarlo_price, r, T, call)
print(f"Lookback Floating Option Price: {lookback_floating_price:.4f}")

# Asian Option Pricing

def EU_asianfixed_pricing(prices, K, r, T, call=True):
    """
    Asian option pricing with fixed strike and arithmetic average.
    """
    avg_price = np.mean(prices, axis=0)
    if call:
        payoff = np.maximum(avg_price - K, 0)
    else:
        payoff = np.maximum(K - avg_price, 0)

    price = np.exp(-r * T) * np.mean(payoff)
    return price

asian_fixed = EU_asianfixed_pricing(montecarlo_price, K, r, T, call=True)

def EU_asianfloating_pricing(prices, r, T, call=True):
    """
    Asian option pricing with floating strike and arithmetic average.
    """
    avg_price = np.mean(prices, axis=0)
    final_price = prices[-1, :]
    if call:
        payoff = np.maximum(final_price - avg_price, 0)
    else:
        payoff = np.maximum(avg_price - final_price, 0)

    price = np.exp(-r * T) * np.mean(payoff)
    return price

asian_floating = EU_asianfloating_pricing(montecarlo_price, r, T, call=True)

print(f"Asian Fixed Strike Call Price: {asian_fixed:.4f}")
print(f"Asian Floating Strike Call Price: {asian_floating:.4f}")

# CLiquet Option pricing
def EU_cliquet_pricing(prices, r, T , cap, floor):
    """
    Euroean Clique option pricing.
    :param prices: prices of the underlying
    :param r: risk-free interest rate
    :param T: expiration date
    :param cap: upper bound of rate of return
    :param floor: lower bound of rate of return
    :return: price of an Euroean Clique option
    """

    returns = pd.DataFrame(prices).pct_change().dropna().to_numpy()

    adjusted_returns = np.clip(returns, floor, cap)

    cliquet_returns = np.sum(adjusted_returns, axis = 0)

    payoff = np.maximum(cliquet_returns, 0)

    price = np.exp(-r * T) * np.mean(payoff)

    return price

option_price_cliquet = EU_cliquet_pricing(montecarlo_price, r, T, cap=0.05, floor=0.0)
print(f"Cliquet Option Price: {option_price_cliquet:.4f}")

# Rainbow Option Pricing
# Best-of European option -> payoff = max(max(S1_T, S2_T, ...) - K, 0)
# We need to simulate multiple paths for multiple assets, since Rainbow Options works on a basket of N assets

def simulate_multiple_asssets(S0_vec, simgas_vec, r, T, n_step, n_path, corr_matrix = None):
    """
    Simulates Monte Carlo trajectories for multiple correlated assets.
    :param S_vec: initial price vector (shape: N)
    :param sigma_vec: volatility of underlyings (shape: N)
    :param r: risk-free rate
    :param T: time to maturity
    :param n_step: number of time steps
    :param n_path: number of simulations
    :param corr_matrix: correlation matrix between assets (NxN), optional
    :return: array shape (n_step+1, n_path, N)
    """
    N = len(S0_vec)
    dt = T / n_step
    prices = np.zeros((n_step + 1, n_path, N))
    prices[0, :, :] = S0_vec

    if corr_matrix is None:
        corr_matrix = np.eye(N)
    L = np.linalg.cholesky(corr_matrix)

    for t in range(1, n_step + 1):
        z = np.random.randn(n_path, N)
        correlated_z = z @ L.T
        drift = (r - 0.5 * np.array(simgas_vec)**2) * dt
        diffusion = np.array(simgas_vec) * np.sqrt(dt) * correlated_z
        prices[t] = prices[t - 1] * np.exp(drift + diffusion)

    return prices

def best_of_option_price(prices, K_vec, r, T, call=True):
    """
    Best-of option with a different strike per asset
    :param prices: shape (n_step+1, n_path, N_assets)
    :param K_vec: array of strike prices, shape (N_assets,)
    """
    final_prices = prices[-1, :, :]  # shape (n_path, N_assets)

    if call:
        payoffs = final_prices - K_vec  # broadcasting: each column - corresponding K
    else:
        payoffs = K_vec - final_prices

    payoffs = np.maximum(payoffs, 0)  # ensure non-negative

    best_payoff = np.max(payoffs, axis=1)  # best payoff for each path

    price = np.exp(-r * T) * np.mean(best_payoff)
    return price

S_vec = [100, 95, 50 , 25]
sigma_vec = [0.2, 0.25, 0.30, 0.15]
corr_matrix = [[1.0, 0.5, 0.1, -0.2],
               [0.5, 1.0, 0.3, 0.6],
               [0.1, 0.3, 1.0, 0.25],
               [-0.2, 0.6, 0.25, 1.0]]
K_vec = [90, 80, 45, 20]
r = 0.01
T = 1
n_step = 252
n_path = 10_000

prices = simulate_multiple_asssets(S_vec, sigma_vec, r, T, n_step, n_path, corr_matrix)
best_of_price = best_of_option_price(prices, K_vec, r, T, call)
print(f"Best-of Call Option Price: {best_of_price:.4f}")

# Worst-of European option -> payoff = max(min(S1_T, S2_T, ...) - K, 0)
def worst_of_option_price(prices, K_vec, r, T, call=True):
    final_prices = prices[-1, :, :]  # shape (n_path, N_assets)

    if call:
        payoffs = final_prices - K_vec
    else:
        payoffs = K_vec - final_prices

    payoffs = np.maximum(payoffs, 0)

    worst_payoff = np.min(payoffs, axis=1)  # worst payoff for each path

    price = np.exp(-r * T) * np.mean(worst_payoff)
    return price

worst_of_price = worst_of_option_price(prices, K_vec, r, T, call)
print(f"Worst-of Call Option Price: {worst_of_price:.4f}")