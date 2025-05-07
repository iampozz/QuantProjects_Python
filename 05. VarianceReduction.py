import numpy as np

np.random.seed(50)
# Antithetic Variates Method
def montecarlo_antithetic_pricing(S, K, sigma, r, T, n_step, path, call = True):
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
    dt = T/n_step

    sec_path_pos = np.zeros((n_step + 1, path))
    sec_path_pos[0,:] = S

    sec_path_neg = np.zeros((n_step + 1, path))
    sec_path_neg[0,:] = S

    for p in range(path):
        for n in range(1, n_step + 1):
            z = np.random.randn()
            sec_path_pos[n,p] = sec_path_pos[n - 1,p] * np.exp((r - 0.5 * sigma ** 2)*dt + sigma * np.sqrt(dt) * z)
            sec_path_neg[n,p] = sec_path_pos[n - 1,p] * np.exp((r - 0.5 * sigma ** 2)*dt + sigma * np.sqrt(dt) * -z)

    ST_pos = sec_path_pos[-1,:]
    ST_neg = sec_path_neg[-1,:]
    if call:
        mean_payoff_pos = np.mean(np.maximum(ST_pos - K, 0))
        mean_payoff_neg = np.mean(np.maximum(ST_neg - K, 0))

        mean_payoff = (mean_payoff_neg + mean_payoff_pos) / 2

    else:
        mean_payoff_pos = np.mean(np.maximum(K - ST_pos, 0))
        mean_payoff_neg = np.mean(np.maximum(K - ST_neg, 0))

        mean_payoff = (mean_payoff_neg + mean_payoff_pos) / 2

    price = np.exp(-r*T) * mean_payoff

    return price

S = 100
K = 90
sigma = 0.25
r = 0.01
T = 1
call = True
n_step = 252
path = 10_000

antithetic_price = montecarlo_antithetic_pricing(S, K, sigma, r, T, n_step, path, call = call)
print(f"Antithetic Variance Reduction price: {antithetic_price:.4f}")

# Variance reduction check vs standard montecarlo method
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

n_batch = 10

standard_prices = []

for _ in range(n_batch):
    prices = montecarlo_simulation(S, sigma, r, T, n_step, path)
    price = EU_vanilla_option_pricing(prices, K, r, T, call=call)
    standard_prices.append(price)

antithetic_prices = []

for _ in range(n_batch):
    price = montecarlo_antithetic_pricing(S, K, sigma, r, T, n_step, path, call=call)
    antithetic_prices.append(price)

mean_std = np.mean(standard_prices)
var_std = np.var(standard_prices)

mean_anti = np.mean(antithetic_prices)
var_anti = np.var(antithetic_prices)

print(f"Standard MC - Mean: {mean_std:.4f}, Variance: {var_std:.6f}")
print(f"Antithetic MC - Mean: {mean_anti:.4f}, Variance: {var_anti:.6f}")

