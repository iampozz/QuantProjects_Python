import pandas as pd
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt


def BSM_pricing(S, K, T, r, sigma, call = True):
    """
    Pricing of a plain vanilla option with Black-Scholes-Merton model
    :param S: Spot price of underlying asset
    :param K: Strike price of underlying asset
    :param T: Expiration date in year fraction
    :param r: Risk free rate (annualized)
    :param sigma: Standard deviation (annualized) of the underlying asset
    :param call: If option type is Call (default)
    :return: Option price under BSM model
    """

    # Defining d1 and d2 under BSM model
    d1 = (np.log(S/K) + ((r + 0.5 * sigma ** 2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Pricing formula
    if call:
        opt_price = S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)
    else:
        opt_price = K * np.exp(-r*T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    return opt_price

def CRR_Pricing(S, K, T, r, sigma, n_step, call = True):
    dt = T / n_step  # time increment

    u = np.exp(sigma * np.sqrt(dt))  # up magnitude
    d = 1 / u  # down magnitude
    p = (np.exp(r * dt) - d) / (u - d)  # risk neutral probability

    payoff_tree = np.zeros((n_step + 1, n_step + 1))
    for j in range(n_step + 1):
        ST = S * u ** j * d ** (n_step - j)
        payoff_tree[n_step, j] = max(ST - K, 0) if call else max(K - ST, 0)

    for i in range(n_step - 1, -1, -1):
        for j in range(i + 1):
            payoff_tree[i, j] = np.exp(-r * dt) * (p * payoff_tree[i + 1, j + 1] + (1 - p) * payoff_tree[i + 1, j])

    return payoff_tree[0,0]

S = 100
K = 80
r = 0.01
sigma = 0.25
call = True
T = 1

price = BSM_pricing(S, K, T, r, sigma, call)

# What if sigma changes?

sigmas = np.arange(sigma * 0.5, sigma * 1.5, sigma * 0.05)
prices_call = []
prices_put = []

for sigma in sigmas:
    prices_call.append(BSM_pricing(S, K, T, r, sigma, call))
    prices_put.append(BSM_pricing(S, K, T, r, sigma, False))

prices_call = pd.DataFrame(prices_call, index = sigmas, columns = ['BSM Call Price'])
prices_put = pd.DataFrame(prices_put, index = sigmas, columns = ['BSM Put Price'])

plt.figure()
plt.plot(sigmas, prices_call)
plt.xlabel('Sigma')
plt.ylabel('BSM Option Price')
plt.title('BSM Option Price with different sigma')
plt.legend(prices_call.columns)
plt.show()

plt.figure()
plt.plot(sigmas, prices_put)
plt.xlabel('Sigma')
plt.ylabel('BSM Option Price')
plt.title('BSM Option Price with different sigma')
plt.legend(prices_put.columns)
plt.show()

# What if S changes?
S_multiple = np.arange(S * 0.5, S * 1.5, S * 0.05)

prices_call = []
prices_put = []

for S in S_multiple:
    prices_call.append(BSM_pricing(S, K, T, r, sigma, call))
    prices_put.append(BSM_pricing(S, K, T, r, sigma, False))

prices_call = pd.DataFrame(prices_call, index = S_multiple, columns = ['BSM Call Price'])
prices_put = pd.DataFrame(prices_put, index = S_multiple, columns = ['BSM Put Price'])

plt.figure()
plt.plot(S_multiple, prices_call, color = 'red')
plt.xlabel('Spot Price')
plt.ylabel('BSM Option Price')
plt.title('BSM Option Price with different Spot Price')
plt.legend(prices_call.columns)
plt.show()

plt.figure()
plt.plot(S_multiple, prices_put, color = 'red')
plt.xlabel('Spot Price')
plt.ylabel('BSM Option Price')
plt.title('BSM Option Price with different Spot Price')
plt.legend(prices_put.columns)
plt.show()

# Binomial Tree under CRR model
S = 100
K = 80
r = 0.01
sigma = 0.25
call = call
T = 1
n_step = [10, 100, 1000]

print(f"BSM Price: {price:.4f}")
for step in n_step:
    CRR_price = CRR_Pricing(S, K, T, r, sigma, step, call)
    print(f"CRR Price {step}: {CRR_price:.4f}")