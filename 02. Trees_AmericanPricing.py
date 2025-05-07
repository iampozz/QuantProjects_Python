import pandas as pd
import numpy as np

def CRR_american(S, K, T, r, sigma, n_step, call = True):
    """
    Coxx-Ross-Rubenstein binomial tree for american option
    :param S: Spot price
    :param K: Strike price
    :param T: Expiration date
    :param r: risk free interest rate
    :param sigma: volatility of underlying asset
    :param n_step: number of steps of the tree
    :param call: if the option is call or put
    :return:
    """
    dt = T / n_step

    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    S_tree = np.zeros((n_step + 1, n_step + 1))
    S_tree[0, 0] = S

    for i in range(n_step + 1):
        for j in range(i + 1):
            S_tree[i, j] = S * u ** j * d ** (i - j)

    payoff_tree = np.zeros((n_step + 1, n_step + 1))
    for j in range(n_step + 1):
        payoff_tree[n_step, j] = max(S_tree[n_step, j] - K, 0) if call else max(K - S_tree[n_step, j], 0)

    for i in range(n_step - 1, -1, -1):
        for j in range(i + 1):
            continuation = np.exp(-r * dt) * (
                    p * payoff_tree[i + 1, j + 1] + (1 - p) * payoff_tree[i + 1, j]
            )
            exercise = max(S_tree[i, j] - K, 0) if call else max(K - S_tree[i, j], 0)
            payoff_tree[i, j] = max(continuation, exercise)
    return payoff_tree[0,0]

# American Option Pricing under CRR
S = 100
K = 80
r = 0.01
sigma = 0.25
call = True
T = 1
n_step = [10, 100, 1000]

for step in n_step:
    CRR_price = CRR_american(S, K, T, r, sigma, step, call)
    print(f"CRR American Price {step}: {CRR_price:.4f}")


