# Quantitative Derivatives and Option Pricing Toolkit

## 📌 Overview

This project is a modular and extensible Python toolkit for developing and testing pricing models for vanilla, American, and exotic options. It aims to replicate, validate, and extend academic and industry-standard methodologies in option pricing and risk-neutral simulation — with a clear quantitative finance foundation.

The code and development are guided by practical modeling experience and rigorous mathematical theory, building toward a professional-level quantitative finance toolkit.

---

## 📚 Topics Covered (So Far)

### ✅ **Vanilla Option Pricing**
- European call and put options using Black-Scholes-Merton (closed-form)
- Pricing with binomial trees (Cox-Ross-Rubinstein model)
- Convergence of CRR to BSM prices
- Sensitivity analysis to volatility and spot price

### ✅ **American Option Pricing**
- American calls and puts via CRR trees
- Early exercise logic integrated in backward induction
- Demonstration of when American ≠ European (put options)

### ✅ **Barrier Options**
- Monte Carlo simulation for European **Down-and-Out** and **Down-and-In** options
- Geometric Brownian motion path generation
- Barrier monitoring and knockout/in logic
- Discussion of how barrier and strike affect value
- Check to assess the sanity of the result (i.e. Vanilla option Price = Down-and-In + Down-and-Out)

### ✅ **Exotic Options**
- Pricing of different Exotic options:
  - **Lookback Option - Fixed Strike**
  - **Lookback Option - Floating Strike**
  - **Asian Option - Fixed Strike**
  - **Asian Option - Floating Strike**

### ✅ **Options Greeks Estimation**
- Estimation of Plain Vanilla Option Greeks:
  - Finite Difference Method to estimate Delta, Gamma and Vega (Bump and Reval)
  - Estimation of Delta using Pathwise method
  - Estimation of Vega using LRM (Likelihood Ratio Method)

### ✅ **Variance Reduction Techniques**
- Variance reduction for Montecarlo Methods applied to option pricing:
  - Antithetic Variates method 
---

## 🧠 Goals of the Project

- Build an **educational and practical pricing engine**
- Reinforce **theoretical intuition** with code-based evidence
- Structure code for future integration with:
  - Stochastic volatility models (Heston, SABR)
  - Greeks calculation
  - Structured products and path-dependent payoffs
  - Calibration tools and implied volatility extraction
- 