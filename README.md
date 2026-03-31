# Ecological Data Science: Basic Statistical Methods

In this repository I compiled self-made educational materials for learning selected topics in ecological data science. 
Each topic is thoroughly introduced in LaTeX Documents, and extensively commented Python Code with example data is provided.

## Overview

The following frameworks are discussed.

### 1. From Linear to Generalized to Total Regression

**Linear Regression** starts with the classical Gauss normal equations, solving `y = Ab` via QR decomposition. We reinterpret this as maximum likelihood estimation (MLE) where the data itself is simply a pointwise mean over time of a gaussian distribution, the more statistical way of thinking rather than treating the model as some ground-truth data plus additive noise.

**Generalized Linear Models (GLMs)** extend this by replacing the Gaussian assumption with other distributions from the exponential family (e.g., Gamma for positive ecological measurements), connected through a link function. The Gamma GLM with log link is particularly useful when data span multiple orders of magnitude.

**Total Least Squares & Total GLMs** account for measurement errors in the input features themselves, not only the response variable. Standard regression assumes the design matrix `A` is perfectly measured; TLS corrects both `A` and `y` using SVD. Total GLMs combine this with arbitrary likelihood functions through block coordinate descent, the latter is not too intensively discussed.

**Implementation**: Custom Python classes for each method, demonstrating QR decomposition, gradient ascent with Armijo line search, SVD-based corrections, and errors-in-variables formulations.

### 2. Bayesian Inference and Markov Chain Monte Carlo

**Bayesian Statistics** treats parameters as distributions rather than point estimates. For tree growth modeling, we encode prior knowledge from literature, update it with field data, and obtain full posterior distributions that quantify uncertainty.

**Markov Chain Monte Carlo (MCMC)** solves the intractability of the marginal likelihood `P(D)` by sampling from the assumed posterior distribution. The Metropolis-Hastings algorithm constructs a Markov chain whose stationary distribution is the target posterior, using only the likelihood and prior (the evidence cancels in acceptance ratios).

**Implementation**: Two MCMC examples from scratch in NumPy: tree diameter increment (Normal likelihood, conjugate case with analytic verification) and ring widths (Gamma likelihood, non-conjugate case). Includes diagnostics: trace plots, autocorrelation, acceptance rates.



## References

- **Brooks, S., Gelman, A., Jones, G., & Meng, X.-L. (Eds.). (2011).** *Handbook of Markov Chain Monte Carlo*. Chapman & Hall/CRC.  
  *In-depth coverage of advanced MCMC techniques.*

- **Clark, J. S. (2007).** *Models for Ecological Data: An Introduction*. Princeton University Press.  
  *Bayesian methods tailored for ecology.*

- **Carroll, R. J., Ruppert, D., Stefanski, L. A., & Crainiceanu, C. M.** (2006). *Measurement Error in Nonlinear Models: A Modern Perspective* (2nd ed.). Chapman & Hall / CRC.

All code is pedagogical, prioritizing transparency over efficiency.