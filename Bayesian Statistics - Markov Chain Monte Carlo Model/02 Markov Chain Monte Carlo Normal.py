"""
Metropolis MCMC for Bayesian inference of tree growth rate.

Problem: Estimate the posterior distribution of the mean annual
diameter increment (theta, cm/year) of beech trees, given noisy
field measurements and a Gaussian prior from the literature.

Because prior and likelihood are both Gaussian, the true posterior
is also Gaussian (conjugate case), so we can check our MCMC result
against the exact analytic answer.
"""

import numpy             as     np
import matplotlib.pyplot as     plt
from   scipy.stats       import norm # for Gaussian pdf and logpdf

rng = np.random.default_rng()

# ---------------------------------------------------------------------------
# 1. Simulate Field Data
# ---------------------------------------------------------------------------
theta_true = 0.52 # true mean increment (cm/year), unknown in real life
sigma_obs  = 0.10 # measurement noise (assumed known), standard deviation (cm/year)
n_obs      = 20   # amount of trees measured

# Assume increments are normally distributed around theta_true
data = rng.normal(theta_true, sigma_obs, n_obs) 

print(f"True mean: {theta_true:.4f} cm/year")
print(f"Observed mean: {data.mean():.4f} cm/year (n={n_obs})")

# ---------------------------------------------------------------------------
# 2. Define prior and log-posterior
# ---------------------------------------------------------------------------
# Prior: theta ~ Normal(mu_0, sigma_0)
# Encodes that "increments are typically around 0.4, with some uncertainty"
mu_0    = 0.40
sigma_0 = 0.15

def log_posterior(theta):
    """
    log P(theta | D)  ∝  log P(D | theta) + log P(theta)
    In log-space to avoid floating-point underflow.
    P(D) is a constant and drops out of all ratios.
    """
    log_like  = np.sum(norm.logpdf(data, loc=theta, scale=sigma_obs))
    log_prior = norm.logpdf(theta, loc=mu_0, scale=sigma_0)
    return log_like + log_prior

# ---------------------------------------------------------------------------
# 3. Metropolis algorithm  (symmetric Gaussian proposal -> ratio = 1)
# ---------------------------------------------------------------------------
N          = 50_000     
sigma_prop = 0.05   # proposal std -- tune until acceptance ~30-40%
burn_in    = 5_000  # discard early steps while chain mixes

samples   = np.zeros(N)
theta_cur = mu_0 # start from the prior mean
lp_cur    = log_posterior(theta_cur)
n_accept  = 0

for t in range(N):
    # Propose a new value by adding Gaussian noise (symmetric proposal)
    # rng.normal draws one sample from Normal distribution
    theta_prop = rng.normal(theta_cur, sigma_prop) # standard dev., not variance
    lp_prop    = log_posterior(theta_prop)
    # Acceptance ratio in log-space is log r = log P(theta' | D) - log P(theta | D)
    # Accept if log(Uniform(0,1)) < log r (identical to u < r but stable)
    log_r = lp_prop - lp_cur
    if np.log(rng.uniform()) < log_r:
        theta_cur = theta_prop
        lp_cur    = lp_prop
        n_accept += 1
    samples[t] = theta_cur

acceptance_rate = n_accept / N
# The acceptance rate is a useful diagnostic: too low means chain is stuck, too high means inefficient exploration.
# It does not converge to 100 percent as we draw from a distribution, not optimize a function.
print(f"Acceptance rate: {acceptance_rate:.1%} (target: 23-45%)")
posterior_samples = samples[burn_in:] # discard burn-in

# ---------------------------------------------------------------------------
# 4. Analytic posterior (conjugate Gaussian, for verification only)
# ---------------------------------------------------------------------------
# With Gaussian likelihood and Gaussian prior, the posterior is theta | D ~ Normal(mu_post, sigma_post^2)
precision_lik = n_obs / sigma_obs**2
precision_pri = 1.0   / sigma_0**2
sigma_post    = 1.0   / np.sqrt(precision_lik + precision_pri)
mu_post       = sigma_post**2 * (data.sum() / sigma_obs**2 + mu_0 / sigma_0**2)

print(f"Analytic posterior:  mean={mu_post:.4f}, std={sigma_post:.4f}")
print(f"MCMC posterior:      mean={posterior_samples.mean():.4f}, "
      f"std={posterior_samples.std():.4f}")
lo, hi = np.percentile(posterior_samples, [2.5, 97.5])
print(f"95% credible interval (MCMC): [{lo:.4f}, {hi:.4f}]")

# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Bayesian inference of beech tree growth rate", fontsize=13)
# --- (a) Trace plot ---
ax = axes[0]
plot_iters = 200  # zoom in to see the actual travel from mu_0 = 0.40
ax.plot(samples[:plot_iters], lw=0.8, color="steelblue", alpha=0.8)
#ax.axvline(burn_in, color="red", ls="--", lw=1.2, label=f"burn-in ({burn_in})")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\theta$  (cm/year)")
ax.set_title("(a) Trace plot (first 200 iterations, burn in is very conservative)")
ax.legend(fontsize=9)
# --- (b) Posterior histogram vs analytic ---
ax = axes[1]
theta_grid = np.linspace(0.15, 0.80, 400)
ax.hist(posterior_samples, bins=60, density=True,
        color="steelblue", alpha=0.5, label="MCMC samples")
ax.plot(theta_grid, norm.pdf(theta_grid, mu_post, sigma_post),
        "r-", lw=2, label="Analytic posterior")
ax.plot(theta_grid, norm.pdf(theta_grid, mu_0, sigma_0),
        "k--", lw=1.5, label="Prior", alpha=0.7)
ax.axvline(theta_true, color="darkgreen", ls=":", lw=1.5, label="True value")
ax.set_xlabel(r"$\theta$  (cm/year)")
ax.set_ylabel("Density")
ax.set_title("(b) Posterior vs prior")
ax.legend(fontsize=9)
# --- (c) Autocorrelation of the chain ---
ax = axes[2]
max_lag = 80
acf = np.array([
    # compare the chain with itself, but shifted by 'lag' steps, to see how correlated they are
    np.corrcoef(posterior_samples[:-lag], posterior_samples[lag:])[0, 1]
    for lag in range(1, max_lag + 1)])
ax.bar(range(1, max_lag + 1), acf, color="steelblue", alpha=0.7, width=0.8)
ax.axhline(0, color="k", lw=0.8)
ax.axhline(1.96 / np.sqrt(len(posterior_samples)), color="red",
           ls="--", lw=1, label="95% confidence band")
ax.axhline(-1.96 / np.sqrt(len(posterior_samples)), color="red", ls="--", lw=1)
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.set_title("(c) Chain autocorrelation")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()