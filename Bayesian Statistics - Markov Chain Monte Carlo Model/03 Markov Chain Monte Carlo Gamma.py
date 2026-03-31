import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Simulate field data
# Tree ring widths are strictly positive and right-skewed -> Gamma is natural
# Gamma(shape, scale): mean = shape * scale, variance = shape * scale^2
# ---------------------------------------------------------------------------
shape_true = 5.0   # true shape parameter (unknown in real life)
scale_true = 0.10  # true scale parameter (assumed known here, like sigma before)
n_obs      = 20

data = rng.gamma(shape_true, scale_true, n_obs)
print(f"True mean:     {shape_true * scale_true:.4f} cm/year")
print(f"Observed mean: {data.mean():.4f} cm/year")

# ---------------------------------------------------------------------------
# 2. Prior and log-posterior
# We estimate the shape parameter theta = shape, scale assumed known.
# Prior: Gaussian on theta (could also use Gamma, but Gaussian is fine
#        as long as we stay in positive territory, which we check below)
# ---------------------------------------------------------------------------
mu_0    = 2.0   # prior belief: shape is around 2
sigma_0 = 1.5   # fairly uncertain

def log_posterior(theta):
    if theta <= 0:           # Gamma shape must be positive; reject outright
        return -np.inf
    log_like  = np.sum(gamma.logpdf(data, a=theta, scale=scale_true))
    log_prior = norm.logpdf(theta, loc=mu_0, scale=sigma_0)
    return log_like + log_prior

# ---------------------------------------------------------------------------
# 3. Metropolis algorithm — identical to before
# ---------------------------------------------------------------------------
N          = 50000
sigma_prop = 0.3
burn_in    = 5000

samples   = np.zeros(N)
theta_cur = mu_0
lp_cur    = log_posterior(theta_cur)
n_accept  = 0

for t in range(N):
    theta_prop = rng.normal(theta_cur, sigma_prop)
    lp_prop    = log_posterior(theta_prop)
    if np.log(rng.uniform()) < lp_prop - lp_cur:
        theta_cur = theta_prop
        lp_cur    = lp_prop
        n_accept += 1
    samples[t] = theta_cur

posterior_samples = samples[burn_in:]
print(f"Acceptance rate: {n_accept/N:.1%}")
print(f"MCMC posterior:  mean={posterior_samples.mean():.4f}, "
      f"std={posterior_samples.std():.4f}")
lo, hi = np.percentile(posterior_samples, [2.5, 97.5])
print(f"95% credible interval: [{lo:.4f}, {hi:.4f}]")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Bayesian inference of Gamma shape parameter", fontsize=13)

ax = axes[0]
ax.plot(samples[:200], lw=0.5, color="steelblue", alpha=0.8)
#ax.axvline(burn_in, color="red", ls="--", lw=1.2, label=f"burn-in ({burn_in})")
ax.axhline(shape_true, color="darkgreen", ls=":", lw=1.5, label="True value")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\theta$ (shape)")
ax.set_title("(a) Trace plot")
ax.legend(fontsize=9)

ax = axes[1]
theta_grid = np.linspace(1, 10, 400)
ax.hist(posterior_samples, bins=60, density=True,
        color="steelblue", alpha=0.5, label="MCMC posterior")
ax.plot(theta_grid, norm.pdf(theta_grid, mu_0, sigma_0),
        "k--", lw=1.5, label="Prior", alpha=0.7)
ax.axvline(shape_true, color="darkgreen", ls=":", lw=1.5, label="True value")
ax.axvline(posterior_samples.mean(), color="red", ls="-", lw=1.5,
           label=f"Posterior mean ({posterior_samples.mean():.2f})")
ax.set_xlabel(r"$\theta$ (shape parameter)")
ax.set_ylabel("Density")
ax.set_title("(b) Posterior vs prior")
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()