import numpy as np
import matplotlib.pyplot as plt

class TotalGammaGLM:
    """
    Total Gamma GLM with log link, accounting for measurement errors in both A and y.
    # -------------------------------------------------------------------------------
    # THE MODEL ---------------------------------------------------------------------
    # -------------------------------------------------------------------------------
    Design matrix : A_obs in R^{m x n} (observed, noisy feature matrix)
    True inputs   : A_hat in R^{m x n} (latent, corrected feature matrix)
    Parameters    : b in R^{n+1}       (b[0] = bias, b[1:] = weights)
    Targets       : y in R^m           (Gamma-distributed, y_i > 0)
    Assumed Model : mu_i = exp(b[0] + a_hat_i^T b[1:]) (log link)
    # -------------------------------------------------------------------------------
    # THE OBJECTIVE (joint log-likelihood, dropping constants) ----------------------
    # -------------------------------------------------------------------------------
    J(b, {a_hat_i}) = -k * sum_i [a_hat_aug_i^T b + y_i * exp(-a_hat_aug_i^T b)] - (k / (2 * sigma_x^2)) * sum_i || a_obs_i - a_hat_i ||^2
    The first term is the Gamma-distributed log-likelihood, the second Gaussian log-likelihood for the input errors, and
    log p(a_obs_i | a_hat_i) = const - (1 / 2 * sigma_x^2) * ||a_obs_i - a_hat_i||^2
    # -------------------------------------------------------------------------------
    # BLOCK COORDINATE DESCENT ------------------------------------------------------
    # -------------------------------------------------------------------------------
    dJ/d(a_hat_i) = -k * b[1:] * (1 - y_i / mu_i) + 2 * lambda * (a_obs_i - a_hat_i) = 0
    <=> a_hat_i* = a_obs_i - (k / 2*lambda) * (1 - y_i / mu_i) * b[1:]
    Each corrected input shifts from a_obs_i along the slope direction b[1:], scaled by the relative residual (1 - y_i/mu_i).
    Using this result, dJ/db = k * A_hat_aug^T (y/mu - 1), same as standard GLM but with hat correction.
    Alternating between these two steps guarantees J increases monotonically towards a stationary point (global optimum not guaranteed: non-convex).
    # -------------------------------------------------------------------------------
    # PARAMETERS --------------------------------------------------------------------
    # -------------------------------------------------------------------------------
    k               : Gamma shape parameter (known, controls noise level in y)
    lam             : Precision ratio lambda = k / (2 * sigma_x^2); governs how much the corrected x is allowed to deviate from the observed x.
    max_iter        : Maximum outer iterations (each = one b-step + one a_hat-step)
    max_iter_b      : Maximum gradient-ascent steps in the inner b-update
    tol             : Convergence tolerance on ||grad_b J||
    alpha_0, rho, c : Armijo backtracking parameters (initial step, shrinkage, slope)
    """
    def __init__(self, k=1.0, lam=1.0, max_iter=200, max_iter_b=5000, tol=1e-7, alpha_0=1.0, rho=0.5, c=1e-4):
        self.k         = k
        self.lam       = lam
        self.max_iter  = max_iter
        self.max_iter_b = max_iter_b
        self.tol       = tol
        self.alpha_0   = alpha_0
        self.rho       = rho
        self.c         = c
        self.b         = None # Fitted parameters (bias + weights)
        self.A_hat     = None # Corrected (latent) input matrix
        self.history   = []   # Joint objective value per outer iteration
    def augment(self, A):
        """Prepend a column of ones: [1 | A]."""
        m = A.shape[0]
        return np.hstack([np.ones((m, 1)), A])
    def mu(self, A_hat_aug):
        """Mean prediction via log link: mu_i = exp(a_hat_aug_i^T b)."""
        return np.exp(A_hat_aug @ self.b)
    def objective(self, A_hat_aug, y):
        """Joint objective J(b, {a_hat_i}), dropping constants."""
        eta  = A_hat_aug @ self.b
        A_   = A_hat_aug[:, 1:] # strip bias column to get the feature part
        A_obs_ = self.A_obs     # observed features (no bias column)
        gamma_ll  = -self.k * np.sum(eta + y * np.exp(-eta))
        x_penalty = -self.lam * np.sum((A_obs_ - A_) ** 2)
        return gamma_ll + x_penalty
    def grad_b(self, A_hat_aug, y):
        """Gradient of J w.r.t. b: k * A_hat_aug^T (y/mu - 1)."""
        mu = self.mu(A_hat_aug)
        return self.k * (A_hat_aug.T @ (y / mu - 1.0))
    def armijo_b(self, A_hat_aug, y, grad):
        """Armijo backtracking line search for the b-step."""
        alpha   = self.alpha_0
        j_curr  = self.objective(A_hat_aug, y)
        slope   = self.c * np.dot(grad, grad)
        for _ in range(100):
            b_new         = self.b + alpha * grad
            b_old, self.b = self.b, b_new
            with np.errstate(over='ignore', invalid='ignore'):
                j_new = self.objective(A_hat_aug, y)
            self.b = b_old
            j_new = -np.inf if not np.isfinite(j_new) else j_new
            if j_new >= j_curr + alpha * slope:
                break
            alpha *= self.rho
        return alpha
    # -------------------------------------------------------------------------
    def fit(self, A: np.ndarray, y: np.ndarray):
        """
        Block coordinate descent: alternate between a b-step and a closed-form a_hat-step.
        Inputs:
            A : observed feature matrix in R^{m x n} (without bias column)
            y : target vector in R^m, all entries > 0
        Output: self (fitted parameters b and corrected inputs A_hat)
        """
        m, n     = A.shape
        self.A_obs = A.copy()  # store observed A for the x-penalty
        self.A_hat  = A.copy() # initialise corrected inputs at observed values
        A_hat_aug   = self.augment(self.A_hat)
        # Initialise b: bias at log(mean(y)), weights at zero
        self.b = np.zeros(n + 1)
        self.b[0] = np.log(np.mean(y))
        self.history = []
        for outer in range(self.max_iter):
            self.history.append(self.objective(A_hat_aug, y))
            # --- b-step: gradient ascent with Armijo line search ---
            for _ in range(self.max_iter_b):
                grad = self.grad_b(A_hat_aug, y)
                if np.linalg.norm(grad) < self.tol:
                    break
                alpha  = self.armijo_b(A_hat_aug, y, grad)
                self.b = self.b + alpha * grad
            # --- a_hat-step: closed-form correction for each row ---
            # a_hat_i* = a_obs_i - (k / 2*lambda) * (1 - y_i / mu_i) * b[1:]
            mu          = self.mu(A_hat_aug)
            residuals   = (1.0 - y / mu)[:, np.newaxis]   # shape (m, 1)
            self.A_hat  = self.A_obs - (self.k / (2.0 * self.lam)) * residuals * self.b[1:]
            A_hat_aug   = self.augment(self.A_hat)
            # Convergence check on outer gradient norm
            grad_norm = np.linalg.norm(self.grad_b(A_hat_aug, y))
            if grad_norm < self.tol:
                print(f"Converged at outer iteration {outer} (||grad_b J|| < {self.tol})")
                break
    def predict(self, A: np.ndarray):
        """
        Predict mean targets for new (test) inputs using corrected parameters.
        At prediction time, no x-correction is applied (latent a_hat is unknown);
        the fitted b encodes the relationship for future observed inputs.
        Output = mu = exp(A_aug @ b)
        """
        return np.exp(self.augment(A) @ self.b)
    def score(self, A: np.ndarray, y: np.ndarray):
        """
        R² on the log scale (natural for multiplicative / Gamma models).
        R² = 1 - SS_res / SS_tot, where residuals are in log(y) space.
        R² = 1 is a perfect fit, 0 means no better than the log-mean.
        """
        log_y   = np.log(y)
        log_mu  = np.log(self.predict(A))
        ss_res  = np.sum((log_y - log_mu) ** 2)
        ss_tot  = np.sum((log_y - log_y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot

def example():
    rng      = np.random.default_rng()
    x_true   = np.linspace(1, 5, 60)
    # True model: log(mu) = 0.5 * x + 0.2  =>  b_true = [0.2, 0.5]
    k        = 10
    mu_true  = np.exp(0.5 * x_true + 0.2)
    # Add noise to BOTH x and y
    sigma_x  = 0.3
    x_noisy  = x_true + rng.normal(0, sigma_x, size=len(x_true))
    y_noisy  = rng.gamma(shape=k, scale=mu_true / k)
    A_noisy  = x_noisy.reshape(-1, 1)
    # Total Gamma GLM (accounts for x-error)
    # lam = k / (2 * sigma_x^2): high lambda = trust x-measurements more
    lam       = k / (2 * sigma_x**2)
    tot_model = TotalGammaGLM(k=k, lam=lam, max_iter=300, max_iter_b=50, alpha_0=0.1)
    tot_model.fit(A_noisy, y_noisy)
    print("True parameters (bias, w_x): [0.200, 0.500]")
    print(f"Total Gamma GLM (bias, w_x): {np.round(tot_model.b, 4)}")
    print(f"R² log-scale Total: {tot_model.score(A_noisy, y_noisy):.4f}")
    x_plot    = np.linspace(0.5, 5.5, 200)
    A_plot    = x_plot.reshape(-1, 1)
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.scatter(x_noisy, y_noisy, color='steelblue', alpha=0.5, s=35, edgecolors='navy', linewidth=0.4, label='Noisy observations')
    ax.plot(x_plot, np.exp(0.5 * x_plot + 0.2), color='black', linestyle='--', linewidth=1.5, label='True mean $\\mu$')
    ax.plot(x_plot, tot_model.predict(A_plot), color='crimson', linewidth=2.5, label='Total Gamma GLM')
    ax.scatter(tot_model.A_hat.flatten(), y_noisy, color='crimson', alpha=0.25, s=20, label='Corrected $\\hat{x}$')
    for x_o, x_c, y_i in zip(x_noisy, tot_model.A_hat.flatten(), y_noisy):
        ax.plot([x_o, x_c], [y_i, y_i], color='crimson', linewidth=0.6, alpha=0.4)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Total Gamma GLM', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.show()

example()