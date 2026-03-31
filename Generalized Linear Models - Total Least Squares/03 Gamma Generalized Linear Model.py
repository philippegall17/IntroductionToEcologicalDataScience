import numpy as np
import matplotlib.pyplot as plt

class GammaGLMRegression:
    """
    Gamma GLM with log link, solved via gradient ascent on the negative log-likelihood.
    Design matrix : A in R^{m x (n+1)} (first column is all-ones for bias)
    Parameters    : b in R^{n+1}       (b[0] = bias, b[1:] = weights)
    Targets       : y in R^m           (gamma-distributed, hence positive observations, y_i > 0)
    Assumed Model : mu_i = exp(A_i @ b)  (log link guarantees mu_i > 0 for any b)
    Distribution  : Gamma(shape=k, mean=mu_i), known shape k > 0
    Negative Log-likelihood (dropping constants in b):
    --------------------------------------------------
        log L(b) = const - k * sum_i [ a_i^T b + y_i * exp(-a_i^T b) ]
        grad_b log L(b) = k * A^T (y / mu - 1),  mu_i = exp(a_i^T b)
    Armijo backtracking line search:
    --------------------------------------------------
        log L(b + alpha * d) >= log L(b) + c * alpha * grad^T d
        where d = grad_b log L(b) is the ascent direction and c in (0,1).
        Start from alpha = alpha_0 and shrink by factor rho until condition holds.
    This method handles the non-quadratic log-likelihood arising from non-Gaussian
    distributions, where no closed-form normal equation exists, such as this Gamma case. 
    """
    def __init__(self, k=1.0, max_iter=2000, tol=1e-8, alpha_0=1.0, rho=0.5, c=1e-4):
        self.k        = k        # Gamma shape parameter (known)
        self.max_iter = max_iter
        self.tol      = tol
        self.alpha_0  = alpha_0  # Initial step size for Armijo
        self.rho      = rho      # Step shrinkage factor
        self.c        = c        # Sufficient decrease constant
        self.b        = None
        self.history  = []       # Log-likelihood per iteration
    def mu(self, A_aug):
        """Mean prediction via log link: mu = exp(A_aug @ b)."""
        return np.exp(A_aug @ self.b)
    def loglikelihood(self, A_aug, y):
        """Log-likelihood dropping constants in b."""
        eta = A_aug @ self.b
        return -self.k * np.sum(eta + y * np.exp(-eta))
    def grad_loglikelihood(self, A_aug, y):
        """Gradient of log-likelihood w.r.t. b."""
        mu = self.mu(A_aug)
        return self.k * (A_aug.T @ (y / mu - 1.0))
    def armijo(self, A_aug, y, grad):
        """
        Backtracking line search with Armijo condition.
        Searches for alpha such that log L(b + alpha * grad) >= log L(b) + c * alpha * ||grad||^2
        """
        alpha   = self.alpha_0
        ll_curr = self.loglikelihood(A_aug, y)
        slope   = self.c * np.dot(grad, grad) # c * ||grad||^2 >= 0
        for _ in range(100):
            b_new = self.b + alpha * grad
            # Temporarily evaluate log-likelihood at new b
            # (overflow in exp is benign: trial step is rejected and alpha shrinks)
            eta_new = A_aug @ b_new
            with np.errstate(over='ignore', invalid='ignore'):
                ll_new = -self.k * np.sum(eta_new + y * np.exp(-eta_new))
            ll_new = -np.inf if not np.isfinite(ll_new) else ll_new
            if ll_new >= ll_curr + alpha * slope:
                break
            alpha *= self.rho
        return alpha
    def fit(self, A: np.ndarray, y: np.ndarray):
        """
        Gradient ascent on log-likelihood with Armijo step size.
        Output = Fitted parameters b (bias + weights).
        """
        m      = A.shape[0]
        A_aug  = np.hstack([np.ones((m, 1)), A.reshape(m, -1)])
        self.b = np.zeros(A_aug.shape[1])
        self.history = []
        for i in range(self.max_iter):
            self.history.append(self.loglikelihood(A_aug, y))
            grad = self.grad_loglikelihood(A_aug, y)
            if np.linalg.norm(grad) < self.tol:
                print(f"Converged at iteration {i} (||grad|| < {self.tol})")
                break
            alpha  = self.armijo(A_aug, y, grad)
            self.b = self.b + alpha * grad
    def predict(self, A: np.ndarray):
        """Predict mean targets using the fitted parameters. Output = mu = exp(A_aug @ b)."""
        m     = A.shape[0]
        A_aug = np.hstack([np.ones((m, 1)), A.reshape(m, -1)])
        return np.exp(A_aug @ self.b)
    def score(self, A: np.ndarray, y: np.ndarray):
        """
        R² on the log scale (natural for multiplicative models):
        A value of 1 is a perfect fit, 0 means no better than the mean.
        """
        log_y  = np.log(y)
        log_mu = np.log(self.predict(A))
        ss_res = np.sum((log_y - log_mu) ** 2)
        ss_tot = np.sum((log_y - log_y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot

def example():
    rng = np.random.default_rng() # random number generator
    x   = np.linspace(0.0, 5.0, 200)
    # True model: log(mu) = - 0.15*x^2 + 0.8*x + 1.0 => b = [1.0, 0.8, -0.15]
    # Gamma shape k=20: low noise so parameters are cleanly recoverable.
    k       = 20
    log_mu  = 0.8 * x - 0.15 * x**2 + 1.0
    mu_true = np.exp(log_mu)
    y       = rng.gamma(shape=k, scale=mu_true / k) # E[y] = mu, Var[y] = mu^2/k
    A       = np.column_stack([x, x**2])            # Features: [x, x^2] => expect b ≈ [1.0, 0.8, -0.15]
    model   = GammaGLMRegression(k=k, max_iter=3000, alpha_0=0.1)
    model.fit(A, y)
    print("Fitted parameters (bias, w_x, w_x2):", np.round(model.b, 4))
    print("True  parameters (bias, w_x, w_x2):  [1.0, 0.8, -0.15]")
    print(f"R² (log scale): {model.score(A, y):.4f}")
    mu_hat = model.predict(A)
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.scatter(x, y, label='Gamma samples', color='steelblue', alpha=0.55,
               s=40, edgecolors='navy', linewidth=0.4)
    ax.plot(x, mu_true, label='True mean μ', color='black',
            linewidth=2, linestyle='--')
    ax.plot(x, mu_hat,  label='Fitted mean μ̂', color='crimson', linewidth=2.5)
    for xi, yi, yi_hat in zip(x, y, mu_hat):
        ax.plot([xi, xi], [yi, yi_hat], color='grey', linestyle='-', linewidth=0.6, alpha=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Gamma GLM (log link) — Fit vs Data', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

example()