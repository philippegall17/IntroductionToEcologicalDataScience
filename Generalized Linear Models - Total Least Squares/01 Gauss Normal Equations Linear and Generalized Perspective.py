import numpy as np
import matplotlib.pyplot as plt

class GaussNormalRegression:
    """
    An API for Linear Regression solved via the Gauss Normal Equation and QR Decomposition.
    # -------------------------------------------------------------------------------
    # THE ORDINARY LEAST SQUARES PERSPECTIVE ----------------------------------------
    # -------------------------------------------------------------------------------
    Design matrix : A in R^{m x (n+1)} (first column is all-ones for bias)
    Parameters    : b in R^{n+1}       (b[0] = bias, b[1:] = weights)
    Targets       : y in R^m           (observed data)
    Assumed Model : y = A b            (predicted targets)
    # -------------------------------------------------------------------------------
    # THE GENERALIZED LINEAR REGRESSION PERSPECTIVE (SEE NOTES) ---------------------
    # -------------------------------------------------------------------------------
    Gaussian GLM with Identity Link, solved via the Gauss Normal Equation (QR Decomposition).
    This model assumes:
    1. Distribution : y_i ~ N(mu_i, sigma^2) (The observation IS the distribution, the model function is its mean)
    2. Link Function: mu_i = eta_i           (Identity link)
    3. Predictor    : eta_i = a_i^T b        (Linear combination of features)
    Maximum Likelihood Estimation (MLE) for the Gaussian case is equivalent to 
    minimizing the Residual Sum of Squares (RSS) of J(b) = ||y - Ab||² as the loss function.
    # -------------------------------------------------------------------------------
    # IMPLEMENTATION NOTES ----------------- ----------------------------------------
    # -------------------------------------------------------------------------------
    J(b) = ||y - Ab||² = (y - Ab)ᵀ(y - Ab) = yᵀy - yᵀAb - Abᵀy + (Ab)ᵀAb = yᵀy - 2bᵀAᵀy + bᵀAᵀAb
    Convex quadratic minimization with unique global minimum:
    ---------------------------------------------------------
        d/db (bᵀAᵀy)  = Aᵀy   (gradient of a linear form)
        d/db (bᵀAᵀAb) = 2AᵀAb (gradient of a quadratic form as AᵀA is always symmetric)
         => dJ/db = -2Aᵀy + 2AᵀAb = 0 <=> AᵀAb  = Aᵀy (Normal Equation, linear in b)
    In practice, solve least squares using SVD (inverting AᵀA is numerically unstable):
        1. Eigendecompose:        AᵀA = VΛVᵀ      (Numpy uses other methods to not square the condition number by forming AᵀA)
        2. Singular values:       s_i = sqrt(λ_i) 
        3. Left Singular Vectors: From AV = UΣ, dividing by s_i, u_i = (1/s_i)Av_i
        4. Then the least-squares solution is b = V Σ⁻¹ Uᵀ y, where Σ⁻¹ is the pseudo-inverse (invert non-zero s_i, rest set to zero).
    However, we use a QR decomposition to calculate b = R⁻¹ Qᵀ y, which is more convenient and stable. For each column j of A:
        Take the current column below the diagonal x = A[j:, j]
        Construct a reflection vector v = x + sign(x_1) ||x|| e_1
        Form the Householder matrix: H = I - 2vvᵀ and apply A -> HA
        Accumulate all reflections: Q = Product(H_i)_i=1^k (orthogonal), R = QᵀA (upper triangular)
    This method is linear in the parameters b, whereas you need Gauss Newton Methods for nonlinear-parameter models (e.g. exp(b))
    """
    def __init__(self):
        self.b = None # Our calculated parameters
    # The calculations
    def qr_householder(self, A, accuracy=1e-14):
        A    = A.astype(float)
        m, n = A.shape
        Q    = np.eye(m)
        R    = A.copy()
        for j in range(min(m - 1, n)):
            x     = R[j:, j]
            e1    = np.concatenate(([1.0], np.zeros_like(x[1:])))
            alpha = -np.sign(x[0]) * np.linalg.norm(x)
            v     = x - alpha * e1
            if np.linalg.norm(v) < accuracy:
                continue
            v = v / np.linalg.norm(v)
            R[j:, :] -= 2 * np.outer(v, v @ R[j:, :])
            Q[:, j:] -= 2 * np.outer(Q[:, j:] @ v, v)
        return Q, R
    def fit(self, A: np.ndarray, y: np.ndarray):
        """Solve b = R⁻¹ Qᵀ y via QR decomposition. Output = Predicted parameters b."""
        m      = A.shape[0]
        A_aug  = np.hstack([np.ones((m, 1)), A.reshape(m, -1)])
        n      = A_aug.shape[1]
        Q, R   = self.qr_householder(A_aug)
        self.b = np.linalg.solve(R[:n, :n], (Q.T @ y)[:n])
    # The quality checks
    def predict(self, A):
        """Predict targets using the fitted parameters. Output = Predicted targets y."""
        m = A.shape[0]
        return np.hstack([np.ones((m, 1)), A.reshape(m, -1)]) @ self.b
    def score(self, A, y):
        """
        Calculate R² score of the fitted model:
        The R² score is defined as 1 - (SS_res / SS_tot), where SS_res (residual sum of squares) = Σ(y_i - ŷ_i)²
        and SS_tot (total sum of squares) = Σ(y_i - ȳ)². R² represents the proportion of variance in the dependent 
        variable that is predictable from the independent variables. R² = 1 is a perfect fit, =0 no fit, <0 worse than mean.
        """
        y_hat  = self.predict(A)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot

def example_ols():
    # We set that sample interval is equal in data set and fit
    x = np.linspace(0, 20, 100)
    # True model: y = 1.5·sin(x) + 0.8·exp(-x/3) + noise
    y = 1.5 * np.sin(x) + 0.8 * np.exp(-x / 3) + 7.5 + np.random.normal(0, 0.2, size=x.shape)
    # Features: [sin(x), exp(-x/3)] => expect b ≈ [7.5, 1.5, 0.8]
    # These features are in reality not known, here we explicitly construct them to behave like sin and exp.
    # A can be written very versatile here with as many entries as you like. E.g. only np.sin(x) for R^2 = 0.94, or with exp for R^2 = 0.96 depending on noise.
    A = np.column_stack([np.sin(x), np.exp(-x / 3)])
    # Alternatively: A = np.column_stack([x**1, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10]) 
    model = GaussNormalRegression()
    model.fit(A, y)
    print("Fitted parameters (bias, w1, w2):", model.b)
    print("R² score:", model.score(A, y))
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data', color='blue', alpha=0.6, s=50, edgecolors='navy', linewidth=0.5)
    plt.plot(x, model.predict(A), label='Fitted sin/exp', color='green', linewidth=2.5)
    y_pred = model.predict(A)
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], color='grey', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Gaussian Normal Regression - sin/exp Fit', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.show()

def example_glr():
    rng = np.random.default_rng() # Random Number Generator
    x = np.linspace(0, 20, 100)
    # 1. Define the Mean Function
    mu_true = 7.5 + 1.5 * np.sin(x) + 0.8 * np.exp(-x / 3)
    # 2. Generate Data via Distributional Sampling
    # Instead of adding noise, we draw each y_i from N(mu_i, sigma)
    sigma = 0.2
    y = rng.normal(loc=mu_true, scale=sigma)
    # 3. Features (Basis Functions, our ansatz how the model function might have looked like)
    A = np.column_stack([np.sin(x), np.exp(-x / 3)])
    model = GaussNormalRegression()
    model.fit(A, y)
    print("Fitted parameters (bias, w_sin, w_exp):", np.round(model.b, 4))
    print(f"R² score: {model.score(A, y):.4f}")
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Observations $y \sim \mathcal{N}(\mu, \sigma^2)$', 
                color='royalblue', alpha=0.7, edgecolors='navy')
    plt.plot(x, mu_true, label='True Mean $\mu$', color='black', linestyle='--', linewidth=1.5)
    plt.plot(x, model.predict(A), label='MLE Fit (Identity Link)', alpha=0.4, color='crimson', linewidth=2.5)
    y_pred = model.predict(A)
    plt.vlines(x, y, y_pred, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
    plt.title('Gaussian GLM (Identity Link)', fontsize=13)
    plt.xlabel('Input x')
    plt.ylabel('Observation y')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

example_ols()
example_glr()