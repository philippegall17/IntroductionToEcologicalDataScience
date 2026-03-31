import numpy as np
import matplotlib.pyplot as plt

class TotalLeastSquaresRegression:
    """ 
    Total Least Squares (TLS) implementation using SVD.
    Assumes noise in both the target vector y AND additionally in the input design matrix A.
    ----------------------------------------------------------------------------------------
    Fit a Total Least Squares (TLS) model to the data minimizing the orthogonal distance
    in both the feature matrix A and the target vector y. This is in contrast
    to Ordinary Least Squares (OLS) which only minimizes normal distributed errors in y.
    Extracts the coefficients from the last (smallest) right singular vector,
    which represents the direction of minimum variance (the noise subspace).
    """
    def __init__(self):
        self.b      = None
        self.mu_A   = None
        self.mu_y   = None
        self.std_A  = None
        self.std_y  = None
    def fit(self, A: np.ndarray, y: np.ndarray):
        m, n = A.shape
        y = y.reshape(-1, 1)
        # 1. Center and Scale the data (Crucial for TLS to prevent axis-dominance)
        self.mu_A, self.std_A = np.mean(A, axis=0), np.std(A, axis=0)
        self.mu_y, self.std_y = np.mean(y, axis=0), np.std(y, axis=0)
        A_scaled = (A - self.mu_A) / self.std_A
        y_scaled = (y - self.mu_y) / self.std_y
        # 2. Build the augmented matrix [A | y] in R ^ m x (n + 1)
        Z = np.hstack([A_scaled, y_scaled])
        # 3. Perform Singular Value Decomposition on the augmented matrix
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        V = Vt.T
        # The solution is derived from the last right singular vector
        # representing the direction of least variance (the noise), as along it the data is least spread
        # V = [[V_A], [V_y]] where V_A is n x 1 and V_y is 1 x 1
        v_final = V[:, -1:]
        v_a = v_final[:n, :]
        v_y = v_final[n:, :]
        # Calculate b in scaled space, then transform back to original units
        b_scaled = - v_a / v_y
        self.b = b_scaled * (self.std_y / self.std_A.reshape(-1, 1))
    def predict(self, A):
        # Apply the centering transform and the recovered slope
        return (A - self.mu_A) @ self.b + self.mu_y
    def orthogonal_score(self, A, y):
        """
        Calculates the variance explained based on orthogonal distances.
        """
        y = y.reshape(-1, 1)
        # We perform the score calculation in scaled space to keep it units-agnostic
        A_scaled = (A - self.mu_A) / self.std_A
        y_scaled = (y - self.mu_y) / self.std_y
        # 1. Total Orthogonal Variance (Distance of points to the centroid)
        Z = np.hstack([A_scaled, y_scaled])
        ss_tot_ortho = np.sum(Z**2)
        # 2. Residual Orthogonal Variance (The variance along the weakest principal axis)
        # This is exactly what the SVD finds in the last singular value (s_min)
        _, S, _ = np.linalg.svd(Z)
        ss_res_ortho = S[-1]**2
        return 1 - (ss_res_ortho / ss_tot_ortho)
    
def example_tls():
    """
    Shows the difference between Total Least Squares (TLS) and Ordinary Least Squares (OLS) regression for errors in inputs.
    """
    rng = np.random.default_rng(42)
    n_points = 50
    # 1. True relationship: Quadratic
    # Note: In reality, x_noisy**2 would have dependent noise, but TLS treats columns as independent observers.
    x_true = np.linspace(0, 10, n_points)
    y_true = 2 * x_true**2 + 3 * x_true
    # Add noise to BOTH x and y
    x_noisy = x_true + rng.normal(0, 0.5, size=n_points)
    y_noisy = y_true + rng.normal(0, 5.0, size=n_points)
    # Quadratic Design Matrix
    A_noisy = np.column_stack([x_noisy, x_noisy**2])
    # Fit TLS
    tls_model = TotalLeastSquaresRegression()
    tls_model.fit(A_noisy, y_noisy)
    # Fit OLS for comparison (Using Vandermonde: [1, x, x^2])
    A_ols = np.column_stack([np.ones(n_points), x_noisy, x_noisy**2])
    b_ols = np.linalg.lstsq(A_ols, y_noisy, rcond=None)[0]
    # Scores
    ols_r2 = 1 - np.sum((y_noisy - A_ols @ b_ols)**2) / np.sum((y_noisy - np.mean(y_noisy))**2)
    print(f"OLS R-squared (Vertical):   {ols_r2:.4f}")
    print(f"TLS Score (Orthogonal):     {tls_model.orthogonal_score(A_noisy, y_noisy):.4f}")
    print(f"TLS Coefficients (x, x^2): \n{tls_model.b.flatten()}")
    plt.figure(figsize=(10, 6))
    plt.scatter(x_noisy, y_noisy, label="Measured Data (Noisy x and y)", alpha=0.6, color='gray')
    # Generate smooth plot range
    x_plot = np.linspace(min(x_noisy), max(x_noisy), 100)
    A_plot_tls = np.column_stack([x_plot, x_plot**2])
    plt.plot(x_plot, tls_model.predict(A_plot_tls), color='crimson', label="TLS Quadratic Fit (Orthogonal)", linewidth=2)
    plt.plot(x_plot, b_ols[0] + b_ols[1]*x_plot + b_ols[2]*x_plot**2, color='black', linestyle='--', label="OLS Quadratic Fit (Vertical)")
    plt.title("Total Least Squares vs OLS: Quadratic Sensor Data Recovery")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

example_tls()