"""
advanced_methods.py
===================
  1. Kernel Ridge Regression  — phi tuyến thông qua kernel trick
  2. Bayesian Linear Regression — ước lượng theo xác suất, trả về phân phối posterior
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats

from part1.ridge_lasso import ridge_fit
from part1.cross_validation import kfold_cv
from part2.model_comparison import evaluate_model


# ======================================================================
# 1. KERNEL RIDGE REGRESSION
# ======================================================================

def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    """
    Hàm kernel RBF (Radial Basis Function / Gaussian):

        k(x, x') = exp(−‖x − x'‖² / (2ℓ²))

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
    X2 : np.ndarray, shape (n2, p)
    length_scale : float — bandwidth ℓ > 0

    Returns
    -------
    K : np.ndarray, shape (n1, n2)
    """
    if length_scale <= 0:
        raise ValueError(f"length_scale phải > 0, nhận được {length_scale}")

    # Tính ‖x_i - x_j‖² theo công thức: ‖a-b‖² = ‖a‖² + ‖b‖² - 2aᵀb
    # Hiệu quả hơn vòng lặp kép khi n lớn
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)   # (n1, 1)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)   # (n2, 1)
    cross = X1 @ X2.T                               # (n1, n2)
    dist_sq = sq1 + sq2.T - 2 * cross              # (n1, n2)
    dist_sq = np.maximum(dist_sq, 0.0)             # clip sai số float

    return np.exp(-dist_sq / (2.0 * length_scale ** 2))


def _poly_kernel(X1: np.ndarray, X2: np.ndarray, degree: int = 2, coef0: float = 1.0) -> np.ndarray:
    """
    Hàm kernel Polynomial:

        k(x, x') = (xᵀx' + c)^d

    Parameters
    ----------
    X1, X2  : np.ndarray
    degree  : int — bậc đa thức
    coef0   : float — hằng số c >= 0
    """
    return (X1 @ X2.T + coef0) ** degree


KERNELS = {
    'rbf':  _rbf_kernel,
    'poly': _poly_kernel,
}

def _krr_core_solver(
    X: np.ndarray,
    y: np.ndarray,
    kernel_fn,
    lam: float,
) -> dict:
    """
    Lõi toán học của Kernel Ridge Regression.

    Giải hệ:
        (K + λI) α = y   →   α = (K + λI)⁻¹ y

    Trong đó K[i,j] = kernel_fn(xᵢ, xⱼ) là Gram matrix (n×n).
    Dự đoán sau đó được tính bằng:
        ŷ(x*) = k(x*)ᵀ α

    Parameters
    ----------
    X         : np.ndarray (n, p) — dữ liệu train
    y         : np.ndarray (n,)
    kernel_fn : callable(X1, X2) -> np.ndarray — hàm kernel đã được bind tham số
    lam       : float >= 0 — hệ số regularization

    Returns
    -------
    dict với các key:
        alpha    : np.ndarray (n,) — vector hệ số dual
        X_train  : np.ndarray (n, p) — bản sao dữ liệu train (cần cho predict)
        K_train  : np.ndarray (n, n) — Gram matrix trên tập train
        lam      : float
        n        : int
        p        : int
    """
    n, p = X.shape

    # 1. Tính Gram matrix K (n×n) — ma trận độ tương đồng giữa các điểm train
    K = kernel_fn(X, X)

    # 2. Giải (K + λI) α = y  →  α = (K + λI)⁻¹ y
    A = K + lam * np.eye(n)
    try:
        alpha = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        # Fallback lstsq nếu ma trận gần suy biến (lam quá nhỏ)
        alpha, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    return {
        "alpha":   alpha,
        "X_train": X.copy(),
        "K_train": K,
        "lam":     lam,
        "n":       n,
        "p":       p,
    }

class KernelRidgeRegressor:
    """
    Kernel Ridge Regression (KRR).

    Công thức dự đoán:
        α = (K + λI)⁻¹ y      ← fit: học vector alpha
        ŷ(x*) = k(x*)ᵀ α      ← predict: tổ hợp độ tương đồng

    Khi kernel = 'rbf', mô hình phi tuyến trong không gian Gaussian.
    Khi λ→0 và kernel là dot-product, kết quả tiệm cận Ridge thông thường.

    Parameters
    ----------
    kernel        : 'rbf' | 'poly'
    lam           : float >= 0  — hệ số regularization
    length_scale  : float > 0   — bandwidth ℓ cho RBF kernel
    degree        : int         — bậc cho polynomial kernel
    coef0         : float       — hằng số c cho polynomial kernel

    Attributes (sau fit)
    ----------
    alpha_     : np.ndarray (n,)  — vector hệ số dual
    X_train_   : np.ndarray       — dữ liệu train (cần cho predict)
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        lam: float = 1.0,
        length_scale: float = 1.0,
        degree: int = 2,
        coef0: float = 1.0,
    ):
        if kernel not in KERNELS:
            raise ValueError(f"kernel phải là một trong {list(KERNELS.keys())}")
        if lam < 0:
            raise ValueError(f"lam phải >= 0, nhận được {lam}")

        self.kernel = kernel
        self.lam = lam
        self.length_scale = length_scale
        self.degree = degree
        self.coef0 = coef0

        self.alpha_ = None
        self.X_train_ = None
        self.y_train_ = None
        self._kernel_fn = None   # hàm kernel đã bind tham số
        self._fitted = False

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Dispatcher: tính kernel matrix dựa trên tham số."""
        if self.kernel == 'rbf':
            return _rbf_kernel(X1, X2, length_scale=self.length_scale)
        elif self.kernel == 'poly':
            return _poly_kernel(X1, X2, degree=self.degree, coef0=self.coef0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KernelRidgeRegressor':
        """
        Học vector alpha = (K + λI)⁻¹ y trên tập train.

        Parameters
        ----------
        X : (n, p) — ma trận đặc trưng (chưa có cột intercept)
        y : (n,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Bind tham số kernel rồi truyền vào solver
        self._kernel_fn = self._compute_kernel
        res = _krr_core_solver(X, y, self._kernel_fn, self.lam)

        # Lưu kết quả từ solver vào attributes của class
        self.alpha_   = res["alpha"]
        self.X_train_ = res["X_train"]
        self.y_train_ = y.copy() 
        self._fitted = True
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Dự đoán ŷ = k(X_new, X_train) @ alpha.

        Parameters
        ----------
        X_new : (m, p)

        Returns
        -------
        y_pred : (m,)
        """
        if not self._fitted:
            raise RuntimeError("Gọi fit() trước khi predict().")

        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        # K_new: (m, n)  →  mỗi hàng là độ tương đồng của 1 điểm test với toàn bộ train
        K_new = self._compute_kernel(X_new, self.X_train_)
        return K_new @ self.alpha_

    def evaluate(self, X: np.ndarray, y: np.ndarray, inverse_transform: bool = False) -> dict:
        """Tính MAE, RMSE, R² trên tập dữ liệu cho trước."""
        y_pred = self.predict(X)
        return evaluate_model(y, y_pred, inverse_transform=inverse_transform)

    def summary(self):
        """In tóm tắt mô hình."""
        if not self._fitted:
            print("Mô hình chưa được fit.")
            return
            
        print("=== KERNEL RIDGE REGRESSION ===")
        print(f"Kernel: {self.kernel}")
        print(f"Lambda: {self.lam}")
        if self.kernel == 'rbf':
            print(f"Length scale: {self.length_scale}")
        elif self.kernel == 'poly':
            print(f"Degree: {self.degree}, Coef0: {self.coef0}")
        print("-" * 30)
        print(f"Số điểm dữ liệu train: {self.X_train_.shape[0]}")
        print(f"Số đặc trưng: {self.X_train_.shape[1]}")
        
        # --- Tự động tính metrics trên tập train ---
        metrics = self.evaluate(self.X_train_, self.y_train_)
        print(f"  MAE  (train): {metrics['mae']:.6f}")
        print(f"  RMSE (train): {metrics['rmse']:.6f}")
        print(f"  R²   (train): {metrics['r2']:.6f}")

    @classmethod
    def cv_search(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        lam_grid: list = None,
        ls_grid: list = None,
        k: int = 5,
        kernel: str = 'rbf',
        degree: int = 2,           
        coef0: float = 1.0,       
        random_state: int = 42,
        verbose: bool = True,
    ) -> dict:
        """
        Grid search λ và length_scale bằng k-fold CV.

        Tận dụng infrastructure của kfold_cv nhưng dùng KRR thay OLS/Ridge/Lasso.
        (kfold_cv hỗ trợ model='ols'/'ridge'/'lasso')

        Returns
        -------
        dict: best_lam, best_length_scale, best_cv_mse, results_grid
        """
        if lam_grid is None:
            lam_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        if ls_grid is None:
            ls_grid = [0.1, 0.5, 1.0, 2.0, 5.0] if kernel == 'rbf' else [None]

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        # Tạo fold splits (tái dụng logic của kfold_cv)
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n)
        fold_sizes = np.full(k, n // k)
        fold_sizes[:n % k] += 1
        boundaries = np.concatenate([[0], np.cumsum(fold_sizes)])

        results = []
        best_mse = float('inf')
        best_lam = lam_grid[0]
        best_ls = ls_grid[0] if ls_grid[0] is not None else 1.0

        for lam in lam_grid:
            for ls in ls_grid:
                fold_mses = []
                for i in range(k):
                    val_idx = indices[boundaries[i]: boundaries[i + 1]]
                    train_idx = np.concatenate([
                        indices[:boundaries[i]],
                        indices[boundaries[i + 1]:]
                    ])
                    
                    model_kw = {'kernel': kernel, 'lam': lam}
                    if kernel == 'rbf' and ls is not None:
                        model_kw['length_scale'] = ls
                    elif kernel == 'poly':
                        model_kw['degree'] = degree
                        model_kw['coef0'] = coef0
                        
                    m = cls(**model_kw).fit(X[train_idx], y[train_idx])
                    y_pred = m.predict(X[val_idx])
                    fold_mses.append(float(np.mean((y[val_idx] - y_pred) ** 2)))

                cv_mse = float(np.mean(fold_mses))
                entry = {'lam': lam, 'length_scale': ls, 'cv_mse': cv_mse}
                results.append(entry)

                if cv_mse < best_mse:
                    best_mse = cv_mse
                    best_lam = lam
                    best_ls = ls if ls is not None else 1.0

        if verbose:
            print(f"[KRR CV] Best: lam={best_lam}, length_scale={best_ls}, CV MSE={best_mse:.6f}")

        return {
            'best_lam': best_lam,
            'best_length_scale': best_ls,
            'best_cv_mse': best_mse,
            'results_grid': results,
        }

def compare_krr_vs_ols(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1.0,
    length_scale: float = 1.0,
    k: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    So sánh Kernel Ridge (RBF) và OLS thông thường trên cùng dữ liệu.
    Tận dụng hàm kfold_cv từ Part 1 cho OLS.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    # --- OLS (dùng kfold_cv từ Part 1) ---
    ols_cv = kfold_cv(X, y, k=k, model='ols', random_state=random_state)

    # --- Kernel Ridge với manual k-fold ---
    n = len(y)
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n)
    fold_sizes = np.full(k, n // k)
    fold_sizes[:n % k] += 1
    boundaries = np.concatenate([[0], np.cumsum(fold_sizes)])

    krr_fold_mses = []
    for i in range(k):
        val_idx = indices[boundaries[i]: boundaries[i + 1]]
        train_idx = np.concatenate([indices[:boundaries[i]], indices[boundaries[i + 1]:]])
        m = KernelRidgeRegressor(kernel='rbf', lam=lam, length_scale=length_scale)
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[val_idx])
        krr_fold_mses.append(float(np.mean((y[val_idx] - y_pred) ** 2)))

    krr_cv_mse = float(np.mean(krr_fold_mses))

    if verbose:
        print("\n         SO SÁNH: Kernel Ridge vs OLS")
        print(f"  {'Method':<25} {'CV MSE':>12}")
        print("-" * 40)
        print(f"  {'OLS':<25} {ols_cv['cv_mse']:>12.6f}")
        print(f"  {'Kernel Ridge (RBF)':<25} {krr_cv_mse:>12.6f}")
        winner = "Kernel Ridge" if krr_cv_mse < ols_cv["cv_mse"] else "OLS"
        print(f"\n  Mô hình tốt hơn: {winner}")

    return {
        "ols_cv_mse": ols_cv["cv_mse"],
        "krr_cv_mse": krr_cv_mse,
        "winner": "krr" if krr_cv_mse < ols_cv["cv_mse"] else "ols",
    }


# ======================================================================
# 2. BAYESIAN LINEAR REGRESSION
# ======================================================================

def _bayesian_core_solver(
    X_design: np.ndarray,
    y: np.ndarray,
    alpha: float,
    sigma2: float,
    fit_intercept: bool = True,
) -> dict:
    """
    Lõi toán học của Bayesian Linear Regression.

    Prior:
        β ~ N(m₀, S₀)  với  m₀ = 0,  S₀ = (σ²/α)·I
        (intercept không bị penalize nếu fit_intercept=True)

    Posterior:
        Sₙ = (S₀⁻¹ + (1/σ²) XᵀX)⁻¹
        mₙ = Sₙ (S₀⁻¹ m₀ + (1/σ²) Xᵀy)

    Parameters
    ----------
    X_design : (n, k) — ma trận design đã có cột 1 (nếu fit_intercept=True).
    y        : (n,)
    alpha    : precision của prior.
    sigma2   : phương sai nhiễu σ².
    fit_intercept : bool — có penalize intercept hay không.

    Returns
    -------
    dict:
        m_n     : posterior mean (k,)
        S_n     : posterior covariance (k, k)
        sigma2  : float — giữ nguyên σ² đầu vào
        k       : int — số tham số
    """
    k = X_design.shape[1]

    # --- Prior: β ~ N(m₀, S₀)  với  S₀ = (σ²/α)·I ---
    # Tương đương: precision matrix S₀⁻¹ = (α/σ²)·I
    # Khi fit_intercept=True, ta KHÔNG penalize intercept
    prior_precision = (alpha / sigma2) * np.eye(k)
    if fit_intercept:
        prior_precision[0, 0] = 0.0   # không penalize intercept

    m_0 = np.zeros(k)

    # --- Posterior: Sₙ = (S₀⁻¹ + (1/σ²) XᵀX)⁻¹ ---
    # Posterior precision = prior_precision + (1/σ²) XᵀX
    XtX = X_design.T @ X_design
    posterior_precision = prior_precision + XtX / sigma2

    try:
        S_n = np.linalg.inv(posterior_precision)
    except np.linalg.LinAlgError:
        S_n = np.linalg.pinv(posterior_precision)

    # Posterior mean:  mₙ = Sₙ (S₀⁻¹ m₀ + (1/σ²) Xᵀy)
    Xty = X_design.T @ y
    rhs = prior_precision @ m_0 + Xty / sigma2
    m_n = S_n @ rhs

    return {
        "m_n":    m_n,
        "S_n":    S_n,
        "sigma2": sigma2,
        "k":      k,
    }
    
class BayesianLinearRegressor:
    """
    Bayesian Linear Regression với Gaussian conjugate prior.

    Mô hình:
        Prior   :  β ~ N(m₀, S₀)     — niềm tin trước khi thấy dữ liệu
        Likelihood: y | X, β ~ N(Xβ, σ²I)
        Posterior:  β | X, y ~ N(mₙ, Sₙ)   ← cập nhật sau khi có dữ liệu

    Công thức posterior (conjugate Gaussian):
        Sₙ = (S₀⁻¹ + (1/σ²) XᵀX)⁻¹
        mₙ = Sₙ (S₀⁻¹ m₀ + (1/σ²) Xᵀy)

    Kết nối với Ridge:
        Khi prior là N(0, (σ²/λ)I), posterior mean = β̂_Ridge = (XᵀX + λI)⁻¹ Xᵀy

    Parameters
    ----------
    alpha    : float — precision của prior (alpha = 1/τ², τ² là phương sai prior)
               Tương đương λ/σ² trong Ridge. Lớn = prior mạnh → shrinkage mạnh.
    sigma2   : float — phương sai nhiễu σ². Nếu None, tự ước lượng từ dữ liệu.
    fit_intercept : bool — tự động thêm cột 1 vào X.

    Attributes (sau fit)
    ----------
    m_n_     : np.ndarray (k,) — posterior mean (điểm ước lượng tốt nhất)
    S_n_     : np.ndarray (k,k) — posterior covariance (độ không chắc chắn)
    sigma2_  : float — phương sai nhiễu đã học
    """

    def __init__(
        self,
        alpha: float = 1.0,
        sigma2: float = None,
        fit_intercept: bool = True,
    ):
        if alpha <= 0:
            raise ValueError(f"alpha (precision) phải > 0, nhận được {alpha}")

        self.alpha = alpha
        self.sigma2 = sigma2
        self.fit_intercept = fit_intercept

        self.m_n_ = None
        self.S_n_ = None
        self.sigma2_ = None
        self.X_train_ = None
        self.y_train_ = None
        self._X_design = None
        self._fitted = False

    def _make_design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegressor':
        """
        Tính posterior mean mₙ và covariance Sₙ.

        Nếu sigma2=None khi khởi tạo, ước lượng σ² từ OLS residuals trước,
        sau đó cập nhật posterior.
        """
        X_design = self._make_design(X)
        y = np.asarray(y, dtype=float).ravel()
        n, k = X_design.shape

        # --- Ước lượng σ² nếu chưa biết ---
        if self.sigma2 is None:
            # Dùng Ridge (λ nhỏ) từ Part 1 để tránh vấn đề rank khi n < k
            res = ridge_fit(X, y, lam=1e-8)
            residuals = y - X_design @ res["beta_hat"]
            df = max(n - k, 1)
            self.sigma2_ = float(np.sum(residuals ** 2) / df)
        else:
            self.sigma2_ = float(self.sigma2)

        self.X_train_ = X.copy()          
        self.y_train_ = y.copy()         
        
        # Gọi lõi toán học
        res = _bayesian_core_solver(
            X_design, y,
            alpha=self.alpha,
            sigma2=self.sigma2_,
            fit_intercept=self.fit_intercept,
        )

        self.m_n_ = res["m_n"]
        self.S_n_ = res["S_n"]
        self._X_design = X_design
        self._fitted = True
        return self

    def predict(
        self,
        X_new: np.ndarray,
        return_std: bool = True,
    ):
        """
        Dự đoán phân phối posterior predictive tại X_new.

        Predictive distribution:
            p(y* | x*, X, y) = N(mₙᵀ x*, σ²_pred)
            σ²_pred = σ² + x*ᵀ Sₙ x*

        Parameters
        ----------
        X_new      : (m, p)
        return_std : bool — nếu True trả về (y_mean, y_std) thay vì chỉ y_mean

        Returns
        -------
        y_mean : (m,)
        y_std  : (m,) — nếu return_std=True
        """
        if not self._fitted:
            raise RuntimeError("Gọi fit() trước khi predict().")

        X_design = self._make_design(X_new)
        y_mean = X_design @ self.m_n_

        if not return_std:
            return y_mean

        # Predictive variance: σ²_pred[i] = σ² + x*_i ᵀ Sₙ x*_i
        # Dùng einsum để hiệu quả
        pred_var = self.sigma2_ + np.einsum('ij,jk,ik->i', X_design, self.S_n_, X_design)
        pred_var = np.maximum(pred_var, 0.0)   # clip sai số float
        y_std = np.sqrt(pred_var)

        return y_mean, y_std

    def credible_interval(
        self,
        X_new: np.ndarray,
        confidence: float = 0.95,
    ) -> tuple:
        """
        Tính credible interval (khoảng tin cậy Bayesian) cho dự đoán.

        Parameters
        ----------
        X_new      : (m, p)
        confidence : float — mức tin cậy (mặc định 95%)

        Returns
        -------
        lower : (m,)
        upper : (m,)
        """
        alpha_level = 1.0 - confidence
        z = stats.norm.ppf(1.0 - alpha_level / 2.0)
        y_mean, y_std = self.predict(X_new, return_std=True)
        return y_mean - z * y_std, y_mean + z * y_std

    def evaluate(self, X: np.ndarray, y: np.ndarray, inverse_transform: bool = False) -> dict:
        """Tính MAE, RMSE, R² trên tập dữ liệu cho trước."""
        y_pred = self.predict(X, return_std=False)
        return evaluate_model(y, y_pred, inverse_transform=inverse_transform)

    def summary(self, feature_names: list = None):
        """In tóm tắt mô hình."""
        if not self._fitted:
            print("Mô hình chưa được fit.")
            return

        print("=== BAYESIAN LINEAR REGRESSION ===")
        print(f"Alpha (prior precision): {self.alpha}")
        print(f"Sigma2 (noise variance): {self.sigma2_:.6f}")
        print(f"Fit intercept: {self.fit_intercept}")
        
        # --- Tự động tính metrics trên tập train ---
        metrics = self.evaluate(self.X_train_, self.y_train_)
        print(f"  MAE  (train): {metrics['mae']:.6f}")
        print(f"  RMSE (train): {metrics['rmse']:.6f}")
        print(f"  R²   (train): {metrics['r2']:.6f}")
    
        print("-" * 30)
        self.coef_summary(feature_names=feature_names, verbose=True)

    def coef_summary(self, feature_names: list = None, verbose: bool = True) -> dict:
        """
        Tóm tắt posterior của từng hệ số.
        
        Trả về posterior mean và posterior standard deviation cho mỗi β_j.
        posterior std = sqrt(diag(Sₙ)) — độ không chắc chắn về hệ số.
        """
        if not self._fitted:
            raise RuntimeError("Gọi fit() trước.")

        k = len(self.m_n_)
        coef_std = np.sqrt(np.maximum(np.diag(self.S_n_), 0.0))

        if feature_names is None:
            if self.fit_intercept:
                feature_names = ["Intercept"] + [f"X_{i}" for i in range(1, k)]
            else:
                feature_names = [f"X_{i}" for i in range(1, k + 1)]

        if verbose:
            print("\n        BAYESIAN COEFFICIENT SUMMARY")
            print(f"  {'Feature':>15} {'Post. Mean':>14} {'Post. Std':>12} {'95% CI Lower':>14} {'95% CI Upper':>14}")
            print("-" * 75)
            for name, mean, std in zip(feature_names, self.m_n_, coef_std):
                lo = mean - 1.96 * std
                hi = mean + 1.96 * std
                print(f"  {name:>15} {mean:>14.6f} {std:>12.6f} {lo:>14.6f} {hi:>14.6f}")
            print("-" * 75)
            print(f"  sigma² = {self.sigma2_:.6f}  |  alpha (prior precision) = {self.alpha}")

        return {
            "feature_names": feature_names,
            "posterior_mean": self.m_n_,
            "posterior_std": coef_std,
        }

    def ridge_equivalence_check(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = True
    ) -> dict:
        """
        Kiểm chứng: posterior mean mₙ luôn bằng β̂_Ridge(λ=α) bất kể σ² bằng bao nhiêu.

        Chứng minh (σ² triệt tiêu hoàn toàn):
        ──────────────────────────────────────
        Cho prior β ~ N(0, (σ²/α)·I), tức S₀⁻¹ = (α/σ²)·I và m₀ = 0.

        Posterior precision:
            Sₙ⁻¹ = S₀⁻¹ + (1/σ²)·XᵀX
                  = (α/σ²)·I + (1/σ²)·XᵀX
                  = (1/σ²)·(αI + XᵀX)          ← factor ra 1/σ²

        Posterior covariance:
            Sₙ = σ²·(αI + XᵀX)⁻¹               ← σ² xuất hiện

        RHS của posterior mean (m₀ = 0):
            S₀⁻¹·m₀ + (1/σ²)·Xᵀy = (1/σ²)·Xᵀy

        Posterior mean:
            mₙ = Sₙ · (1/σ²)·Xᵀy
               = [σ²·(αI + XᵀX)⁻¹] · (1/σ²)·Xᵀy
               = (αI + XᵀX)⁻¹·Xᵀy              ← σ² triệt tiêu HOÀN TOÀN

        Đây chính xác là β̂_Ridge với λ = α:
            β̂_Ridge = (XᵀX + αI)⁻¹·Xᵀy

        ⇒ mₙ = β̂_Ridge(λ=α) với MỌI GIÁ TRỊ σ² > 0.

        Tận dụng hàm ridge_fit từ Part 1 để kiểm chứng số học.
        """
        if not self._fitted:
            raise RuntimeError("Gọi fit() trước.")

        # λ = α vì σ² triệt tiêu hoàn toàn trong công thức posterior mean
        # Điều này đúng với MỌI giá trị σ², không phụ thuộc vào cách ước lượng σ²
        lam = self.alpha
        res_ridge = ridge_fit(X, y, lam=lam)
        beta_ridge = res_ridge["beta_hat"]
        beta_bayes = self.m_n_

        max_diff = float(np.max(np.abs(beta_bayes - beta_ridge)))
        passed = max_diff < 1e-6

        if verbose:
            print("\n         KIỂM CHỨNG: Bayesian ↔ Ridge")
            print(f"  {'Coef':>10} {'Bayesian (mₙ)':>16} {'Ridge (β̂)':>16} {'|delta|':>12}")
            labels = ["intercept"] + [f"β_{i}" for i in range(1, len(beta_bayes))]
            for lbl, b, r in zip(labels, beta_bayes, beta_ridge):
                print(f"  {lbl:>10}  {b:>15.6f}  {r:>15.6f}  {abs(b-r):>11.2e}")
            print(f"  Max diff = {max_diff:.2e}  → {'PASS' if passed else 'FAIL'}")

        return {
            "beta_bayes": beta_bayes,
            "beta_ridge": beta_ridge,
            "max_diff": max_diff,
            "passed": passed,
        }


def compare_bayesian_vs_ols(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    k_cv: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    So sánh Bayesian LR và OLS về R², RMSE trên tập train và CV MSE.
    Tận dụng kfold_cv từ Part 1 cho OLS.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    # OLS CV (Part 1)
    ols_cv = kfold_cv(X, y, k=k_cv, model='ols', random_state=random_state)

    # Bayesian manual k-fold CV (không có hàm sẵn, tự viết)
    n = len(y)
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n)
    fold_sizes = np.full(k_cv, n // k_cv)
    fold_sizes[:n % k_cv] += 1
    boundaries = np.concatenate([[0], np.cumsum(fold_sizes)])

    bayes_fold_mses = []
    for i in range(k_cv):
        val_idx = indices[boundaries[i]: boundaries[i + 1]]
        train_idx = np.concatenate([indices[:boundaries[i]], indices[boundaries[i + 1]:]])
        m = BayesianLinearRegressor(alpha=alpha).fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[val_idx], return_std=False)
        bayes_fold_mses.append(float(np.mean((y[val_idx] - y_pred) ** 2)))

    bayes_cv_mse = float(np.mean(bayes_fold_mses))

    if verbose:
        print("\n         SO SÁNH: Bayesian LR vs OLS")
        print(f"  {'Method':<25} {'CV MSE':>12} {'CV RMSE':>12}")
        print("-" * 52)
        print(f"  {'OLS':<25} {ols_cv['cv_mse']:>12.6f} {ols_cv['cv_rmse']:>12.6f}")
        print(f"  {'Bayesian LR':<25} {bayes_cv_mse:>12.6f} {float(np.sqrt(bayes_cv_mse)):>12.6f}")
        winner = "Bayesian" if bayes_cv_mse < ols_cv["cv_mse"] else "OLS"
        print(f"\n  Mô hình tốt hơn: {winner}")

    return {
        "ols_cv_mse": ols_cv["cv_mse"],
        "bayes_cv_mse": bayes_cv_mse,
        "winner": "bayesian" if bayes_cv_mse < ols_cv["cv_mse"] else "ols",
    }