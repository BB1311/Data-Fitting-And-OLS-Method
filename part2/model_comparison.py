"""
ridge_lasso_integrated.py
=========================
Cài đặt Ridge và Lasso Regression.
- Lõi toán học (Core Solvers) giữ nguyên từ Part 1.
- Cấu trúc OOP (Base, Kế thừa, Properties) áp dụng từ Part 2.
"""

import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# CORE SOLVERS (Lõi toán học từ Part 1)
# ══════════════════════════════════════════════════════════════════════

def _ridge_core_solver(X_design: np.ndarray, y: np.ndarray, lam: float) -> dict:
    """Giải phương trình Ridge Regression dạng kín (Closed-form)."""
    if lam < 0:
        raise ValueError(f"Lỗi: lambda phải >= 0, nhưng nhận được lambda={lam}.")

    n, k = X_design.shape

    # Ma trận phạt: không phạt intercept (cột 0)
    penalty = lam * np.eye(k)
    penalty[0, 0] = 0.0

    XtX = X_design.T @ X_design
    Xty = X_design.T @ y
    A = XtX + penalty

    try:
        beta_hat = np.linalg.solve(A, Xty)
    except np.linalg.LinAlgError:
        # Fallback sang pseudo-inverse nếu ma trận suy biến
        beta_hat = np.linalg.pinv(A) @ Xty

    return {"beta_hat": beta_hat}


def _soft_threshold(z: float, gamma: float) -> float:
    """Toán tử ngưỡng mềm (dùng cho vòng lặp scalar)."""
    return float(np.sign(z) * max(abs(z) - gamma, 0.0))


def _lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    tol_f: float | None = None,
) -> np.ndarray:
    """Coordinate Descent cho bài toán Lasso."""
    n, k   = X.shape
    beta   = np.zeros(k)
    residuals = y.copy()          # residuals = y - X @ beta, ban đầu beta = 0
    x_sq   = np.sum(X ** 2, axis=0)   # ||x_j||^2

    # Hàm mục tiêu ban đầu (nếu theo dõi tol_f)
    if tol_f is not None:
        r     = y - X @ beta
        f_old = (0.5 / n) * float(r @ r) + lam * float(np.sum(np.abs(beta[1:])))
        
    for _ in range(max_iter):
        beta_old = beta.copy()

        for j in range(k):
            if x_sq[j] == 0.0:
                continue

            # Rút thành phần của j ra khỏi residuals
            residuals += X[:, j] * beta[j]
            rho_j = float(X[:, j] @ residuals)

            if j == 0:
                # Intercept: không bị phạt
                beta[j] = rho_j / x_sq[j]
            else:
                # Feature: phạt ngưỡng mềm
                beta[j] = _soft_threshold(rho_j, n * lam) / x_sq[j]
                
            # Đưa đóng góp mới của biến j vào residuals
            residuals -= X[:, j] * beta[j]

        # Kiểm tra hội tụ theo thay đổi của β
        if np.max(np.abs(beta - beta_old)) < tol:
            break

        # Kiểm tra hội tụ theo hàm mục tiêu
        if tol_f is not None:
            r     = y - X @ beta
            f_new = (0.5 / n) * float(r @ r) + lam * float(np.sum(np.abs(beta[1:])))
            if np.abs(f_new - f_old) / max(1.0, np.abs(f_old)) < tol_f:
                break
            f_old = f_new

    return beta


# ══════════════════════════════════════════════════════════════════════
# OOP ARCHITECTURE (Cấu trúc Class từ Part 2)
# ══════════════════════════════════════════════════════════════════════

class _BaseRegressor:
    """Class nền tảng chứa logic dùng chung."""
    
    def __init__(self, lam: float, fit_intercept: bool):
        if lam < 0:
            raise ValueError(f"lambda phải >= 0, nhận được {lam}.")
        self.lam           = lam
        self.fit_intercept = fit_intercept
        self._beta_hat      : np.ndarray | None = None
        self._fitted_values : np.ndarray | None = None
        self._residuals     : np.ndarray | None = None

    def _build_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Chuyển X (n,p) → X_design (n, p+1) nếu fit_intercept=True."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def _check_fitted(self):
        if self._beta_hat is None:
            raise RuntimeError("Gọi .fit(X, y) trước khi dùng mô hình.")

    def _store_results(self, beta_hat: np.ndarray, X_design: np.ndarray, y: np.ndarray):
        """Lưu kết quả chung sau khi solver tìm được β̂."""
        self._beta_hat      = beta_hat
        self._fitted_values = X_design @ beta_hat
        self._residuals     = y - self._fitted_values

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán ŷ = X_design · β̂."""
        self._check_fitted()
        return self._build_design_matrix(X) @ self._beta_hat

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Tính R² trên tập (X, y)."""
        self._check_fitted()
        y      = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        rss    = float(np.sum((y - y_pred) ** 2))
        tss    = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - rss / tss if tss > 0 else 0.0

    @property
    def coef_(self) -> np.ndarray:
        self._check_fitted()
        return self._beta_hat[1:] if self.fit_intercept else self._beta_hat

    @property
    def intercept_(self) -> float:
        self._check_fitted()
        return float(self._beta_hat[0]) if self.fit_intercept else 0.0

    @property
    def fitted_values(self) -> np.ndarray:
        self._check_fitted()
        return self._fitted_values

    @property
    def residuals(self) -> np.ndarray:
        self._check_fitted()
        return self._residuals


# ══════════════════════════════════════════════════════════════════════
# RIDGE REGRESSION
# ══════════════════════════════════════════════════════════════════════
class RidgeRegressor(_BaseRegressor):
    """
    Ridge Regression. Gọi hàm _ridge_core_solver bên dưới vỏ bọc OOP.
    """
    def __init__(self, lam: float = 1.0, fit_intercept: bool = True):
        super().__init__(lam=lam, fit_intercept=fit_intercept)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        X_design = self._build_design_matrix(X)
        y        = np.asarray(y, dtype=float).ravel()

        # Uỷ quyền tính toán cho core solver
        res = _ridge_core_solver(X_design, y, self.lam)
        
        self._store_results(res["beta_hat"], X_design, y)
        return self

    @staticmethod
    def ridge_trace(
        X: np.ndarray, y: np.ndarray, lambdas: np.ndarray = None,
        fit_intercept: bool = True, ax: plt.Axes = None, show: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        
        if lambdas is None: lambdas = np.logspace(-4, 4, 100)
        coef_paths = np.array([RidgeRegressor(lam=l, fit_intercept=fit_intercept).fit(X, y).coef_ for l in lambdas])

        _own_fig = ax is None
        if _own_fig: fig, ax = plt.subplots(figsize=(9, 5))

        for j in range(coef_paths.shape[1]): ax.plot(lambdas, coef_paths[:, j], label=f"β_{j+1}")
        ax.set_xscale("log")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set(xlabel="λ (log scale)", ylabel="Hệ số hồi quy β̂", title="Ridge Trace")
        ax.legend(loc="upper right", fontsize=8, ncol=max(1, coef_paths.shape[1] // 8))
        ax.grid(True, alpha=0.3)

        if _own_fig and show: plt.tight_layout(); plt.show()
        return lambdas, coef_paths

    def __repr__(self) -> str:
        status = "fitted" if self._beta_hat is not None else "not fitted"
        return f"RidgeRegressor(lam={self.lam}, fit_intercept={self.fit_intercept}, status={status})"


# ══════════════════════════════════════════════════════════════════════
# LASSO REGRESSION
# ══════════════════════════════════════════════════════════════════════
class LassoRegressor(_BaseRegressor):
    """
    Lasso Regression. Gọi hàm _lasso_coordinate_descent bên dưới vỏ bọc OOP.
    """
    def __init__(
        self, lam: float = 1.0, fit_intercept: bool = True,
        max_iter: int = 1000, tol: float = 1e-6, tol_f: float | None = None,
    ):
        super().__init__(lam=lam, fit_intercept=fit_intercept)
        self.max_iter = max_iter
        self.tol      = tol
        self.tol_f    = tol_f

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegressor":
        X_design = self._build_design_matrix(X)
        y        = np.asarray(y, dtype=float).ravel()

        # Uỷ quyền tính toán cho core solver
        beta = _lasso_coordinate_descent(
            X=X_design, y=y, lam=self.lam, 
            max_iter=self.max_iter, tol=self.tol, tol_f=self.tol_f
        )

        self._store_results(beta, X_design, y)
        return self

    @property
    def n_nonzero(self) -> int:
        self._check_fitted()
        return int(np.sum(np.abs(self.coef_) > 1e-10))

    @staticmethod
    def lasso_path(
        X: np.ndarray, y: np.ndarray, lambdas: np.ndarray = None,
        fit_intercept: bool = True, ax: plt.Axes = None, show: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        
        if lambdas is None: lambdas = np.logspace(-4, 0, 50)
        coef_paths = np.array([
            LassoRegressor(lam=l, fit_intercept=fit_intercept, max_iter=5000, tol=1e-8).fit(X, y).coef_ 
            for l in lambdas
        ])

        _own_fig = ax is None
        if _own_fig: fig, ax = plt.subplots(figsize=(9, 5))

        for j in range(coef_paths.shape[1]): ax.plot(lambdas, coef_paths[:, j], label=f"β_{j+1}")
        ax.set_xscale("log")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set(xlabel="λ (log scale)", ylabel="Hệ số hồi quy β̂", title="Lasso Path")
        ax.legend(loc="upper right", fontsize=8, ncol=max(1, coef_paths.shape[1] // 8))
        ax.grid(True, alpha=0.3)

        if _own_fig and show: plt.tight_layout(); plt.show()
        return lambdas, coef_paths

    def __repr__(self) -> str:
        status = "fitted" if self._beta_hat is not None else "not fitted"
        return f"LassoRegressor(lam={self.lam}, fit_intercept={self.fit_intercept}, status={status})"