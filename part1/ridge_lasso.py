"""
ridge_lasso.py
==============
Cài đặt Ridge Regression (dạng kín) và Lasso Regression (coordinate descent).

Cấu trúc file:
  - _ridge_core_solver        : Giải phương trình Ridge dạng kín (lõi toán học).
  - ridge_fit                 : Hàm hỗ trợ nhanh cho Ridge.
  - RidgeRegressor            : Class OOP bao gồm fit / predict / ridge_trace.
  - _soft_threshold           : Toán tử ngưỡng mềm dùng trong Lasso.
  - _lasso_coordinate_descent : Coordinate Descent cho Lasso.
  - lasso_fit                 : Hàm hỗ trợ nhanh cho Lasso.
  - LassoRegressor            : Class OOP bao gồm fit / predict / lasso_path.
  - verify_ridge_sklearn      : Kiểm chứng Ridge với sklearn.
  - verify_lasso_sklearn      : Kiểm chứng Lasso với sklearn.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# RIDGE REGRESSION (Dạng kín / Closed-form)
# ============================================================

def _ridge_core_solver(X_design: np.ndarray, y: np.ndarray, lam: float) -> dict:
    """
    Giải phương trình Ridge Regression dạng kín:

        beta_hat_ridge = (X^T X + lambda * I_k)^{-1} X^T y

    Lưu ý: cột intercept (cột 0 của X_design) KHÔNG bị phạt —
    phần tử [0,0] của ma trận phạt được đặt về 0 theo quy ước chuẩn.

    Parameters
    ----------
    X_design : np.ndarray, shape (n, k)
        Ma trận thiết kế đã bao gồm cột 1 (intercept) ở vị trí cột đầu.
    y : np.ndarray, shape (n,)
        Vector biến mục tiêu.
    lam : float
        Hệ số phạt lambda >= 0.

    Returns
    -------
    dict với các key: beta_hat, y_hat, residuals, rss, n, k, lam.
    """
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
        raise ValueError(
            "Lỗi: Ma trận (X^T X + lambda*I) không khả nghịch. "
            "Hãy thử tăng giá trị lambda."
        )

    y_hat    = X_design @ beta_hat
    residuals = y - y_hat
    rss      = float(residuals @ residuals)

    return {
        "beta_hat":  beta_hat,
        "y_hat":     y_hat,
        "residuals": residuals,
        "rss":       rss,
        "n":         n,
        "k":         k,
        "lam":       lam,
    }


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> dict:
    """
    Ước lượng Ridge Regression (closed-form).

    Tự động chèn cột intercept vào X trước khi giải.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)  — không chứa cột intercept.
    y : np.ndarray, shape (n,)
    lam : float >= 0,  default=1.0

    Returns
    -------
    dict: beta_hat (p+1,), y_hat, residuals, rss, n, k, lam.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones     = np.ones((X.shape[0], 1))
    X_design = np.hstack([ones, X])
    return _ridge_core_solver(X_design, y, lam)


class RidgeRegressor:
    """
    Ridge Regression với giao diện OOP tương tự OLSRegressor.

    Parameters
    ----------
    lam : float >= 0,  default=1.0
    fit_intercept : bool,  default=True
    """

    def __init__(self, lam: float = 1.0, fit_intercept: bool = True):
        self.lam           = lam
        self.fit_intercept = fit_intercept
        self.beta_hat      = None
        self._X_design     = None
        self._y            = None
        self._fitted_values = None
        self._residuals    = None
        self.n = self.k = None

    # ------------------------------------------------------------------
    def _prepare_design_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        X_design = self._prepare_design_matrix(X)
        y        = np.asarray(y, dtype=float).ravel()
        res      = _ridge_core_solver(X_design, y, self.lam)

        self.beta_hat       = res["beta_hat"]
        self._X_design      = X_design
        self._y             = y
        self._fitted_values = res["y_hat"]
        self._residuals     = res["residuals"]
        self.n, self.k      = res["n"], res["k"]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._prepare_design_matrix(X) @ self.beta_hat

    def _check_fitted(self):
        if self.beta_hat is None:
            raise RuntimeError("Lỗi: Gọi fit() trước khi sử dụng mô hình.")

    @property
    def residuals(self)     -> np.ndarray: 
        self._check_fitted()
        return self._residuals
    
    @property
    def fitted_values(self) -> np.ndarray: 
        self._check_fitted()
        return self._fitted_values

    # ------------------------------------------------------------------
    @staticmethod
    def ridge_trace(
        X: np.ndarray,
        y: np.ndarray,
        lambdas: np.ndarray = None,
        fit_intercept: bool = True,
        show: bool = True,
        save_path: str = None,
    ) -> tuple:
        """
        Vẽ Ridge Trace: sự thay đổi của từng beta_j theo lambda.

        Parameters
        ----------
        X, y          : dữ liệu huấn luyện.
        lambdas       : dãy lambda. Mặc định: 100 điểm log-scale [1e-4, 1e4].
        fit_intercept : có thêm intercept không.
        show          : hiển thị biểu đồ (plt.show()).
        save_path     : nếu không None, lưu biểu đồ vào đường dẫn này.

        Returns
        -------
        (lambdas, coef_paths)  — shape (n_lam,), (n_lam, k).
        """
        if lambdas is None:
            lambdas = np.logspace(-4, 4, 100)

        coef_paths = []
        for lam in lambdas:
            m = RidgeRegressor(lam=lam, fit_intercept=fit_intercept).fit(X, y)
            coef_paths.append(m.beta_hat)
        coef_paths = np.array(coef_paths)

        fig, ax = plt.subplots(figsize=(9, 5))
        start = 1 if fit_intercept else 0
        for j in range(start, coef_paths.shape[1]):
            ax.plot(lambdas, coef_paths[:, j], label=f"beta_{j}")
        ax.set_xscale("log")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("lambda (log scale)")
        ax.set_ylabel("Gia tri he so beta_hat")
        ax.set_title("Ridge Trace: He so hoi quy theo lambda")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
        if show:
            plt.show()
        plt.close(fig)

        return lambdas, coef_paths


# ============================================================
# LASSO REGRESSION (Coordinate Descent)
# ============================================================

def _soft_threshold(z: float, gamma: float) -> float:
    """
    Ham nguong mem (Soft Thresholding):
        S(z, gamma) = sign(z) * max(|z| - gamma, 0)

    Parameters
    ----------
    z     : gia tri can ap dung nguong.
    gamma : nguong (>= 0).
    """
    if gamma < 0:
        raise ValueError(f"gamma phải >= 0, nhưng nhận được {gamma}")
    return float(np.sign(z) * max(abs(z) - gamma, 0.0))


def _lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    tol_f: float = None,
) -> np.ndarray:
    """
    Coordinate Descent cho bai toan Lasso (intercept khong bi phat):

        min_{beta}  (1/2n) * ||y - X beta||^2  +  lambda * ||beta||_1

    Parameters
    ----------
    X        : (n, k) — da co cot intercept o vi tri 0.
    y        : (n,)
    lam      : he so phat >= 0.
    max_iter : so vong lap toi da.
    tol      : nguong hoi tu (max |Delta beta| < tol).
    tol_f    : ngưỡng hội tụ bổ sung cho relative change của hàm mục tiêu.
               Nếu None, chỉ dùng tol. Khuyến nghị 1e-8 khi cần hội tụ chặt.

    Returns
    -------
    beta : (k,)
    """
    n, k   = X.shape
    beta   = np.zeros(k)
    residuals = y.copy()          # residuals = y - X @ beta, hiện tại beta = 0
    x_sq   = np.sum(X ** 2, axis=0)   # ||x_j||^2, shape (k,)

    # Hàm mục tiêu ban đầu (nếu tol_f được chỉ định)
    if tol_f is not None:
        r     = y - X @ beta
        f_old = (0.5 / n) * float(r @ r) + lam * float(np.sum(np.abs(beta[1:])))
        
    for _ in range(max_iter):
        beta_old = beta.copy()

        for j in range(k):
            if x_sq[j] == 0.0:
                continue

            residuals += X[:, j] * beta[j]
            # Tính tương quan giữa biến j và residuals hiện tại
            rho_j = float(X[:, j] @ residuals)

            if j == 0:
                # Intercept: khong bi phat
                beta[j] = rho_j / x_sq[j]
            else:
                # n * lam la nguong tuong ung khi ham muc tieu co (1/2n)
                beta[j] = _soft_threshold(rho_j, n * lam) / x_sq[j]
                
            # Đưa đóng góp MỚI của biến j vào residuals
            residuals -= X[:, j] * beta[j]

         # Hội tụ 1 – theo thay đổi của β
        if np.max(np.abs(beta - beta_old)) < tol:
            break

        # Hội tụ 2 – theo thay đổi của hàm mục tiêu (nếu được kích hoạt)
        if tol_f is not None:
            r     = y - X @ beta
            f_new = (0.5 / n) * float(r @ r) + lam * float(np.sum(np.abs(beta[1:])))
            if np.abs(f_new - f_old) / max(1.0, np.abs(f_old)) < tol_f:
                break
            f_old = f_new

    return beta


def lasso_fit(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    tol_f: float = None,
) -> dict:
    """
    Ước lượng Lasso Regression bằng Coordinate Descent.

    Tự động chèn cột intercept vào X trước khi giải.

    Parameters
    ----------
    X        : (n, p) — khong chua cot intercept.
    y        : (n,)
    lam      : float >= 0,  default=1.0
    max_iter : int,          default=1000
    tol      : float,        default=1e-6
    tol_f    : float or None, default=None. Nếu không None, thêm điều kiện hội tụ dựa trên relative change của hàm mục tiêu.

    Returns
    -------
    dict: beta_hat (p+1,), y_hat, residuals, rss, n_nonzero, n, k, lam.
    """
    if lam < 0:
        raise ValueError(f"Lỗi: lambda phải >= 0, nhưng nhận được lambda={lam}.")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    ones     = np.ones((X.shape[0], 1))
    X_design = np.hstack([ones, X])

    beta_hat  = _lasso_coordinate_descent(X_design, y, lam, max_iter, tol, tol_f = tol_f)
    y_hat     = X_design @ beta_hat
    residuals = y - y_hat
    rss       = float(residuals @ residuals)
    n_nonzero = int(np.sum(beta_hat[1:] != 0))

    return {
        "beta_hat":  beta_hat,
        "y_hat":     y_hat,
        "residuals": residuals,
        "rss":       rss,
        "n_nonzero": n_nonzero,
        "n":         X_design.shape[0],
        "k":         X_design.shape[1],
        "lam":       lam,
    }


class LassoRegressor:
    """
    Lasso Regression với giao diện OOP tương tự RidgeRegressor.

    Parameters
    ----------
    lam           : float >= 0,    default=1.0
    fit_intercept : bool,          default=True
    max_iter      : int,           default=1000
    tol           : float,         default=1e-6
    tol_f         : float or None, default=None
    """

    def __init__(
        self,
        lam:           float = 1.0,
        fit_intercept: bool  = True,
        max_iter:      int   = 1000,
        tol:           float = 1e-6,
        tol_f:         float = None,
    ):
        self.lam           = lam
        self.fit_intercept = fit_intercept
        self.max_iter      = max_iter
        self.tol           = tol
        self.beta_hat      = None
        self._X_design     = None
        self._y            = None
        self._fitted_values = None
        self._residuals    = None
        self.tol_f = tol_f

    def _prepare_design_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegressor":
        X_design = self._prepare_design_matrix(X)
        y        = np.asarray(y, dtype=float).ravel()

        self.beta_hat       = _lasso_coordinate_descent(
                                  X_design, y, self.lam, self.max_iter, self.tol, tol_f= self.tol_f)
        self._X_design      = X_design
        self._y             = y
        self._fitted_values = X_design @ self.beta_hat
        self._residuals     = y - self._fitted_values
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._prepare_design_matrix(X) @ self.beta_hat

    def _check_fitted(self):
        if self.beta_hat is None:
            raise RuntimeError("Lỗi: Gọi fit() trước khi sử dụng mô hình.")

    @property
    def residuals(self)     -> np.ndarray: 
        self._check_fitted()
        return self._residuals
    @property
    def fitted_values(self) -> np.ndarray: 
        self._check_fitted()
        return self._fitted_values

    @property
    def n_nonzero(self) -> int:
        """Số hệ số != 0 (không tính intercept)."""
        self._check_fitted()
        return int(np.sum(self.beta_hat[1:] != 0))

    @staticmethod
    def lasso_path(
        X: np.ndarray,
        y: np.ndarray,
        lambdas: np.ndarray = None,
        fit_intercept: bool = True,
        show: bool = True,
        save_path: str = None,
    ) -> tuple:
        """
        Vẽ Lasso Path: theo dõi sự thay đổi beta_j khi lambda tăng dần.

        Parameters
        ----------
        X, y          : dữ liệu.
        lambdas       : dãy lambda nhỏ → lớn. Mặc định: 50 điểm [1e-4, 1.0].
        fit_intercept : bool.
        show          : hiển thị biểu đồ.
        save_path     : lưu biểu đồ nếu không None.

        Returns
        -------
        (lambdas, coef_paths)
        """
        if lambdas is None:
            lambdas = np.logspace(-4, 0, 50)

        coef_paths = []
        for lam in lambdas:
            m = LassoRegressor(lam=lam, fit_intercept=fit_intercept,
                               max_iter=5000, tol=1e-8).fit(X, y)
            coef_paths.append(m.beta_hat)
        coef_paths = np.array(coef_paths)

        fig, ax = plt.subplots(figsize=(9, 5))
        start = 1 if fit_intercept else 0
        for j in range(start, coef_paths.shape[1]):
            ax.plot(lambdas, coef_paths[:, j], label=f"beta_{j}")
        ax.set_xscale("log")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("lambda (log scale)")
        ax.set_ylabel("Gia tri he so beta_hat")
        ax.set_title("Lasso Path: He so hoi quy theo lambda")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
        if show:
            plt.show()
        plt.close(fig)

        return lambdas, coef_paths


# ============================================================
# KIEM CHUNG VOI SKLEARN
# ============================================================

def verify_ridge_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1.0,
    tol: float = 1e-6,
) -> dict:
    """
    So sanh RidgeRegressor (scratch) voi sklearn.linear_model.Ridge.

    Parameters
    ----------
    X, y : du lieu.
    lam  : he so phat.
    tol  : nguong sai so cho phep.

    Returns
    -------
    dict: beta_scratch, beta_sklearn, max_diff, passed.
    """
    from sklearn.linear_model import Ridge

    model_scratch  = RidgeRegressor(lam=lam).fit(X, y)
    beta_scratch   = model_scratch.beta_hat

    model_sk       = Ridge(alpha=lam, fit_intercept=True)
    model_sk.fit(X, y)
    beta_sklearn   = np.concatenate([[model_sk.intercept_], model_sk.coef_])

    max_diff = float(np.max(np.abs(beta_scratch - beta_sklearn)))
    passed   = max_diff < tol

    print("\n         VERIFICATION: Ridge Scratch vs sklearn")
    print(f"  {'Coef':>12} {'Scratch':>14} {'sklearn':>14} {'|delta|':>14}")
    labels = ["intercept"] + [f"beta_{i}" for i in range(1, len(beta_scratch))]
    for label, b_s, b_sk in zip(labels, beta_scratch, beta_sklearn):
        print(f"  {label:>12}  {b_s:>13.6f}  {b_sk:>13.6f}  {abs(b_s - b_sk):>13.2e}")
    print(f"  lambda = {lam}")
    print(f"  Max |beta_scratch - beta_sklearn| = {max_diff:.2e}")
    print(f"  Ket qua khop (tol={tol:.0e})       : {'PASS' if passed else 'FAIL'}")

    return {"beta_scratch": beta_scratch, "beta_sklearn": beta_sklearn,
            "max_diff": max_diff, "passed": passed}


def verify_lasso_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 0.1,
    tol: float = 1e-4,
) -> dict:
    """
    So sanh LassoRegressor (scratch) voi sklearn.linear_model.Lasso.

    Quy uoc: ca hai deu toi uu (1/2n)||y - Xb||^2 + lam||b||_1
    (sklearn dung alpha = lam, normalize theo n trong ham mat mat).

    Parameters
    ----------
    X, y : du lieu.
    lam  : he so phat.
    tol  : nguong sai so (noi long hon Ridge vi dung iterative solver).

    Returns
    -------
    dict: beta_scratch, beta_sklearn, max_diff, passed.
    """
    from sklearn.linear_model import Lasso

    model_scratch = LassoRegressor(lam=lam, max_iter=10000, tol=1e-8).fit(X, y)
    beta_scratch  = model_scratch.beta_hat

    model_sk      = Lasso(alpha=lam, fit_intercept=True, max_iter=10000, tol=1e-8)
    model_sk.fit(X, y)
    beta_sklearn  = np.concatenate([[model_sk.intercept_], model_sk.coef_])

    max_diff = float(np.max(np.abs(beta_scratch - beta_sklearn)))
    passed   = max_diff < tol

    print("\n         VERIFICATION: Lasso Scratch vs sklearn")
    print(f"  {'Coef':>12} {'Scratch':>14} {'sklearn':>14} {'|delta|':>14}")
    labels = ["intercept"] + [f"beta_{i}" for i in range(1, len(beta_scratch))]
    for label, b_s, b_sk in zip(labels, beta_scratch, beta_sklearn):
        print(f"  {label:>12}  {b_s:>13.6f}  {b_sk:>13.6f}  {abs(b_s - b_sk):>13.2e}")
    print(f"  lambda = {lam}")
    print(f"  Max |beta_scratch - beta_sklearn| = {max_diff:.2e}")
    print(f"  Ket qua khop (tol={tol:.0e})      : {'PASS' if passed else 'FAIL'}")

    return {"beta_scratch": beta_scratch, "beta_sklearn": beta_sklearn,
            "max_diff": max_diff, "passed": passed}