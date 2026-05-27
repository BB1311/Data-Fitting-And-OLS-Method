"""
ridge_lasso.py
==============
Cài đặt Ridge và Lasso Regression từ đầu (from scratch).

Lý thuyết
---------
Ridge (L2):  β̂_ridge = argmin  ½n·‖y − Xβ‖² + λ‖β‖²
                       = (XᵀX + λI_p)⁻¹ Xᵀy     ← closed-form
             (intercept KHÔNG bị phạt: penalty[0,0] = 0)

Lasso (L1):  β̂_lasso = argmin  ½n·‖y − Xβ‖² + λ‖β‖₁
             Không có closed-form → giải bằng Coordinate Descent.
             Toán tử ngưỡng mềm: S(z, γ) = sign(z)·max(|z|−γ, 0)

Giao diện
---------
    RidgeRegressor(lam, fit_intercept)
    LassoRegressor(lam, fit_intercept, max_iter, tol, tol_f)

Cả hai đều có:
    .fit(X, y)          → self
    .predict(X)         → y_pred
    .score(X, y)        → R²
    .coef_              → β̂ (không tính intercept), shape (p,)
    .intercept_         → β̂₀ (scalar), 0.0 nếu fit_intercept=False
    .fitted_values      → ŷ trên tập train
    .residuals          → y − ŷ trên tập train

Kiểm chứng
----------
    verify_ridge(X, y, lam)   → so sánh với sklearn.Ridge
    verify_lasso(X, y, lam)   → so sánh với sklearn.Lasso
"""

import numpy as np
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
# BASE CLASS — dùng chung cho Ridge và Lasso
# ══════════════════════════════════════════════════════════════════════
class _BaseRegressor:
    """
    Base class chứa logic dùng chung:
      - _build_design_matrix : thêm cột 1 nếu fit_intercept=True
      - predict, score       : dùng sau khi fit()
      - coef_, intercept_    : tách hệ số từ beta_hat
      - fitted_values, residuals : property chỉ đọc
      - _check_fitted        : guard trước predict/score/property
    """

    def __init__(self, lam: float, fit_intercept: bool):
        if lam < 0:
            raise ValueError(f"lambda phải >= 0, nhận được {lam}.")
        self.lam           = lam
        self.fit_intercept = fit_intercept
        # Các thuộc tính gán sau fit()
        self._beta_hat      : np.ndarray | None = None
        self._fitted_values : np.ndarray | None = None
        self._residuals     : np.ndarray | None = None

    # ------------------------------------------------------------------
    # TIỆN ÍCH NỘI BỘ
    # ------------------------------------------------------------------
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

    def _store_results(self, beta_hat: np.ndarray, X_design: np.ndarray,
                       y: np.ndarray):
        """Lưu kết quả chung sau khi solver tìm được β̂."""
        self._beta_hat      = beta_hat
        self._fitted_values = X_design @ beta_hat
        self._residuals     = y - self._fitted_values

    # ------------------------------------------------------------------
    # GIAO DIỆN CÔNG KHAI
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán ŷ = X_design · β̂."""
        self._check_fitted()
        return self._build_design_matrix(X) @ self._beta_hat

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Tính R² trên tập (X, y).

            R² = 1 − RSS / TSS

        Returns
        -------
        float — R² ∈ (−∞, 1].  R²=1 là khớp hoàn hảo.
        """
        self._check_fitted()
        y      = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        rss    = float(np.sum((y - y_pred) ** 2))
        tss    = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - rss / tss if tss > 0 else 0.0

    # ------------------------------------------------------------------
    # PROPERTIES (chỉ đọc)
    # ------------------------------------------------------------------
    @property
    def coef_(self) -> np.ndarray:
        """β̂ của các feature (không tính intercept), shape (p,)."""
        self._check_fitted()
        return self._beta_hat[1:] if self.fit_intercept else self._beta_hat

    @property
    def intercept_(self) -> float:
        """β̂₀ (scalar).  0.0 nếu fit_intercept=False."""
        self._check_fitted()
        return float(self._beta_hat[0]) if self.fit_intercept else 0.0

    @property
    def fitted_values(self) -> np.ndarray:
        """ŷ = X_design · β̂ trên tập train."""
        self._check_fitted()
        return self._fitted_values

    @property
    def residuals(self) -> np.ndarray:
        """ê = y − ŷ trên tập train."""
        self._check_fitted()
        return self._residuals


# ══════════════════════════════════════════════════════════════════════
# RIDGE REGRESSION
# ══════════════════════════════════════════════════════════════════════
class RidgeRegressor(_BaseRegressor):
    """
    Ridge Regression — closed-form solution.

        β̂_ridge = (XᵀX + λ·D)⁻¹ Xᵀy

    D là ma trận phạt: D = diag(0, 1, 1, …, 1)
    (intercept không bị phạt).

    Parameters
    ----------
    lam           : float ≥ 0, hệ số phạt λ  (mặc định 1.0)
    fit_intercept : bool, có thêm intercept không  (mặc định True)

    Ví dụ
    -----
    >>> reg = RidgeRegressor(lam=1.0).fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    >>> print(reg.score(X_test, y_test))
    """

    def __init__(self, lam: float = 1.0, fit_intercept: bool = True):
        super().__init__(lam=lam, fit_intercept=fit_intercept)

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        """
        Giải phương trình Ridge:

            (XᵀX + λ·D) β = Xᵀy

        Dùng np.linalg.solve (nhanh hơn inv và số học ổn định hơn).

        Parameters
        ----------
        X : (n, p)  — chưa có cột bias
        y : (n,)
        """
        X_design = self._build_design_matrix(X)
        y        = np.asarray(y, dtype=float).ravel()
        n, k     = X_design.shape

        # Ma trận phạt: không phạt intercept (vị trí [0,0])
        D         = np.eye(k)
        D[0, 0]   = 0.0
        A         = X_design.T @ X_design + self.lam * D
        b         = X_design.T @ y

        try:
            beta_hat = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback sang pseudo-inverse nếu A suy biến
            beta_hat = np.linalg.pinv(A) @ b

        self._store_results(beta_hat, X_design, y)
        return self

    # ------------------------------------------------------------------
    # RIDGE TRACE (static)
    # ------------------------------------------------------------------
    @staticmethod
    def ridge_trace(
        X        : np.ndarray,
        y        : np.ndarray,
        lambdas  : np.ndarray = None,
        fit_intercept: bool   = True,
        ax       : plt.Axes   = None,
        show     : bool       = True,
        save_path: str        = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vẽ Ridge Trace: hệ số β̂_j theo λ (log-scale).

        Parameters
        ----------
        X, y          : dữ liệu huấn luyện
        lambdas       : dãy λ (mặc định 100 điểm log-space [1e-4, 1e4])
        fit_intercept : bool
        ax            : plt.Axes hiện có; None → tạo Figure mới
        show          : gọi plt.show() sau khi vẽ
        save_path     : đường dẫn lưu hình (None = không lưu)

        Returns
        -------
        lambdas      : np.ndarray, shape (n_lam,)
        coef_paths   : np.ndarray, shape (n_lam, p)  — không tính intercept
        """
        if lambdas is None:
            lambdas = np.logspace(-4, 4, 100)

        coef_paths = []
        for lam in lambdas:
            reg = RidgeRegressor(lam=lam, fit_intercept=fit_intercept).fit(X, y)
            coef_paths.append(reg.coef_)
        coef_paths = np.array(coef_paths)  # (n_lam, p)

        _own_fig = ax is None
        if _own_fig:
            fig, ax = plt.subplots(figsize=(9, 5))

        p = coef_paths.shape[1]
        for j in range(p):
            ax.plot(lambdas, coef_paths[:, j], label=f"β_{j+1}")

        ax.set_xscale("log")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("λ (log scale)")
        ax.set_ylabel("Hệ số hồi quy β̂")
        ax.set_title("Ridge Trace: Hệ số theo λ")
        ax.legend(loc="upper right", fontsize=8, ncol=max(1, p // 8))
        ax.grid(True, alpha=0.3)

        if _own_fig:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=120, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()

        return lambdas, coef_paths

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        status = "fitted" if self._beta_hat is not None else "not fitted"
        return f"RidgeRegressor(lam={self.lam}, fit_intercept={self.fit_intercept}, status={status})"


# ══════════════════════════════════════════════════════════════════════
# SOFT-THRESHOLDING OPERATOR  (hàm nội bộ, dùng trong Lasso CD)
# ══════════════════════════════════════════════════════════════════════
def _soft_threshold(z: np.ndarray, gamma: float) -> np.ndarray:
    """
    Toán tử ngưỡng mềm (vectorized):

        S(z, γ) = sign(z) · max(|z| − γ, 0)

    Parameters
    ----------
    z     : scalar hoặc ndarray
    gamma : float ≥ 0

    Returns
    -------
    ndarray cùng shape với z
    """
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


# ══════════════════════════════════════════════════════════════════════
# LASSO REGRESSION
# ══════════════════════════════════════════════════════════════════════
class LassoRegressor(_BaseRegressor):
    """
    Lasso Regression — Coordinate Descent.

    Bài toán tối ưu (quy ước ½n):

        min_β  ½n · ‖y − Xβ‖² + λ · ‖β‖₁

    Giải bằng Cyclic Coordinate Descent:
        • Mỗi bước cập nhật β_j, giữ cố định tất cả β_{k≠j}.
        • Residual được maintain tăng dần (O(n) mỗi bước, không rebuild).
        • Intercept không bị phạt.

    Parameters
    ----------
    lam           : float ≥ 0, hệ số phạt λ        (mặc định 1.0)
    fit_intercept : bool                             (mặc định True)
    max_iter      : int, số vòng lặp tối đa          (mặc định 1000)
    tol           : float, ngưỡng hội tụ ‖Δβ‖_∞     (mặc định 1e-6)
    tol_f         : float | None, ngưỡng hội tụ      (mặc định None)
                    bổ sung theo relative change của objective.
                    Khuyến nghị 1e-8 khi cần chặt hơn.

    Ví dụ
    -----
    >>> reg = LassoRegressor(lam=0.1, max_iter=2000).fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    >>> print(f"R²={reg.score(X_test, y_test):.4f}, nnz={reg.n_nonzero}")
    """

    def __init__(
        self,
        lam           : float        = 1.0,
        fit_intercept : bool         = True,
        max_iter      : int          = 1000,
        tol           : float        = 1e-6,
        tol_f         : float | None = None,
    ):
        super().__init__(lam=lam, fit_intercept=fit_intercept)
        self.max_iter = max_iter
        self.tol      = tol
        self.tol_f    = tol_f
        self.n_iter_  : int | None = None   # số vòng lặp thực tế đến hội tụ

    # ------------------------------------------------------------------
    # FIT — Cyclic Coordinate Descent
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegressor":
        """
        Giải Lasso bằng Coordinate Descent.

        Cập nhật từng β_j theo công thức:

            ρ_j   = Xⱼᵀ · r⁽ʲ⁾           (partial residual)
            β_j   = S(ρ_j, n·λ) / ‖Xⱼ‖²  (j ≥ 1, feature)
            β_0   = ρ_0 / ‖X₀‖²           (intercept, không phạt)

        Parameters
        ----------
        X : (n, p) — chưa có cột bias
        y : (n,)
        """
        X_design = self._build_design_matrix(X)
        y        = np.asarray(y, dtype=float).ravel()
        n, k     = X_design.shape

        beta      = np.zeros(k)
        residuals = y.copy()                        # r = y − X·β, ban đầu β=0
        x_sq      = (X_design ** 2).sum(axis=0)    # ‖Xⱼ‖², shape (k,)

        # Lưu objective ban đầu (dùng nếu tol_f được bật)
        f_old = self._objective(y, beta, X_design, n) if self.tol_f else None

        for iteration in range(self.max_iter):
            beta_old = beta.copy()

            for j in range(k):
                if x_sq[j] == 0.0:
                    continue                         # cột hằng → bỏ qua

                # Khôi phục partial residual: r⁽ʲ⁾ = r + Xⱼ·β_j
                residuals += X_design[:, j] * beta[j]
                rho_j      = float(X_design[:, j] @ residuals)

                if j == 0:
                    # Intercept — không phạt
                    beta[j] = rho_j / x_sq[j]
                else:
                    # Feature — phạt ngưỡng mềm (n·λ vì objective dùng ½n)
                    beta[j] = float(_soft_threshold(rho_j, n * self.lam) / x_sq[j])

                # Cập nhật lại residual sau khi cập nhật β_j
                residuals -= X_design[:, j] * beta[j]

            # ── Kiểm tra hội tụ ────────────────────────────────────
            # Điều kiện 1: thay đổi hệ số
            if np.max(np.abs(beta - beta_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break

            # Điều kiện 2: relative change của objective (nếu bật)
            if self.tol_f is not None:
                f_new = self._objective(y, beta, X_design, n)
                if np.abs(f_new - f_old) / max(1.0, np.abs(f_old)) < self.tol_f:
                    self.n_iter_ = iteration + 1
                    break
                f_old = f_new
        else:
            self.n_iter_ = self.max_iter

        self._store_results(beta, X_design, y)
        return self

    # ------------------------------------------------------------------
    # TIỆN ÍCH NỘI BỘ
    # ------------------------------------------------------------------
    @staticmethod
    def _objective(y: np.ndarray, beta: np.ndarray,
                   X_design: np.ndarray, n: int) -> float:
        """
        Tính objective function:
            f(β) = ½n · ‖y − X·β‖² + λ · ‖β[1:]‖₁
        Dùng để kiểm tra hội tụ theo tol_f.
        """
        r = y - X_design @ beta
        return float((0.5 / n) * (r @ r))

    # ------------------------------------------------------------------
    # PROPERTY BỔ SUNG
    # ------------------------------------------------------------------
    @property
    def n_nonzero(self) -> int:
        """Số hệ số feature ≠ 0 (không tính intercept)."""
        self._check_fitted()
        return int(np.sum(np.abs(self.coef_) > 1e-10))

    @property
    def sparsity(self) -> float:
        """Tỉ lệ hệ số = 0: sparsity = 1 − n_nonzero / p."""
        self._check_fitted()
        p = len(self.coef_)
        return 1.0 - self.n_nonzero / p if p > 0 else 0.0

    # ------------------------------------------------------------------
    # LASSO PATH (static)
    # ------------------------------------------------------------------
    @staticmethod
    def lasso_path(
        X        : np.ndarray,
        y        : np.ndarray,
        lambdas  : np.ndarray = None,
        fit_intercept: bool   = True,
        ax       : plt.Axes   = None,
        show     : bool       = True,
        save_path: str        = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vẽ Lasso Path: hệ số β̂_j theo λ (log-scale).
        Khi λ tăng, hệ số co dần về 0 (sparse).

        Parameters
        ----------
        X, y          : dữ liệu huấn luyện
        lambdas       : dãy λ (mặc định 50 điểm log-space [1e-4, 1])
        fit_intercept : bool
        ax            : plt.Axes hiện có; None → tạo Figure mới
        show          : gọi plt.show() sau khi vẽ
        save_path     : đường dẫn lưu hình (None = không lưu)

        Returns
        -------
        lambdas    : np.ndarray, shape (n_lam,)
        coef_paths : np.ndarray, shape (n_lam, p)
        """
        if lambdas is None:
            lambdas = np.logspace(-4, 0, 50)

        coef_paths = []
        for lam in lambdas:
            reg = LassoRegressor(lam=lam, fit_intercept=fit_intercept,
                                 max_iter=5000, tol=1e-8).fit(X, y)
            coef_paths.append(reg.coef_)
        coef_paths = np.array(coef_paths)  # (n_lam, p)

        _own_fig = ax is None
        if _own_fig:
            fig, ax = plt.subplots(figsize=(9, 5))

        p = coef_paths.shape[1]
        for j in range(p):
            ax.plot(lambdas, coef_paths[:, j], label=f"β_{j+1}")

        ax.set_xscale("log")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("λ (log scale)")
        ax.set_ylabel("Hệ số hồi quy β̂")
        ax.set_title("Lasso Path: Hệ số theo λ (co dần về 0)")
        ax.legend(loc="upper right", fontsize=8, ncol=max(1, p // 8))
        ax.grid(True, alpha=0.3)

        if _own_fig:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=120, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()

        return lambdas, coef_paths

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        status = "fitted" if self._beta_hat is not None else "not fitted"
        return (f"LassoRegressor(lam={self.lam}, fit_intercept={self.fit_intercept}, "
                f"max_iter={self.max_iter}, tol={self.tol}, status={status})")


# ══════════════════════════════════════════════════════════════════════
# KIỂM CHỨNG VỚI SKLEARN
# ══════════════════════════════════════════════════════════════════════
def verify_ridge(
    X   : np.ndarray,
    y   : np.ndarray,
    lam : float = 1.0,
    tol : float = 1e-6,
) -> dict:
    """
    So sánh RidgeRegressor (scratch) với sklearn.Ridge.

    Returns
    -------
    dict: beta_scratch, beta_sklearn, max_diff, passed
    """
    from sklearn.linear_model import Ridge

    scratch  = RidgeRegressor(lam=lam).fit(X, y)
    sk       = Ridge(alpha=lam, fit_intercept=True).fit(X, y)

    b_scratch = np.concatenate([[scratch.intercept_], scratch.coef_])
    b_sklearn = np.concatenate([[sk.intercept_],      sk.coef_])
    max_diff  = float(np.max(np.abs(b_scratch - b_sklearn)))
    passed    = max_diff < tol

    _print_verify_table("Ridge", b_scratch, b_sklearn, lam, max_diff, tol, passed)
    return {"beta_scratch": b_scratch, "beta_sklearn": b_sklearn,
            "max_diff": max_diff, "passed": passed}


def verify_lasso(
    X   : np.ndarray,
    y   : np.ndarray,
    lam : float = 0.1,
    tol : float = 1e-4,
) -> dict:
    """
    So sánh LassoRegressor (scratch) với sklearn.Lasso.

    Quy ước hàm mất mát khớp nhau:
        scratch : ½n·‖y−Xβ‖² + λ·‖β‖₁
        sklearn : (1/2n)·‖y−Xβ‖² + alpha·‖β‖₁  với alpha = lam

    Returns
    -------
    dict: beta_scratch, beta_sklearn, max_diff, passed
    """
    from sklearn.linear_model import Lasso

    scratch   = LassoRegressor(lam=lam, max_iter=10_000, tol=1e-8).fit(X, y)
    sk        = Lasso(alpha=lam, fit_intercept=True, max_iter=10_000, tol=1e-8).fit(X, y)

    b_scratch = np.concatenate([[scratch.intercept_], scratch.coef_])
    b_sklearn = np.concatenate([[sk.intercept_],      sk.coef_])
    max_diff  = float(np.max(np.abs(b_scratch - b_sklearn)))
    passed    = max_diff < tol

    _print_verify_table("Lasso", b_scratch, b_sklearn, lam, max_diff, tol, passed)
    return {"beta_scratch": b_scratch, "beta_sklearn": b_sklearn,
            "max_diff": max_diff, "passed": passed}


def _print_verify_table(
    name     : str,
    b_scratch: np.ndarray,
    b_sklearn: np.ndarray,
    lam      : float,
    max_diff : float,
    tol      : float,
    passed   : bool,
):
    """In bảng so sánh scratch vs sklearn."""
    print(f"\n{'─'*56}")
    print(f"  VERIFICATION: {name} Scratch vs sklearn  (λ={lam})")
    print(f"{'─'*56}")
    print(f"  {'Hệ số':>12}  {'Scratch':>13}  {'sklearn':>13}  {'|Δ|':>11}")
    print(f"  {'─'*12}  {'─'*13}  {'─'*13}  {'─'*11}")
    labels = ["intercept"] + [f"β_{i}" for i in range(1, len(b_scratch))]
    for label, bs, bsk in zip(labels, b_scratch, b_sklearn):
        print(f"  {label:>12}  {bs:>13.6f}  {bsk:>13.6f}  {abs(bs-bsk):>11.2e}")
    print(f"{'─'*56}")
    print(f"  Max |Δ| = {max_diff:.2e}   tol = {tol:.0e}   "
          f"→  {'✓ PASS' if passed else '✗ FAIL'}")

