import numpy as np
from scipy import stats

def _ols_core_solver(X_design: np.ndarray, y: np.ndarray) -> dict:
    """
    Thực hiện giải phương trình Normal Equations trên ma trận đã chuẩn bị.
    Đây là nơi chứa công thức toán học cốt lõi.
    """
    n, k = X_design.shape
    df = n - k  # Bậc tự do của phần dư 
    
    # Chặn lỗi thiếu bậc tự do
    if df <= 0:
        raise ValueError(
            f"Lỗi: Không đủ bậc tự do (n={n} <= k={k}). "
            "Cần số lượng quan sát (n) lớn hơn số lượng tham số (k) để ước lượng phương sai nhiễu."
        )

    # Giải β̂ = (XᵀX)⁻¹ Xᵀy
    XtX = X_design.T @ X_design
    Xty = X_design.T @ y

    if np.linalg.matrix_rank(XtX) < k:
        raise ValueError(
            "Lỗi: Ma trận XᵀX không đủ hạng (đa cộng tuyến hoàn hảo hoặc gần hoàn hảo). "
            "Nghiệm OLS không tồn tại duy nhất."
        )

    beta_hat = np.linalg.solve(XtX, Xty)

    y_hat = X_design @ beta_hat
    residuals = y - y_hat
    rss = float(residuals @ residuals)
    sigma2_hat = rss / df

    return {
        "beta_hat": beta_hat,
        "sigma2_hat": sigma2_hat,
        "y_hat": y_hat,
        "residuals": residuals,
        "X_design": X_design,
        "n": n,    
        "k": k,    
        "df": df   
    }

def ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Tính nghiệm bằng phương pháp Bình phương tối thiểu (OLS) và ước lượng phương sai nhiễu.
    
    Hàm này tự động chèn thêm một cột toàn số 1 vào ma trận đầu vào 'X' để tính toán hệ số chặn (intercept / beta_0).

    Parameters
    ----------
    X : np.ndarray
        Ma trận đặc trưng đầu vào, kích thước (n, p). 
        (Không chứa sẵn cột hệ số chặn).
    y : np.ndarray
        Vector biến mục tiêu, kích thước (n,).

    Returns
    -------
    dict
        Một dictionary chứa các kết quả ước lượng, bao gồm:
        - "beta_hat" (np.ndarray): Vector hệ số ước lượng OLS, kích thước (p+1,).
        - "sigma2_hat" (float): Ước lượng không chệch của phương sai nhiễu.
        - "y_hat" (np.ndarray): Vector giá trị dự đoán ŷ = Xβ̂.
        - "residuals" (np.ndarray): Vector phần dư ε̂ = y - ŷ.
        - "X_design" (np.ndarray): Ma trận thiết kế đã thêm cột 1, kích thước (n, p+1).
        - "n" (int): Số lượng quan sát.
        - "k" (int): Số lượng tham số mô hình (bao gồm cả hệ số chặn).
        - "df" (int): Bậc tự do của phần dư (n - k).

    Raises
    ------
    ValueError
        Nếu ma trận thiết kế không có hạng đầy đủ (rank < p + 1), thường do lỗi đa cộng tuyến hoàn hảo.
    """
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1))
    X_design = np.hstack([ones, X.reshape(X.shape[0], -1)])
    
    return _ols_core_solver(X_design, y)

class OLSRegressor:
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.beta_hat = None
        self.sigma2_hat = None
        self.n = None
        self.k = None
        self.df = None

        self._X_design = None
        self._y = None
        self._residuals = None
        self._fitted_values = None

    def _prepare_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Đồng bộ logic thêm intercept cho cả fit và predict."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: 
            X = X.reshape(-1, 1)
        
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            return np.hstack([ones, X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OLSRegressor":
        X_design = self._prepare_design_matrix(X)
        y = np.asarray(y, dtype=float).ravel()
        
        res = _ols_core_solver(X_design, y)
        self.beta_hat = res["beta_hat"]
        self.sigma2_hat = res["sigma2_hat"]
        self.n = res["n"]
        self.k = res["k"]
        self.df = res["df"]
        
        self._X_design = res["X_design"]
        self._y = y
        self._residuals = res["residuals"]
        self._fitted_values = res["y_hat"]
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_design = self._prepare_design_matrix(X)
        return X_design @ self.beta_hat

    def _check_fitted(self):
        """Kiểm tra mô hình đã được fit chưa."""
        if self.beta_hat is None:
            raise RuntimeError("Lỗi: Gọi fit() trước khi sử dụng mô hình.")
        
    @property
    def residuals(self):
        """Trả về vector phần dư (residuals) trên tập huấn luyện."""
        self._check_fitted()
        return self._residuals

    @property
    def fitted_values(self):
        """Trả về vector giá trị dự báo (fitted values) trên tập huấn luyện."""
        self._check_fitted()
        return self._fitted_values
    
def compute_rss(y: np.ndarray, y_hat: np.ndarray) -> float:
    """RSS = ||y - ŷ||² (Residual Sum of Squares - Tổng bình phương phần dư)"""
    residuals = np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float)
    return float(residuals @ residuals)

def compute_tss(y: np.ndarray) -> float:
    """TSS = ||y - ȳ||² (Total Sum of Squares - Tổng bình phương toàn phần)"""
    y = np.asarray(y, dtype=float)
    return float(np.sum((y - y.mean()) ** 2))

def compute_ess(y: np.ndarray, y_hat: np.ndarray) -> float:
    """ESS = ||ŷ - ȳ||² (Explained Sum of Squares - Tổng bình phương giải thích được)"""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    y_bar = y.mean()
    diff = y_hat - y_bar
    return float(diff @ diff)

def compute_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Hệ số xác định R² (Dùng khi gọi độc lập)"""
    rss = compute_rss(y, y_hat)
    tss = compute_tss(y)
    if tss == 0:
        raise ValueError("TSS = 0")
    return 1.0 - rss / tss

def compute_r2_adj(y: np.ndarray, y_hat: np.ndarray, p: int) -> float:
    """R² hiệu chỉnh (Dùng khi gọi độc lập)"""
    n = len(y)
    if n <= p + 1:
        raise ValueError("n phải lớn hơn p+1.")
    r2 = compute_r2(y, y_hat)
    return 1.0 - (n - 1) / (n - p - 1) * (1.0 - r2)

def model_metrics(y: np.ndarray, y_hat: np.ndarray, p: int, verbose: bool = True) -> dict:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    n = len(y)

    rss = compute_rss(y, y_hat)
    tss = compute_tss(y)
    ess = compute_ess(y, y_hat)
    r2 = compute_r2(y, y_hat)
    r2_adj = compute_r2_adj(y, y_hat, p)

    df_model = p
    df_resid = n - p - 1
    
    if df_model == 0 or df_resid <= 0:
        f_stat = np.nan
        f_pvalue = np.nan
    else:
        f_stat = (ess / df_model) / (rss / df_resid)
        f_pvalue = float(1.0 - stats.f.cdf(f_stat, df_model, df_resid))

    metrics = {
        "rss":      rss,
        "tss":      tss,
        "ess":      ess,
        "r2":       r2,
        "r2_adj":   r2_adj,
        "f_stat":   f_stat,
        "f_pvalue": f_pvalue,
        "n":        n,
        "p":        p,
    }

    if verbose:
        print("        MODEL METRICS SUMMARY")
        print(f"  n (observations)  : {n}")
        print(f"  p (features)      : {p}")
        print(f"  RSS               : {rss:.6f}")
        print(f"  TSS               : {tss:.6f}")
        print(f"  ESS               : {ess:.6f}")
        print(f"  R²                : {r2:.6f}")
        print(f"  R² adjusted       : {r2_adj:.6f}")
        print(f"  F-statistic       : {f_stat:.4f}  (df1={df_model}, df2={df_resid})")
        print(f"  F p-value         : {f_pvalue:.6e}")

    return metrics

def verify_with_sklearn(X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # 1. OLS tự cài đặt (Scratch)
    model_scratch = OLSRegressor(fit_intercept=True)
    model_scratch.fit(X, y)
    beta_scratch = model_scratch.beta_hat 
    y_hat_scratch = model_scratch.predict(X)
    r2_scratch = compute_r2(y, y_hat_scratch)

    # 2. Sklearn
    model_sk = LinearRegression(fit_intercept=True)
    model_sk.fit(X, y)
    beta_sklearn = np.concatenate([[model_sk.intercept_], model_sk.coef_])
    y_hat_sklearn = model_sk.predict(X)
    r2_sklearn = r2_score(y, y_hat_sklearn)   

    # 3. Numpy lstsq
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    beta_numpy, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

    # So sánh sai số
    max_diff_sk    = np.max(np.abs(beta_scratch - beta_sklearn))
    max_diff_numpy = np.max(np.abs(beta_scratch - beta_numpy))
    diff_r2        = abs(r2_scratch - r2_sklearn)

    print("         VERIFICATION: OLS Scratch vs sklearn vs numpy")
    header = f"{'Coef':>12} {'Scratch':>14} {'sklearn':>14} {'numpy lstsq':>14}"
    print(header)
    labels = ["intercept"] + [f"β_{i}" for i in range(1, len(beta_scratch))]
    for label, b_s, b_sk, b_np in zip(labels, beta_scratch, beta_sklearn, beta_numpy):
        print(f"{label:>12}  {b_s:>13.6f}  {b_sk:>13.6f}  {b_np:>13.6f}")
    print(f"  Max |β_scratch - β_sklearn| = {max_diff_sk:.2e}")
    print(f"  Max |β_scratch - β_numpy  | = {max_diff_numpy:.2e}")
    print(f"  |R²_scratch - R²_sklearn|   = {diff_r2:.2e}")
    
    tol = 1e-8
    ok = max_diff_sk < tol and max_diff_numpy < tol and diff_r2 < tol
    print(f"  Kết quả khớp (tol={tol:.0e})    : {'PASS' if ok else 'FAIL'}")

    return {
        "beta_scratch":  beta_scratch,
        "beta_sklearn":  beta_sklearn,
        "beta_numpy":    beta_numpy,
        "y_hat_scratch": y_hat_scratch,
        "y_hat_sklearn": y_hat_sklearn,
        "max_diff_sk":   max_diff_sk,
        "max_diff_numpy": max_diff_numpy,
        "diff_r2":       diff_r2,
        "passed":        ok,
    }
    
if __name__ == "__main__":
    print("OLS IMPLEMENTATION DEMO")

    np.random.seed(42)
    n = 150
    X_demo = np.random.randn(n, 3)
    beta_true = np.array([5.0, 2.0, -1.5, 0.8])
    y_demo = beta_true[0] + X_demo @ beta_true[1:] + np.random.randn(n)

    print(f"\n[Demo] n={n}, p=3, β_true={beta_true}")

    res = ols_fit(X_demo, y_demo)
    print(f"\nols_fit() → β̂={res['beta_hat']}, σ̂²={res['sigma2_hat']:.4f}")

    model = OLSRegressor().fit(X_demo, y_demo)
    y_hat_demo = model.predict(X_demo)
    print(f"OLSRegressor → β̂={model.beta_hat}")

    model_metrics(y_demo, y_hat_demo, p=3, verbose=True)
    verify_with_sklearn(X_demo, y_demo)