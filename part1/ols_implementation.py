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
        - "X_design" (np.ndarray): Ma trận design đã thêm cột 1, kích thước (n, p+1).
        - "n" (int): Số lượng quan sát.
        - "k" (int): Số lượng tham số mô hình (bao gồm cả hệ số chặn).
        - "df" (int): Bậc tự do của phần dư (n - k).

    Raises
    ------
    ValueError
        Nếu ma trận design không có hạng đầy đủ (rank < p + 1), thường do lỗi đa cộng tuyến hoàn hảo.
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

    def summary(self):
        """In ra bảng tóm tắt các phép kiểm định hệ số"""
        self._check_fitted()
        # Gọi lại hàm phía trên, truyền X đã tiền xử lý, bỏ qua bước check intercept
        return coef_inference(
            X=self._X_design, 
            y=self._y, 
            beta_hat=self.beta_hat, 
            sigma2=self.sigma2_hat
        )


def hat_matrix(X: np.ndarray, add_intercept: bool = True) -> tuple[np.ndarray, bool]:
    """
    Tính Ma trận chiếu (Hat Matrix) H = X(X^T X)^-1 X^T 
    và kiểm tra tính lũy đẳng (idempotent: H^2 = H).
    
    Parameters
    ----------
    X : np.ndarray
        Ma trận đặc trưng đầu vào.
    add_intercept : bool, default=True
        Nếu True, tự động chèn thêm cột 1 vào X để tạo thành ma trận design.
        
    Returns
    -------
    tuple(np.ndarray, bool)
        - H: Ma trận chiếu kích thước (n, n).
        - is_idempotent: True nếu H @ H == H (trong phạm vi sai số cho phép).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    # Chuẩn bị ma trận design
    if add_intercept:
        ones = np.ones((X.shape[0], 1))
        X_design = np.hstack([ones, X])
    else:
        X_design = X
        
    # Tính X^T * X
    XtX = X_design.T @ X_design
    
    # Tính (X^T * X)^-1 với xử lý ngoại lệ cho ma trận suy biến
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        raise ValueError("Lỗi: Ma trận (X^T * X) không khả nghịch. Dữ liệu của bạn có thể đang gặp hiện tượng đa cộng tuyến hoàn hảo làm ma trận bị suy biến.")
    
    # Tính H = X * (X^T * X)^-1 * X^T
    H = X_design @ XtX_inv @ X_design.T
    
    # Kiểm tra tính lũy đẳng: H^2 = H
    # Sử dụng np.allclose để tránh lỗi sai số dấu phẩy động
    is_idempotent = np.allclose(H @ H, H)
    
    return H, is_idempotent


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


def coef_inference(X: np.ndarray, y: np.ndarray, beta_hat: np.ndarray, sigma2: float, verbose: bool = True) -> dict:
    """
    Tính standard errors, t-statistics, p-values và khoảng tin cậy 95% cho từng hệ số hồi quy.
    
    Parameters
    ----------
    X : np.ndarray
        Ma trận đặc trưng (n, p). Hàm sẽ tự động kiểm tra xem X đã có cột intercept hay chưa.
    y : np.ndarray
        Vector biến mục tiêu (n,).
    beta_hat : np.ndarray
        Vector hệ số hồi quy (p+1,).
    sigma2 : float
        Ước lượng phương sai nhiễu (sigma^2_hat).
    verbose : bool
        Nếu True, in bảng tóm tắt hệ số ra màn hình.
        
    Returns
    -------
    dict
        Chứa các mảng 'se', 't_stats', 'p_values', 'ci_lower', 'ci_upper'.
    """
    X = np.asarray(X, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float).ravel()
    n = X.shape[0]
    
    # Đồng bộ độ dài của X với beta_hat (chèn thêm cột 1 nếu X đang là ma trận gốc)
    if X.shape[1] == len(beta_hat) - 1:
        ones = np.ones((n, 1))
        X_design = np.hstack([ones, X.reshape(n, -1)])
    elif X.shape[1] == len(beta_hat):
        X_design = X
    else:
        raise ValueError("Số lượng cột của X không khớp với kích thước của vector beta_hat.")
        
    k = X_design.shape[1] # Số tham số (p + 1)
    df = n - k
    
    if df <= 0:
        raise ValueError("Không đủ bậc tự do (n <= k) để thực hiện kiểm định giả thuyết.")
        
    # Tính ma trận (X^T * X)^-1
    XtX = X_design.T @ X_design
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        raise ValueError("Lỗi: Ma trận (X^T * X) suy biến. Không thể tính sai số chuẩn.")
        
    # 1. Sai số chuẩn (Standard Error - SE)
    # Lấy đường chéo chính của ma trận hiệp phương sai
    var_beta = sigma2 * np.diag(XtX_inv)
    se = np.sqrt(var_beta)
    
    # 2. Kiểm định Student (t-statistic)
    t_stats = beta_hat / se
    
    # 3. p-values (Kiểm định 2 phía: H0: beta_j = 0)
    p_values = 2 * (1.0 - stats.t.cdf(np.abs(t_stats), df=df))
    
    # 4. Khoảng tin cậy 95%
    alpha = 0.05
    # Tìm giá trị tới hạn t (critical value): là giới hạn mà ta sử dụng để ra quyết định chấp nhận hay bác bỏ H0
    # |t_stats| > t_crit thì bác bỏ H0 (Trong hàm này, dùng p-values để kiểm định chứ không dùng cách này)
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df)
    
    ci_lower = beta_hat - t_crit * se
    ci_upper = beta_hat + t_crit * se
    
    if verbose:
        print("\n        COEFFICIENT INFERENCE SUMMARY")
        header = f"{'Coef':>10} {'Estimate':>12} {'Std Error':>12} {'t value':>10} {'Pr(>|t|)':>12} {'[0.025':>12} {'0.975]':>12}"
        print(header)
        print("-" * 88)
        for i in range(k):
            # Hiển thị nhãn phù hợp
            label = "Intercept" if (i == 0 and X.shape[1] == k - 1) else f"X_{i}"
            
            # Format p-value với dấu '*' nếu có ý nghĩa thống kê (p < 0.05)
            p_val_str = f"{p_values[i]:12.4e}"
            sig_mark = "*" if p_values[i] < 0.05 else " "
            
            print(f"{label:>10} {beta_hat[i]:12.6f} {se[i]:12.6f} {t_stats[i]:10.4f} {p_val_str}{sig_mark} {ci_lower[i]:12.6f} {ci_upper[i]:12.6f}")
        print("-" * 88)
        print(f"Bậc tự do (df) = {df}.  Mức ý nghĩa alpha = {alpha}. (*) p < 0.05")

    return {
        "se": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_crit": t_crit
    }


def vif(X: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    Tính Hệ số phóng đại phương sai - Variance Inflation Factor (VIF) cho từng biến đặc trưng trong X.
    
    Công thức: VIF_j = 1 / (1 - R_j^2)
    Trong đó R_j^2 là hệ số xác định khi hồi quy biến X_j theo tất cả các biến X còn lại.
    
    Parameters
    ----------
    X : np.ndarray
        Ma trận đặc trưng đầu vào, kích thước (n, p). 
        (Đưa vào X gốc, không bao gồm cột intercept toàn số 1).
    verbose : bool
        Nếu True, in kết quả VIF ra màn hình.
        
    Returns
    -------
    np.ndarray
        Vector chứa giá trị VIF cho từng biến, kích thước (p,).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    vif_values = np.zeros(p)
    
    for j in range(p):
        # Tách biến mục tiêu X_j và các biến giải thích X_{-j}
        y_target = X[:, j]
        X_features = np.delete(X, j, axis=1)
        
        # Thêm cột intercept cho ma trận X_features
        ones = np.ones((n, 1))
        X_design = np.hstack([ones, X_features])
        
        # Giải OLS: Dùng np.linalg.lstsq thay vì nghịch đảo trực tiếp
        # để đảm bảo thuật toán ổn định dù ma trận có lân cận suy biến
        beta_hat, _, _, _ = np.linalg.lstsq(X_design, y_target, rcond=None)
        
        # Tính giá trị dự đoán
        y_pred = X_design @ beta_hat
        
        # Tính R^2 cho mô hình hồi quy hiện tại
        tss = np.sum((y_target - np.mean(y_target)) ** 2)
        rss = np.sum((y_target - y_pred) ** 2)
        
        # Xử lý trường hợp TSS = 0 (biến X_j là hằng số)
        if tss == 0:
            r2 = 0.0
        else:
            r2 = 1.0 - (rss / tss)
        
        # Xử lý trường hợp R^2 tiến gần 1 (đa cộng tuyến hoàn hảo)
        # Mẫu số (1 - r2) tiến về 0 thì VIF tiến ra vô cùng
        if r2 >= 1.0 or np.isclose(r2, 1.0):
            vif_values[j] = np.inf
        else:
            vif_values[j] = 1.0 / (1.0 - r2)
            
    if verbose:
        print("\n        VARIANCE INFLATION FACTOR (VIF)")
        print("-" * 45)
        print(f"{'Feature':>10} {'VIF Score':>15} {'Warning':>15}")
        print("-" * 45)
        for j in range(p):
            warning = "Đa cộng tuyến nghiêm trọng!" if vif_values[j] > 10.0 else ""
            print(f"{'X_' + str(j):>10} {vif_values[j]:15.4f}  {warning}")
        print("-" * 45)
        print("Lưu ý: VIF > 10 cho thấy hiện tượng đa cộng tuyến nghiêm trọng.")
            
    return vif_values


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

    print("\n" + "="*50)
    print("DEMO: CÁC HÀM NÂNG CAO (HAT, INFERENCE, VIF)")
    print("="*50)

    # Minh họa Hat Matrix
    print("\n1. HAT MATRIX")
    H, is_idemp = hat_matrix(X_demo, add_intercept=True)
    print(f"  -> Kích thước ma trận chiếu H: {H.shape}")
    print(f"  -> Ma trận H có tính lũy đẳng (H^2 = H) không?: {is_idemp}")

    # Minh họa Suy diễn Thống kê (t-test, p-value, CI)
    # Lưu ý: truyền sigma2_hat từ dict 'res' hoặc thuộc tính của 'model' đều được
    coef_inference(X_demo, y_demo, model.beta_hat, model.sigma2_hat)

    # Minh họa tính VIF
    vif(X_demo)