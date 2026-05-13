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
        if X.ndim == 1: X = X.reshape(-1, 1)
        
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
    
    mse = rss / df_resid if df_resid > 0 else np.nan
    rmse = float(np.sqrt(mse)) if not np.isnan(mse) else np.nan
    mae = float(np.mean(np.abs(y - y_hat)))
    
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
        "mse":      mse,   
        "rmse":     rmse,  
        "mae":      mae,  
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
        print(f"  MSE (σ̂²)          : {mse:.6f}")
        print(f"  RMSE              : {rmse:.6f}")
        print(f"  MAE               : {mae:.6f}")
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

def _assert_close(a, b, tol=1e-8, msg=""):
    diff = np.max(np.abs(np.asarray(a) - np.asarray(b)))
    assert diff < tol, f"FAIL [{msg}]: max_diff={diff:.4e} (tol={tol:.0e})"

# Core OLS & Class OLSRegressor
def test_ols_fit_returns_correct_keys():
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([3, 5, 7], dtype=float)
    result = ols_fit(X, y)
    
    expected_keys = ("beta_hat", "sigma2_hat", "y_hat", "residuals", "X_design", "n", "k", "df")
    for key in expected_keys:
        assert key in result, f"FAIL: thiếu key '{key}'"
        
    assert result["n"] == 3, f"FAIL: n phải bằng 3, nhưng ra {result['n']}"
    assert result["k"] == 2, f"FAIL: k phải bằng 2, nhưng ra {result['k']}"
    assert result["df"] == 1, f"FAIL: df phải bằng 1, nhưng ra {result['df']}"
    print("test_ols_fit_returns_correct_keys PASSED")

def test_ols_simple_known_data():
    """Test 1: Simple Linear Regression (y = 1 + 2x)."""
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([3, 5, 7], dtype=float)
    model = OLSRegressor(fit_intercept=True).fit(X, y)
    _assert_close(model.beta_hat[0], 1.0, msg="Intercept phải bằng 1")
    _assert_close(model.beta_hat[1], 2.0, msg="Slope phải bằng 2")
    _assert_close(model.residuals, [0, 0, 0], msg="Dữ liệu hoàn hảo, residuals = 0")
    print("test_ols_simple_known_data PASSED")

def test_ols_multiple_regression_known_data():
    """Test 2: Multiple Regression (y = 10 + 2x1 - 5x2) với n=4 để df > 0."""
    X = np.array([
        [1, 0], 
        [2, 1], 
        [3, 0],
        [4, 1]   
    ], dtype=float)
    # Tính tay y = 10 + 2*x1 - 5*x2:
    # Dòng 1: 10 + 2(1) - 0 = 12
    # Dòng 2: 10 + 2(2) - 5 = 9
    # Dòng 3: 10 + 2(3) - 0 = 16
    # Dòng 4: 10 + 2(4) - 5 = 13 
    y = np.array([12, 9, 16, 13], dtype=float)
    
    model = OLSRegressor(fit_intercept=True).fit(X, y)
    _assert_close(model.beta_hat, [10.0, 2.0, -5.0], msg="Beta multiple regression sai")
    print("test_ols_multiple_regression_known_data PASSED")

def test_regressor_properties():
    """residuals và fitted_values phải nhất quán với y và y_hat (tĩnh)."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.1, 3.9, 6.1, 7.9])
    model = OLSRegressor().fit(X, y)
    y_hat = model.predict(X)

    _assert_close(model.fitted_values, y_hat, msg="fitted_values == predict(X)")
    _assert_close(model.residuals, y - y_hat, msg="residuals == y - ŷ")
    assert model.residuals.shape == (4,), "FAIL: shape residuals"
    print("test_regressor_properties PASSED")

def test_predict_before_fit_raises():
    """predict() phải raise RuntimeError nếu chưa gọi fit()."""
    model = OLSRegressor()
    try:
        model.predict(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert False, "FAIL: phải raise RuntimeError"
    except RuntimeError:
        print("test_predict_before_fit_raises PASSED")

def test_fit_intercept_false():
    """fit_intercept=False: không thêm cột 1, beta_hat có p phần tử."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = 2.0 * X.ravel()
    model = OLSRegressor(fit_intercept=False).fit(X, y)
    assert model.beta_hat.shape == (1,), "FAIL: shape kỳ vọng (1,)"
    _assert_close(model.beta_hat[0], 2.0, msg="slope=2")
    print("test_fit_intercept_false PASSED")

# Các hàm tính Sum of Squares (RSS, TSS, ESS)
def test_rss_known_values():
    y = np.array([1.0, 2.0, 3.0])
    _assert_close(compute_rss(y, y), 0.0, msg="RSS perfect fit")
    _assert_close(compute_rss(y, np.array([2.0, 2.0, 2.0])), 2.0, msg="RSS hand calc")
    print("test_rss_known_values PASSED")

def test_tss_known_values():
    _assert_close(compute_tss(np.array([1.0, 2.0, 3.0])), 2.0, msg="TSS variance")
    _assert_close(compute_tss(np.array([5.0, 5.0, 5.0])), 0.0, msg="TSS constant")
    print("test_tss_known_values PASSED")

def test_ess_known_values():
    y = np.array([1.0, 2.0, 3.0]) 
    _assert_close(compute_ess(y, y), 2.0, msg="ESS perfect fit")
    _assert_close(compute_ess(y, np.array([2.0, 2.0, 2.0])), 0.0, msg="ESS worst model")
    print("test_ess_known_values PASSED")

# Các hàm R2 và Metrics
def test_r2_known_values():
    y = np.array([1.0, 3.0])
    y_hat = np.array([1.5, 2.5])
    _assert_close(compute_r2(y, y_hat), 0.75, msg="R2 hand calc")
    _assert_close(compute_r2(y, y), 1.0, msg="R2 perfect")
    print("test_r2_known_values PASSED")

def test_r2_adj_and_metrics_hand_calc():
    """Test gộp R²_adj và model_metrics với đủ MSE, RMSE, MAE."""
    y = np.array([1.0, 2.0, 3.0, 4.0])            
    y_hat = np.array([1.5, 1.5, 3.5, 3.5])        
    
    _assert_close(compute_r2(y, y_hat), 0.8, msg="R2")
    _assert_close(compute_r2_adj(y, y_hat, p=1), 0.7, msg="R2_adj")
    
    m = model_metrics(y, y_hat, p=1, verbose=False)
    _assert_close(m["rss"], 1.0, msg="Metric RSS")
    _assert_close(m["tss"], 5.0, msg="Metric TSS")
    _assert_close(m["ess"], 4.0, msg="Metric ESS")
    _assert_close(m["mse"], 0.5, msg="Metric MSE")
    _assert_close(m["rmse"], np.sqrt(0.5), msg="Metric RMSE")
    _assert_close(m["mae"], 0.5, msg="Metric MAE")
    _assert_close(m["f_stat"], 8.0, msg="Metric F-Stat")
    print("test_r2_adj_and_metrics_hand_calc PASSED")

def test_metrics_second_case():
    """Test Case 2: Kiểm tra model_metrics với bộ dữ liệu chuẩn OLS. Đảm bảo định lý phân rã phương sai: TSS = ESS + RSS.
    """
    # Dữ liệu thực từ mô hình OLS: n=3, p=1
    y = np.array([2.0, 5.0, 5.0])          # mean(y) = 4.0
    y_hat = np.array([2.5, 4.0, 5.5])      # mean(y_hat) = 4.0
    
    # TSS = ||y - mean||^2 = (-2)^2 + 1^2 + 1^2 = 6.0
    # ESS = ||y_hat - mean||^2 = (-1.5)^2 + 0^2 + 1.5^2 = 2.25 + 2.25 = 4.5
    # Tính RSS = ||y - y_hat||^2 = (-0.5)^2 + 1^0^2 + (-0.5)^2 = 0.25 + 1.0 + 0.25 = 1.5
    # TSS (6.0) = ESS (4.5) + RSS (1.5)
    
    # Tính R² = ESS / TSS = 4.5 / 6.0 = 0.75
    _assert_close(compute_r2(y, y_hat), 0.75, msg="R2 case 2") 
    
    # Tính R²_adj = 1 - [(n-1)/(n-p-1)] * (1 - R²) = 1 - (2/1)*0.25 = 0.5
    _assert_close(compute_r2_adj(y, y_hat, p=1), 0.5, msg="R2_adj case 2")
    
    m = model_metrics(y, y_hat, p=1, verbose=False)
    _assert_close(m["tss"], 6.0, msg="TSS case 2")
    _assert_close(m["ess"], 4.5, msg="ESS case 2")
    _assert_close(m["rss"], 1.5, msg="RSS case 2")
    _assert_close(m["mse"], 1.5, msg="MSE case 2") # RSS(1.5) / df_resid(1)
    
    print("test_metrics_second_case PASSED")

# Edge Cases
def test_raises_on_errors():
    """Kiểm tra các ngoại lệ cơ bản."""
    # Đa cộng tuyến
    try: ols_fit(np.array([[1, 1], [2, 2]]), np.array([1, 2]))
    except ValueError: pass

    # TSS = 0
    try: compute_r2(np.array([4, 4]), np.array([1, 2]))
    except ValueError: pass

    # Thiếu DOF (n<=k)
    try: ols_fit(np.array([[1, 2], [3, 4]]), np.array([1, 2]))
    except ValueError: pass
    print("test_raises_on_errors PASSED")

def test_r2_adj_raises_insufficient_data():
    """compute_r2_adj() phải raise ValueError khi n <= p+1."""
    try:
        compute_r2_adj(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 2.9]), p=2)
        assert False, "FAIL: phải raise ValueError"
    except ValueError:
        print("test_r2_adj_raises_insufficient_data PASSED")

def test_verify_sklearn_static():
    X = np.array([[1.5, 2.1], [3.0, 1.2], [4.5, 5.0], [2.2, 3.3]], dtype=float)
    y = np.array([5.1, 8.2, 12.5, 7.3], dtype=float)
    result = verify_with_sklearn(X, y)
    assert result["passed"], "FAIL: OLS tĩnh không khớp Sklearn"
    print("test_verify_sklearn_static PASSED")

def run_all_tests():
    print("        CHẠY TOÀN BỘ UNIT TESTS (STATIC DATA)")
    tests = [
        test_ols_fit_returns_correct_keys,
        test_ols_simple_known_data,
        test_ols_multiple_regression_known_data,
        test_regressor_properties,
        test_predict_before_fit_raises,
        test_fit_intercept_false,
        test_rss_known_values,
        test_tss_known_values,
        test_ess_known_values,
        test_r2_known_values,
        test_r2_adj_and_metrics_hand_calc,
        test_metrics_second_case,
        test_raises_on_errors,
        test_r2_adj_raises_insufficient_data,
        test_verify_sklearn_static
    ]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR in {test_fn.__name__}: {e}")
            failed += 1
    print('\n')
    print(f"  Results: {passed} passed, {failed} failed / {len(tests)} total")
    return failed == 0


if __name__ == "__main__":
    print("OLS IMPLEMENTATION DEMO & UNIT TESTS")

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
    run_all_tests()