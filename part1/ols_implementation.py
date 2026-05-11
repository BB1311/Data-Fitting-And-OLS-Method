import numpy as np
from scipy import stats

def _ols_core_solver(X_design: np.ndarray, y: np.ndarray) -> dict:
    """
    Thực hiện giải phương trình Normal Equations trên ma trận đã chuẩn bị.
    Đây là nơi chứa công thức toán học cốt lõi.
    """
    n, k = X_design.shape
    df = n - k

    # Giải β̂ = (XᵀX)⁻¹ Xᵀy 
    XtX = X_design.T @ X_design
    Xty = X_design.T @ y
    
    try:
        beta_hat = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        raise ValueError("Ma trận không đủ hạng (đa cộng tuyến hoàn hảo).")

    y_hat = X_design @ beta_hat
    residuals = y - y_hat
    rss = float(residuals @ residuals)
    sigma2_hat = rss / (n - k) if n > k else 0.0

    return {
        "beta_hat": beta_hat,
        "sigma2_hat": sigma2_hat,
        "y_hat": y_hat,
        "residuals": residuals,
        "X_design": X_design,
        "n": n,    # Số lượng quan sát
        "k": k,    # Số lượng tham số (p + 1)
        "df": df   # Bậc tự do của phần dư (n - k)
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
    
def compute_rss(y, y_hat):
    residuals = np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float)
    return float(residuals @ residuals)

def compute_tss(y):
    y = np.asarray(y, dtype=float)
    return float(np.sum((y - y.mean()) ** 2))

def compute_ess(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    y_bar = y.mean()
    diff = y_hat - y_bar
    return float(diff @ diff)

def compute_r2(y, y_hat):
    rss = compute_rss(y, y_hat)
    tss = compute_tss(y)
    if tss == 0:
        raise ValueError("TSS = 0")
    return 1.0 - rss / tss

def compute_r2_adj(y, y_hat, p):
    n = len(y)
    if n <= p + 1:
        raise ValueError("n phải lớn hơn p+1.")
    r2 = compute_r2(y, y_hat)
    return 1.0 - (n - 1) / (n - p - 1) * (1.0 - r2)

def model_metrics(y, y_hat, p, verbose=True):
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    n = len(y)

    rss = compute_rss(y, y_hat)
    tss = compute_tss(y)
    ess = compute_ess(y, y_hat)
    if tss == 0:
        raise ValueError("TSS = 0")
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
        "rss": rss, "tss": tss, "ess": ess,
        "r2": r2, "r2_adj": r2_adj,
        "f_stat": f_stat, "f_pvalue": f_pvalue,
        "n": n, "p": p,
    }

    if verbose:
        print("        MODEL METRICS SUMMARY")
        print(f"  n={n}, p={p}, RSS={rss:.6f}, TSS={tss:.6f}, ESS={ess:.6f}")
        print(f"  R²={r2:.6f}, R²_adj={r2_adj:.6f}")
        print(f"  F={f_stat:.4f} (df1={df_model}, df2={df_resid}), p={f_pvalue:.6e}")

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

    # --- So sánh sai số ---
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

# --- ols_fit ---
def test_ols_fit_returns_correct_keys():
    np.random.seed(0)
    X = np.random.randn(30, 2)
    y = np.random.randn(30)
    result = ols_fit(X, y)
    for key in ("beta_hat", "sigma2_hat", "y_hat", "residuals", "X_design"):
        assert key in result, f"FAIL: thiếu key '{key}'"
    assert result["beta_hat"].shape == (3,)
    assert result["y_hat"].shape   == (30,)
    print("test_ols_fit_returns_correct_keys PASSED")

def test_ols_fit_known_solution():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 1.0 + 2.0 * X.ravel()
    res = ols_fit(X, y)
    _assert_close(res["beta_hat"][0], 1.0, tol=1e-8,  msg="intercept")
    _assert_close(res["beta_hat"][1], 2.0, tol=1e-8,  msg="slope")
    _assert_close(np.sum(res["residuals"] ** 2), 0.0, tol=1e-10, msg="RSS≈0")
    print("test_ols_fit_known_solution PASSED")

# --- OLSRegressor ---
def test_ols_simple_regression():
    np.random.seed(0)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2.0 + 3.0 * X.ravel()
    model = OLSRegressor().fit(X, y)
    _assert_close(model.beta_hat[0], 2.0, tol=1e-8, msg="intercept")
    _assert_close(model.beta_hat[1], 3.0, tol=1e-8, msg="slope")
    print("test_ols_simple_regression PASSED")

def test_ols_multiple_regression():
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 2)
    beta_true = np.array([1.0, 2.0, -0.5])
    y = beta_true[0] + X @ beta_true[1:] + 0.1 * np.random.randn(n)
    model = OLSRegressor().fit(X, y)
    _assert_close(model.beta_hat, beta_true, tol=0.1, msg="betas")
    print("test_ols_multiple_regression PASSED")

# --- compute_rss ---
def test_rss_perfect_fit_is_zero():
    """RSS = 0 khi ŷ = y (perfect fit)."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    _assert_close(compute_rss(y, y), 0.0, tol=1e-14, msg="RSS perfect=0")
    print("test_rss_perfect_fit_is_zero PASSED")

def test_rss_known_value():
    """RSS tay: y=[1,2,3], ŷ=[2,2,2] → RSS = (1-2)²+(2-2)²+(3-2)² = 2."""
    y     = np.array([1.0, 2.0, 3.0])
    y_hat = np.array([2.0, 2.0, 2.0])
    _assert_close(compute_rss(y, y_hat), 2.0, tol=1e-12, msg="RSS=2")
    print("test_rss_known_value PASSED")

# --- compute_r2 ---
def test_r2_perfect_fit():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 2.0 + 3.0 * X.ravel()
    y_hat = OLSRegressor().fit(X, y).predict(X)
    _assert_close(compute_r2(y, y_hat), 1.0, tol=1e-10, msg="R²=1")
    print("test_r2_perfect_fit PASSED")

def test_r2_known_value():
    y     = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_hat = np.array([2.0, 2.0, 3.0, 4.0, 4.0])
    _assert_close(compute_r2(y, y_hat), 0.8, tol=1e-12, msg="R²=0.8")
    print("test_r2_known_value PASSED")

# --- compute_r2_adj ---
def test_r2_adj_penalizes_extra_variables():
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 1)
    y = 2.0 * X.ravel() + np.random.randn(n) * 0.5

    model1 = OLSRegressor().fit(X, y)
    y_hat1 = model1.predict(X)
    r2_1 = compute_r2(y, y_hat1)
    r2_adj_1 = compute_r2_adj(y, y_hat1, p=1)

    X_noise = np.random.randn(n, 5)
    X_extended = np.hstack([X, X_noise])

    model2 = OLSRegressor().fit(X_extended, y)
    y_hat2 = model2.predict(X_extended)
    r2_2 = compute_r2(y, y_hat2)
    r2_adj_2 = compute_r2_adj(y, y_hat2, p=6)

    assert r2_2 >= r2_1, "FAIL: R² phải tăng hoặc giữ nguyên khi thêm biến."
    assert r2_adj_2 < r2_adj_1, f"FAIL: R²_adj không phạt biến nhiễu! (Mới {r2_adj_2:.4f} >= Cũ {r2_adj_1:.4f})"
    print("test_r2_adj_penalizes_extra_variables PASSED")

def test_r2_adj_known_value():
    y     = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_hat = np.array([2.0, 2.0, 3.0, 4.0, 4.0])
    expected = 1.0 - (4 / 3) * 0.2
    _assert_close(compute_r2_adj(y, y_hat, p=1), expected, tol=1e-12, msg="R²_adj")
    print("test_r2_adj_known_value PASSED")

# --- model_metrics ---
def test_model_metrics_keys():
    """model_metrics() phải trả về dict đủ 9 key."""
    np.random.seed(0)
    X = np.random.randn(50, 2)
    y = np.random.randn(50)
    y_hat = OLSRegressor().fit(X, y).predict(X)
    m = model_metrics(y, y_hat, p=2, verbose=False)
    for key in ("rss", "tss", "ess", "r2", "r2_adj", "f_stat", "f_pvalue", "n", "p"):
        assert key in m, f"FAIL: thiếu key '{key}'"
    print("test_model_metrics_keys PASSED")

def test_model_metrics_identity():
    np.random.seed(5)
    X = np.random.randn(80, 3)
    y = X @ np.array([1.0, -0.5, 2.0]) + np.random.randn(80)
    y_hat = OLSRegressor().fit(X, y).predict(X)
    m = model_metrics(y, y_hat, p=3, verbose=False)

    _assert_close(m["ess"] + m["rss"], m["tss"], tol=1e-8, msg="ESS+RSS=TSS")
    _assert_close(m["r2"], 1.0 - m["rss"] / m["tss"], tol=1e-12, msg="R²=1-RSS/TSS")
    assert m["r2_adj"] <= m["r2"],       "FAIL: r2_adj > r2"
    assert 0.0 <= m["f_pvalue"] <= 1.0, "FAIL: f_pvalue ngoài [0,1]"
    print("test_model_metrics_identity PASSED")

def test_model_metrics_f_significant():
    """
    Khi mô hình thực sự có ý nghĩa (signal mạnh), F p-value phải rất nhỏ (< 0.001).
    """
    np.random.seed(9)
    n = 200
    X = np.random.randn(n, 3)
    y = 5.0 + X @ np.array([3.0, -2.0, 1.5]) + 0.1 * np.random.randn(n)
    y_hat = OLSRegressor().fit(X, y).predict(X)
    m = model_metrics(y, y_hat, p=3, verbose=False)
    assert m["f_pvalue"] < 0.001, f"FAIL: f_pvalue={m['f_pvalue']:.4f} không nhỏ như kỳ vọng"
    print("test_model_metrics_f_significant PASSED")

# --- misc ---
def test_sigma2_unbiased():
    """E[σ̂²] ≈ σ² = 4 (Monte Carlo 1000 lần, tol=0.3)."""
    np.random.seed(99)
    n, p, sigma2_true = 50, 3, 4.0
    sigma = np.sqrt(sigma2_true)
    estimates = []
    for _ in range(1000):
        X = np.random.randn(n, p)
        y = X @ np.ones(p) + sigma * np.random.randn(n)
        estimates.append(OLSRegressor().fit(X, y).sigma2_hat)
    _assert_close(np.mean(estimates), sigma2_true, tol=0.3, msg="E[σ̂²]≈σ²")
    print(f"test_sigma2_unbiased PASSED  (E[σ̂²]={np.mean(estimates):.4f})")

def test_raises_on_collinear():
    """ols_fit() phải raise ValueError khi X có 2 cột giống nhau."""
    np.random.seed(17)
    col = np.arange(1, 6, dtype=float)
    X   = np.column_stack([col, col])   # rank = 1 < k = 3
    y   = np.random.randn(5)
    try:
        ols_fit(X, y)
        assert False, "FAIL: phải raise ValueError nhưng không raise"
    except ValueError:
        print("test_raises_on_collinear PASSED")

def test_r2_raises_on_constant_y():
    """compute_r2() phải raise ValueError khi y hằng số (TSS = 0)."""
    np.random.seed(0)
    X = np.random.randn(20, 2)
    y = np.ones(20) * 3.0   # TSS = 0
    try:
        compute_r2(y, y)
        assert False, "FAIL: phải raise ValueError nhưng không raise"
    except ValueError:
        print("test_r2_raises_on_constant_y PASSED")

def test_prediction_shape():
    np.random.seed(6)
    X_train = np.random.randn(80, 4)
    y_train = np.random.randn(80)
    y_pred = OLSRegressor().fit(X_train, y_train).predict(np.random.randn(20, 4))
    assert y_pred.shape == (20,)
    print("test_prediction_shape PASSED")

def test_verify_sklearn_match():
    np.random.seed(123)
    X = np.random.randn(200, 5)
    y = X @ np.arange(1, 6, dtype=float) + 0.5 * np.random.randn(200)
    result = verify_with_sklearn(X, y)
    assert result["passed"], "FAIL: OLS scratch không khớp sklearn!"
    print("test_verify_sklearn_match PASSED")


def run_all_tests():
    print("              CHẠY TOÀN BỘ UNIT TESTS")
    tests = [
        # ols_fit
        test_ols_fit_returns_correct_keys,
        test_ols_fit_known_solution,
        # OLSRegressor
        test_ols_simple_regression,
        test_ols_multiple_regression,
        # compute_rss
        test_rss_perfect_fit_is_zero,
        test_rss_known_value,
        # compute_r2 
        test_r2_perfect_fit,
        test_r2_known_value,
        # compute_r2_adj 
        test_r2_adj_penalizes_extra_variables,
        test_r2_adj_known_value,
        # model_metrics 
        test_model_metrics_keys,
        test_model_metrics_identity,
        test_model_metrics_f_significant,
        # misc
        test_sigma2_unbiased,
        test_prediction_shape,
        test_verify_sklearn_match,
        # catch error
        test_raises_on_collinear,
        test_r2_raises_on_constant_y
    ]
    passed = failed = 0
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
    print(f"\n  Results: {passed} passed, {failed} failed / {len(tests)} total")
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