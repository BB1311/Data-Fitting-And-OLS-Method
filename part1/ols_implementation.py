import numpy as np
from scipy import stats

def ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n = X.shape[0]
    p = X.shape[1] if X.ndim == 2 else 1

    # Thêm cột 1
    ones = np.ones((n, 1))
    X_design = np.hstack([ones, X.reshape(n, -1)])
    k = X_design.shape[1]

    if np.linalg.matrix_rank(X_design) < k:
        raise ValueError("Ma trận thiết kế X không có hạng đầy đủ (đa cộng tuyến hoàn hảo). Nghiệm OLS không tồn tại duy nhất.")

    XtX = X_design.T @ X_design
    Xty = X_design.T @ y
    beta_hat = np.linalg.solve(XtX, Xty)

    # Giá trị fit & Phần dư
    y_hat = X_design @ beta_hat
    residuals = y - y_hat
    rss = float(residuals @ residuals)
    df = n - k
    sigma2_hat = rss / df

    return {
        "beta_hat": beta_hat,
        "sigma2_hat": sigma2_hat,
        "y_hat": y_hat,
        "residuals": residuals,
        "X_design": X_design,
    }

class OLSRegressor:
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.beta_hat: np.ndarray | None = None
        self.sigma2_hat: float | None = None
        self.n: int | None = None
        self.p: int | None = None
        self.X_fit: np.ndarray | None = None
        self.y_fit: np.ndarray | None = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Thêm cột 1 vào đầu ma trận X nếu fit_intercept=True."""
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            return np.hstack([ones, X])
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "OLSRegressor":
        """
        Gọi hàm ols_fit độc lập để lấy kết quả và lưu vào class.
        """
        # Tạm thời giả định fit_intercept=True theo yêu cầu bài toán cơ bản
        res = ols_fit(X, y)
        
        # Lưu kết quả vào self để dùng cho predict
        self.beta_hat = res["beta_hat"]
        self.sigma2_hat = res["sigma2_hat"]
        self.X_fit = res["X_design"]
        self.y_fit = y
        self.n = res["X_design"].shape[0]
        self.p = res["X_design"].shape[1] - 1
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        X_design = self._add_intercept(X)
        return X_design @ self.beta_hat
    
    # Kiểm tra đã phù hợp chưa
    def _check_fitted(self):
        if self.beta_hat is None:
            raise RuntimeError("Gọi fit() trước khi sử dụng mô hình.")

def compute_rss(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Residual Sum of Squares (RSS)"""
    residuals = np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float)
    return float(residuals @ residuals)


def compute_tss(y: np.ndarray) -> float:
    """Total Sum of Squares (TSS)"""
    y = np.asarray(y, dtype=float)
    return float(np.sum((y - y.mean()) ** 2))


def compute_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Hệ số xác định R² (Dùng khi gọi độc lập)"""
    rss = compute_rss(y, y_hat)
    tss = compute_tss(y)
    if tss == 0:
        raise ValueError("TSS = 0: biến mục tiêu y không có biến thiên.")
    return 1.0 - rss / tss


def compute_r2_adj(y: np.ndarray, y_hat: np.ndarray, p: int) -> float:
    """R² hiệu chỉnh (Dùng khi gọi độc lập)"""
    n = len(y)
    if n <= p + 1:
        raise ValueError("n phải lớn hơn p+1 để tính R² hiệu chỉnh.")
    r2 = compute_r2(y, y_hat)
    return 1.0 - (n - 1) / (n - p - 1) * (1.0 - r2)


def model_metrics(
    y: np.ndarray,
    y_hat: np.ndarray,
    p: int,
    verbose: bool = True,
) -> dict:
    """Tính tổng hợp các chỉ số đánh giá mô hình."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    n = len(y)

    rss = compute_rss(y, y_hat)
    tss = compute_tss(y)
    ess = tss - rss                    
    
    if tss == 0:
        raise ValueError("TSS = 0: biến mục tiêu y không có biến thiên.")
        
    # Tính trực tiếp R² và R²_adj từ rss và tss đã có 
    r2 = 1.0 - (rss / tss)
    r2_adj = 1.0 - (n - 1) / (n - p - 1) * (1.0 - r2)

    # Kiểm định F tổng thể
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

# Kiểm chứng kết quả với sklearn và numpy

def verify_with_sklearn(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Kiểm chứng kết quả OLS tự cài đặt với sklearn.LinearRegression.
    results: dict chứa beta_hat từ cả hai phương pháp và sự chênh lệch.
    """
    from sklearn.linear_model import LinearRegression

    # OLS tự cài đặt
    model_scratch = OLSRegressor(fit_intercept=True)
    model_scratch.fit(X, y)
    beta_scratch = model_scratch.beta_hat # (p+1,): [intercept, β₁, ..., βₚ]
    y_hat_scratch = model_scratch.predict(X)

    # sklearn
    model_sk = LinearRegression(fit_intercept=True)
    model_sk.fit(X, y)
    beta_sklearn = np.concatenate([[model_sk.intercept_], model_sk.coef_])
    y_hat_sklearn = model_sk.predict(X)

    # numpy lstsq
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    beta_numpy, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat_numpy = X_design @ beta_numpy

    # So sánh
    max_diff_sk    = np.max(np.abs(beta_scratch - beta_sklearn))
    max_diff_numpy = np.max(np.abs(beta_scratch - beta_numpy))

    print("         VERIFICATION: OLS Scratch vs sklearn vs numpy")
    header = f"{'Coef':>12} {'Scratch':>14} {'sklearn':>14} {'numpy lstsq':>14}"
    print(header)
    labels = ["intercept"] + [f"β_{i}" for i in range(1, len(beta_scratch))]
    for label, b_s, b_sk, b_np in zip(labels, beta_scratch, beta_sklearn, beta_numpy):
        print(f"{label:>12}  {b_s:>13.6f}  {b_sk:>13.6f}  {b_np:>13.6f}")
    print(f"  Max |β_scratch - β_sklearn|  = {max_diff_sk:.2e}")
    print(f"  Max |β_scratch - β_numpy  |  = {max_diff_numpy:.2e}")
    tol = 1e-8
    ok = max_diff_sk < tol and max_diff_numpy < tol
    print(f"  Kết quả khớp (tol={tol:.0e})    : {'PASS' if ok else 'FAIL'}")

    return {
        "beta_scratch":  beta_scratch,
        "beta_sklearn":  beta_sklearn,
        "beta_numpy":    beta_numpy,
        "y_hat_scratch": y_hat_scratch,
        "y_hat_sklearn": y_hat_sklearn,
        "max_diff_sk":   max_diff_sk,
        "max_diff_numpy": max_diff_numpy,
        "passed":        ok,
    }

def _assert_close(a, b, tol=1e-8, msg=""):
    """Kiểm tra hai giá trị/ mảng gần bằng nhau."""
    diff = np.max(np.abs(np.asarray(a) - np.asarray(b)))
    assert diff < tol, f"FAIL [{msg}]: max_diff={diff:.4e} (tol={tol:.0e})"


def test_ols_fit_returns_correct_keys():
    """Test 0a: ols_fit() trả về dict đầy đủ các key cần thiết."""
    X = np.random.randn(30, 2)
    y = np.random.randn(30)
    result = ols_fit(X, y)
    for key in ("beta_hat", "sigma2_hat", "y_hat", "residuals", "X_design"):
        assert key in result, f"FAIL: thiếu key '{key}'"
    assert result["beta_hat"].shape == (3,), "FAIL: shape beta_hat sai"
    assert result["y_hat"].shape   == (30,), "FAIL: shape y_hat sai"
    print("test_ols_fit_returns_correct_keys PASSED")


def test_ols_fit_known_solution():
    """
    Test 0b: ols_fit() với dữ liệu y = 1 + 2x (không nhiễu).
    Kỳ vọng: β̂₀ = 1, β̂₁ = 2; RSS ≈ 0.
    """
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 1.0 + 2.0 * X.ravel()

    res = ols_fit(X, y)
    _assert_close(res["beta_hat"][0], 1.0, tol=1e-8,  msg="Test0b intercept")
    _assert_close(res["beta_hat"][1], 2.0, tol=1e-8,  msg="Test0b slope")
    _assert_close(np.sum(res["residuals"] ** 2), 0.0, tol=1e-10, msg="Test0b RSS≈0")
    print("test_ols_fit_known_solution PASSED")


def test_ols_simple_regression():
    """
    Test 1: Hồi quy đơn giản y = 2 + 3x (không nhiễu).
    Kỳ vọng: β̂₀ ≈ 2, β̂₁ ≈ 3.
    """
    np.random.seed(0)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2.0 + 3.0 * X.ravel()

    model = OLSRegressor(fit_intercept=True)
    model.fit(X, y)

    _assert_close(model.beta_hat[0], 2.0, tol=1e-8, msg="Test1 intercept")
    _assert_close(model.beta_hat[1], 3.0, tol=1e-8, msg="Test1 slope")
    print("test_ols_simple_regression PASSED")


def test_ols_multiple_regression():
    """
    Test 2: Hồi quy bội y = 1 + 2x₁ - 0.5x₂ + noise.
    Dùng n=500 để ước lượng đủ chính xác.
    Kiểm chứng β̂ gần với β_true (tol=0.1).
    """
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 2)
    beta_true = np.array([1.0, 2.0, -0.5])
    y = beta_true[0] + X @ beta_true[1:] + 0.1 * np.random.randn(n)

    model = OLSRegressor(fit_intercept=True)
    model.fit(X, y)

    _assert_close(model.beta_hat, beta_true, tol=0.1, msg="Test2 betas")
    print("test_ols_multiple_regression PASSED")


def test_rss_is_nonnegative():
    """
    Test 3: RSS ≥ 0 với mọi dữ liệu.
    """
    np.random.seed(7)
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    model = OLSRegressor().fit(X, y)
    y_hat = model.predict(X)
    rss = compute_rss(y, y_hat)
    assert rss >= 0, f"FAIL Test3: RSS = {rss}"
    print("test_rss_is_nonnegative PASSED")


def test_r2_perfect_fit():
    """
    Test 4: R² = 1 khi dữ liệu không có nhiễu (perfect fit).
    """
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 2.0 + 3.0 * X.ravel()

    model = OLSRegressor().fit(X, y)
    y_hat = model.predict(X)
    r2 = compute_r2(y, y_hat)
    _assert_close(r2, 1.0, tol=1e-10, msg="Test4 R²=1")
    print("test_r2_perfect_fit PASSED")


def test_r2_adj_penalizes_extra_variables():
    """
    Test 5: R²_adj < R² khi p > 0 và dữ liệu có nhiễu.
    (R² luôn tăng khi thêm biến, R²_adj phạt thêm biến không cần thiết.)
    """
    np.random.seed(1)
    n = 100
    X = np.random.randn(n, 3)
    y = np.random.randn(n) # y không liên quan X -> R² thấp

    model = OLSRegressor().fit(X, y)
    y_hat = model.predict(X)
    r2 = compute_r2(y, y_hat)
    r2_adj = compute_r2_adj(y, y_hat, p=3)

    assert r2_adj <= r2, f"FAIL Test5: r2_adj={r2_adj:.4f} > r2={r2:.4f}"
    print("test_r2_adj_penalizes_extra_variables PASSED")


def test_sigma2_unbiased():
    """
    Test 6: Ước lượng σ̂² không chệch – Monte Carlo.
    E[σ̂²] ≈ σ² = 4 (tol=0.3 với 1000 lần lặp).
    """
    np.random.seed(99)
    n, p, sigma2_true = 50, 3, 4.0
    sigma = np.sqrt(sigma2_true)
    sigma2_estimates = []

    for _ in range(1000):
        X = np.random.randn(n, p)
        y = X @ np.ones(p) + sigma * np.random.randn(n)
        model = OLSRegressor().fit(X, y)
        sigma2_estimates.append(model.sigma2_hat)

    mean_estimate = np.mean(sigma2_estimates)
    _assert_close(mean_estimate, sigma2_true, tol=0.3, msg="Test6 E[σ̂²]≈σ²")
    print(f"test_sigma2_unbiased PASSED  (E[σ̂²]={mean_estimate:.4f}, σ²={sigma2_true})")


def test_prediction_shape():
    """
    Test 7: predict() trả về đúng shape (m,).
    """
    X_train = np.random.randn(80, 4)
    y_train = np.random.randn(80)
    X_test  = np.random.randn(20, 4)

    model = OLSRegressor().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == (20,), f"FAIL Test7 shape: {y_pred.shape}"
    print("test_prediction_shape PASSED")


def test_verify_sklearn_match():
    """
    Test 8: Kết quả OLS scratch phải khớp với sklearn (tol=1e-8).
    """
    np.random.seed(123)
    X = np.random.randn(200, 5)
    y = X @ np.arange(1, 6, dtype=float) + 0.5 * np.random.randn(200)
    result = verify_with_sklearn(X, y)
    assert result["passed"], "FAIL Test8: OLS scratch không khớp sklearn!"
    print("test_verify_sklearn_match PASSED")


def run_all_tests():
    """Chạy toàn bộ unit tests."""
    print("               CHẠY TOÀN BỘ UNIT TESTS")
    tests = [
        test_ols_fit_returns_correct_keys,
        test_ols_fit_known_solution,
        test_ols_simple_regression,
        test_ols_multiple_regression,
        test_rss_is_nonnegative,
        test_r2_perfect_fit,
        test_r2_adj_penalizes_extra_variables,
        test_sigma2_unbiased,
        test_prediction_shape,
        test_verify_sklearn_match,
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

    print(f"  Results: {passed} passed, {failed} failed / {len(tests)} total")
    return failed == 0

#demo
if __name__ == "__main__":
    print("OLS IMPLEMENTATION DEMO & UNIT TESTS")

    np.random.seed(42)
    n = 150
    X_demo = np.random.randn(n, 3)
    beta_true = np.array([5.0, 2.0, -1.5, 0.8])   # intercept + 3 coefs
    y_demo = beta_true[0] + X_demo @ beta_true[1:] + np.random.randn(n)

    print(f"\n[Demo] Dữ liệu giả lập: n={n}, p=3")
    print(f"       β_true = {beta_true}")

    # Demo hàm ols_fit độc lập
    print("\nKết quả từ hàm ols_fit()")
    res = ols_fit(X_demo, y_demo)
    print(f"  β̂  = {res['beta_hat']}")
    print(f"  σ̂² = {res['sigma2_hat']:.4f}")
    print(f"  RSS = {np.sum(res['residuals']**2):.4f}")

    # Fit model
    model = OLSRegressor(fit_intercept=True)
    model.fit(X_demo, y_demo)
    y_hat_demo = model.predict(X_demo)

    print(f"\n[OLS] β̂ ước lượng = {model.beta_hat}")
    print(f"      σ̂²            = {model.sigma2_hat:.4f}")

    # Metrics
    print()
    metrics = model_metrics(y_demo, y_hat_demo, p=3, verbose=True)

    # Kiểm chứng
    verify_with_sklearn(X_demo, y_demo)

    # Unit tests
    run_all_tests()