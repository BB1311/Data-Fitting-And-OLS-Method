import numpy as np
from part1.ols_implementation import OLSRegressor, compute_r2
from part1.ridge_lasso import ridge_fit, lasso_fit

# HÀM TIỆN ÍCH
def _mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def _rmse(y_true, y_pred):
    return float(np.sqrt(_mse(y_true, y_pred)))

def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

# HÀM CHÍNH: k-FOLD CROSS-VALIDATION
def kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    model: str = "ols",
    lam: float = 1.0,
    random_state: int = 42,
    return_indices: bool = False,
) -> dict:
    """
    k-Fold Cross-Validation từ scratch.

    Chia dữ liệu thành k fold bằng nhau. Mỗi vòng lặp i:
        - Tập train : k-1 fold còn lại
        - Tập validate: fold thứ i
    Lặp k lần, mỗi lần đổi fold validate.

    CV score theo công thức đề bài:
        CV(k) = (1/k) Σᵢ MSEᵢ

    Parameters
    ----------
    X            : np.ndarray, shape (n, p) — chưa có cột bias
    y            : np.ndarray, shape (n,)
    k            : int — số fold (mặc định 5)
    model        : str — "ols", "ridge", hoặc "lasso"
    lam          : float — λ cho Ridge/Lasso (bỏ qua nếu model="ols")
    random_state : int — seed để reproducible

    Returns
    -------
    dict:
        cv_mse   : float — CV(k) chính theo công thức đề bài
        cv_rmse  : float — căn bậc hai của cv_mse (RMSE tổng thể)
        cv_mae   : float — trung bình MAE qua k fold
        cv_r2    : float — trung bình R² qua k fold
        std_fold_mse  : float — độ lệch chuẩn của MSE giữa các fold (ddof=1)
        std_fold_rmse : float — độ lệch chuẩn của RMSE giữa các fold (ddof=1)
        std_fold_mae  : float — độ lệch chuẩn của MAE giữa các fold
        std_fold_r2   : float — độ lệch chuẩn của R² giữa các fold
        se_cv_mse     : float — sai số chuẩn của CV MSE (= std_mse / sqrt(k))
        fold_mse, fold_rmse, fold_mae, fold_r2: list — giá trị từng fold
        k, model, lam
    """
    assert model in ("ols", "ridge", "lasso"), \
        "model phải là 'ols', 'ridge' hoặc 'lasso'"
    assert k >= 2, "k phải >= 2"

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    assert n >= k, f"Số mẫu ({n}) phải >= k ({k})"

    # Shuffle chỉ số
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n)

    # Chia k fold — phân bổ phần dư vào các fold đầu nếu n % k != 0
    fold_sizes = np.full(k, n // k)
    fold_sizes[: n % k] += 1
    boundaries = np.concatenate([[0], np.cumsum(fold_sizes)])

    fold_mse, fold_rmse, fold_mae, fold_r2 = [], [], [], []

    val_indices_list = []
    for i in range(k):
        val_idx   = indices[boundaries[i] : boundaries[i + 1]]
        val_indices_list.append(val_idx)
        train_idx = np.concatenate([indices[: boundaries[i]],
                                    indices[boundaries[i + 1] :]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

        if model == "ols":
            reg    = OLSRegressor(fit_intercept=True).fit(X_train, y_train)
            y_pred = reg.predict(X_val)
        elif model == "ridge":
            res   = ridge_fit(X_train, y_train, lam=lam)
            beta  = res["beta_hat"]
            y_pred = np.column_stack([np.ones(len(X_val)), X_val]) @ beta
        else:  # lasso
            res   = lasso_fit(X_train, y_train, lam=lam)
            beta  = res["beta_hat"]
            y_pred = np.column_stack([np.ones(len(X_val)), X_val]) @ beta

        fold_mse.append(_mse(y_val, y_pred))
        fold_rmse.append(_rmse(y_val, y_pred))
        fold_mae.append(_mae(y_val, y_pred))
        if len(y_val) < 2:
            r2 = np.nan  # không xác định được R² cho 1 điểm
        else:
            r2 = compute_r2(y_val, y_pred)
        fold_r2.append(r2)

    # Tính trung bình và độ lệch chuẩn
    cv_mse = float(np.mean(fold_mse))
    cv_rmse = float(np.sqrt(cv_mse))
    cv_mae = float(np.mean(fold_mae))
    if np.all(np.isnan(fold_r2)):
        cv_r2 = np.nan
    else:
        cv_r2 = float(np.nanmean(fold_r2))

    std_mse = float(np.std(fold_mse, ddof=1))
    std_rmse = float(np.std(fold_rmse, ddof=1))
    std_mae = float(np.std(fold_mae, ddof=1))
    std_r2 = float(np.std(fold_r2, ddof=1))

    se_cv_mse = std_mse / np.sqrt(k)  # standard error của CV MSE

    result = {
         "cv_mse"       : cv_mse,
        "cv_rmse"      : cv_rmse,
        "cv_mae"       : cv_mae,
        "cv_r2"        : cv_r2,
        "std_fold_mse" : std_mse,
        "std_fold_rmse": std_rmse,
        "std_fold_mae" : std_mae,
        "std_fold_r2"  : std_r2,
        "se_cv_mse"    : se_cv_mse,
        "fold_mse"     : fold_mse,
        "fold_rmse"    : fold_rmse,
        "fold_mae"     : fold_mae,
        "fold_r2"      : fold_r2,
        "k"            : k,
        "model"        : model,
        "lam"          : lam,
    }
    if return_indices:
        result["val_indices"] = val_indices_list
    return result

def compare_models_cv(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    lam_grid: list = None,
    random_state: int = 42,
) -> dict:
    """
    So sánh OLS, Ridge và Lasso với nhiều giá trị λ qua k-fold CV.
    Trả về λ tốt nhất cho Ridge và Lasso (CV MSE thấp nhất).

    Parameters
    ----------
    lam_grid : list of float — danh sách λ cần thử (mặc định log-spaced)

    Returns
    -------
    dict:
        ols_result      : dict
        ridge_results   : list of dict — Ridge cho từng λ
        lasso_results   : list of dict — Lasso cho từng λ
        lam_grid        : list
        best_lam_ridge  : float
        best_lam_lasso  : float
        best_ridge      : dict
        best_lasso      : dict
    """
    if lam_grid is None:
        lam_grid = list(np.logspace(-3, 4, 30))

    ols_result = kfold_cv(X, y, k=k, model="ols", random_state=random_state)

    ridge_results = [
        kfold_cv(X, y, k=k, model="ridge", lam=lam, random_state=random_state)
        for lam in lam_grid
    ]
    lasso_results = [
        kfold_cv(X, y, k=k, model="lasso", lam=lam, random_state=random_state)
        for lam in lam_grid
    ]

    best_ridge_idx = int(np.argmin([r["cv_mse"] for r in ridge_results]))
    best_lasso_idx = int(np.argmin([r["cv_mse"] for r in lasso_results]))

    return {
        "ols_result"    : ols_result,
        "ridge_results" : ridge_results,
        "lasso_results" : lasso_results,
        "lam_grid"      : lam_grid,
        "best_lam_ridge": lam_grid[best_ridge_idx],
        "best_lam_lasso": lam_grid[best_lasso_idx],
        "best_ridge"    : ridge_results[best_ridge_idx],
        "best_lasso"    : lasso_results[best_lasso_idx],
    }

# DEMO
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n, p = 500, 10
    X_d  = rng.standard_normal((n, p))
    y_d  = 3 + X_d @ rng.standard_normal(p) + 1.5 * rng.standard_normal(n)

    result = compare_models_cv(X_d, y_d, k=5, random_state=42)

    ols   = result["ols_result"]
    ridge = result["best_ridge"]
    lasso = result["best_lasso"]

    print(f"\n{'Model':<22} {'CV MSE':>10} {'CV RMSE':>10} {'CV R²':>8}")
    print("-" * 54)
    print(f"{'OLS':<22} {ols['cv_mse']:>10.4f} {ols['cv_rmse']:>10.4f} {ols['cv_r2']:>8.4f}")
    print(f"{'Ridge (best λ)':<22} {ridge['cv_mse']:>10.4f} {ridge['cv_rmse']:>10.4f} {ridge['cv_r2']:>8.4f}")
    print(f"{'Lasso (best λ)':<22} {lasso['cv_mse']:>10.4f} {lasso['cv_rmse']:>10.4f} {lasso['cv_r2']:>8.4f}")
    print(f"\nBest λ Ridge: {result['best_lam_ridge']:.4f}")
    print(f"Best λ Lasso: {result['best_lam_lasso']:.4f}")
