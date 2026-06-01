import os
import sys

import numpy as np
import pandas as pd
from itertools import combinations

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from part1.ols_implementation import OLSRegressor, coef_inference, model_metrics, compute_aic, compute_bic
from part1.ridge_lasso import ridge_fit, lasso_fit
from part1.cross_validation import kfold_cv
from part2.data_pipeline import run_vif_check


# ======================================================================
# HÀM HELPER
# ======================================================================

def evaluate_model(y_true, y_pred, inverse_transform=False):
    """
    Tính MAE, RMSE, R² giữa y thực và y dự đoán.
    Nếu inverse_transform=True, sẽ dùng np.expm1() để chuyển về không gian gốc trước khi tính.

    Returns
    -------
    dict : {"mae": float, "rmse": float, "r2": float}
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if inverse_transform:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_test_split_df(X, y, test_size=0.2, random_state=42):
    """
    Chia X, y thành train/test theo tỷ lệ, giữ nguyên kiểu DataFrame/Series.

    Parameters
    ----------
    X : pd.DataFrame hoặc np.ndarray
    y : pd.Series, np.ndarray hoặc list
    test_size : float (0, 1)
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    n = len(X)
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)
    n_test = int(n * test_size)
    if n_test == 0 or n_test == n:
        raise ValueError(f"test_size={test_size} dẫn đến tập train hoặc test bị rỗng với n={n}.")
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
    else:
        X_arr = np.asarray(X)
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]

    if isinstance(y, pd.Series):
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
    else:
        y_arr = np.asarray(y)
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

    return X_train, X_test, y_train, y_test


def _as_numeric_dataframe(X, feature_names=None):
    """
    Chuẩn hóa X về DataFrame số để các hàm OLS/VIF dùng chung một kiểu dữ liệu.
    """
    if isinstance(X, pd.Series):
        X_df = X.to_frame()
    elif isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if feature_names is None:
            feature_names = [f"X_{i + 1}" for i in range(X_arr.shape[1])]
        X_df = pd.DataFrame(X_arr, columns=feature_names)

    bad_cols = [
        col for col in X_df.columns
        if not pd.api.types.is_numeric_dtype(X_df[col])
    ]
    if bad_cols:
        raise ValueError(
            "OLS chỉ nhận biến đã mã hóa thành số. "
            f"Các cột chưa phải dạng số: {bad_cols}"
        )

    X_df = X_df.astype(float)
    if not np.isfinite(X_df.to_numpy()).all():
        raise ValueError("X có NaN hoặc +/-inf. Hãy xử lý missing value trước khi chọn biến.")
    return X_df


def _as_numeric_vector(y):
    """Chuẩn hóa y về vector 1 chiều dạng float."""
    y_arr = np.asarray(y, dtype=float).ravel()
    if not np.isfinite(y_arr).all():
        raise ValueError("y có NaN hoặc +/-inf. Hãy xử lý target trước khi fit OLS.")
    return y_arr


def _check_xy_shape(X, y):
    """Kiểm tra số dòng của X và y có khớp nhau không."""
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X và y không cùng số dòng: X={X.shape[0]}, y={y.shape[0]}")


def _fit_ols(X, y):
    """Fit OLS tự cài đặt ở part 1 trên DataFrame đã chọn cột."""
    return OLSRegressor(fit_intercept=True).fit(X.to_numpy(), y)


def _safe_model_metrics(y, y_hat, p):
    """Tính metric, có xử lý riêng trường hợp fit hoàn hảo làm RSS = 0."""
    try:
        return model_metrics(y, y_hat, p=p, verbose=False)
    except (ZeroDivisionError, ValueError):
        # Fallback: RSS=0 gây ZeroDivisionError hoặc TSS=0 gây ValueError, tính tay các chỉ số
        rss = float(np.sum((y - y_hat) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        ess = float(np.sum((y_hat - y.mean()) ** 2))
        n = len(y)
        df_model = p
        df_resid = n - p - 1
        r2 = np.nan if tss == 0 else 1.0 - rss / tss
        r2_adj = np.nan if df_resid <= 0 or tss == 0 else 1.0 - (n - 1) / df_resid * (1.0 - r2)
        aic = compute_aic(n, rss, p)
        bic = compute_bic(n, rss, p)

        return {
            "rss": rss,
            "tss": tss,
            "ess": ess,
            "r2": r2,
            "r2_adj": r2_adj,
            "aic": aic,
            "bic": bic,
            "f_stat": np.inf if rss == 0 and df_model > 0 and df_resid > 0 else np.nan,
            "f_pvalue": 0.0 if rss == 0 and df_model > 0 and df_resid > 0 else np.nan,
            "n": n,
            "p": p,
        }


def ols_coefficient_table(X, y, feature_names=None):
    """
    Fit OLS và trả về bảng hệ số, SE, t-stat, p-value cho từng biến.

    Bảng này dùng trực tiếp để quyết định biến nào có p-value lớn nhất.
    """
    X_df = _as_numeric_dataframe(X, feature_names)
    y_arr = _as_numeric_vector(y)
    _check_xy_shape(X_df, y_arr)

    model = _fit_ols(X_df, y_arr)
    # Tắt warning chia 0 khi sigma2 = 0 (fit hoàn hảo)
    with np.errstate(divide="ignore", invalid="ignore"):
        inference = coef_inference(
            X_df.to_numpy(),
            y_arr,
            model.beta_hat,
            model.sigma2_hat,
            verbose=False,
        )

    names = ["Intercept"] + X_df.columns.tolist()
    return pd.DataFrame(
        {
            "feature": names,
            "coef": model.beta_hat,
            "std_error": inference["se"],
            "t_stat": inference["t_stats"],
            "p_value": inference["p_values"],
            "ci_lower": inference["ci_lower"],
            "ci_upper": inference["ci_upper"],
        }
    )

# ======================================================================
# MÔ HÌNH 1: OLS CƠ BẢN (tất cả biến sau pipeline)
# ======================================================================

class OLSBasic:
    """
    OLS cơ bản — Hồi quy tuyến tính với TẤT CẢ các biến sau pipeline.
    Không chọn biến, không regularization.

    Parameters
    ----------
    verbose : bool
        In chi tiết khi gọi summary().

    Attributes (sau khi fit)
    ----------
    model_          : OLSRegressor — mô hình OLS đã fit.
    coef_table_     : DataFrame — bảng hệ số (coef, SE, t, p-value, CI).
    metrics_        : dict — các chỉ số: R², R²-adj, F-stat, ...
    feature_names_  : list[str] — tên các biến.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.model_ = None
        self.coef_table_ = None
        self.metrics_ = None
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None):
        """Fit OLS trên tất cả biến."""
        X_df = _as_numeric_dataframe(X, feature_names)
        y_arr = _as_numeric_vector(y)
        _check_xy_shape(X_df, y_arr)

        self.feature_names_ = X_df.columns.tolist()
        self.model_ = _fit_ols(X_df, y_arr)

        y_hat = self.model_.predict(X_df.to_numpy())
        self.coef_table_ = ols_coefficient_table(X_df, y_arr)
        self.metrics_ = _safe_model_metrics(y_arr, y_hat, p=len(self.feature_names_))
        return self

    def predict(self, X, feature_names=None):
        """Dự đoán y_hat trên dữ liệu mới."""
        if self.model_ is None:
            raise ValueError("Mô hình chưa được fit. Hãy gọi fit() trước.")
        X_df = _as_numeric_dataframe(X, feature_names)
        missing = set(self.feature_names_) - set(X_df.columns)
        if missing:
            raise ValueError(f"Dữ liệu predict thiếu các biến: {missing}")
        return self.model_.predict(X_df[self.feature_names_].to_numpy())

    def evaluate(self, X, y, feature_names=None, inverse_transform=False):
        """Tính MAE, RMSE, R² trên tập dữ liệu cho trước."""
        y_pred = self.predict(X, feature_names)
        return evaluate_model(y, y_pred, inverse_transform=inverse_transform)

    def summary(self):
        """In tóm tắt mô hình OLS cơ bản."""
        if self.model_ is None:
            print("Mô hình chưa được fit.")
            return

        print("=== OLS CƠ BẢN (tất cả biến) ===")
        print(f"Số biến: {len(self.feature_names_)}")
        print("-" * 30)
        print("Các chỉ số đánh giá mô hình (trên tập train):")
        for k, v in self.metrics_.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("-" * 30)
        print("Bảng hệ số:")
        print(self.coef_table_)


# ======================================================================
# MÔ HÌNH 2: OLS CHỌN BIẾN
# ======================================================================
class OLSFeatureSelector:
    """
    Class hỗ trợ chọn biến OLS (Feature Selection) sử dụng p-value và VIF.
    Thiết kế theo dạng hướng đối tượng để dễ dàng lưu trữ trạng thái và đánh giá mô hình.

    Parameters
    ----------
    method : str
        'pvalue' = loại theo p-value, 'vif' = loại theo VIF, 'both' = VIF trước rồi p-value.
    alpha : float
        Ngưỡng p-value để giữ biến (mặc định 0.05).
    vif_threshold : float
        Ngưỡng VIF tối đa cho phép (mặc định 10.0).
    min_features : int
        Số biến tối thiểu phải giữ lại.
    max_iter : int or None
        Số vòng lặp tối đa. None = bằng số biến ban đầu.
    verbose : bool
        In chi tiết quá trình loại biến.

    Attributes (sau khi fit)
    ----------
    selected_features_ : list[str]    - Tên các biến được giữ.
    dropped_features_  : list[str]    - Tên các biến bị loại, theo thứ tự.
    model_             : OLSRegressor - Mô hình OLS cuối cùng.
    coef_table_        : DataFrame    - Bảng hệ số (coef, SE, t, p-value, CI).
    metrics_           : dict         - Các chỉ số: R², R²-adj, F-stat, ...
    vif_table_         : DataFrame    - Bảng VIF sau cùng (nếu có).
    """
    def __init__(self, method="both", alpha=0.05, vif_threshold=10.0, min_features=1, max_iter=None, verbose=True):
        self.method = method
        self.alpha = alpha
        self.vif_threshold = vif_threshold
        self.min_features = min_features
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.selected_features_ = None
        self.dropped_features_ = None
        self.history_ = None
        self.model_ = None
        self.coef_table_ = None
        self.metrics_ = None
        self.vif_table_ = None
        self.X_selected_ = None

    def fit(self, X, y, feature_names=None):
        """Chạy backward elimination để chọn biến, lưu kết quả vào attributes."""
        X_df = _as_numeric_dataframe(X, feature_names)
        y_arr = _as_numeric_vector(y)
        _check_xy_shape(X_df, y_arr)

        if self.method == "pvalue":
            result = self._backward_elimination_pvalue(X_df, y_arr)
        elif self.method == "vif":
            result = self._backward_elimination_vif(X_df, y_arr)
        elif self.method == "both":
            # Bước 1: loại đa cộng tuyến bằng VIF (chưa cần y)
            vif_result = self._backward_elimination_vif(X_df, y=None)
            # Bước 2: loại biến không có ý nghĩa thống kê bằng p-value
            pvalue_result = self._backward_elimination_pvalue(
                vif_result["X_selected"], y_arr
            )

            # Gộp lịch sử 2 giai đoạn: VIF trước, p-value sau
            pvalue_result["dropped_features"] = (
                vif_result["dropped_features"] + pvalue_result["dropped_features"]
            )
            pvalue_result["history"] = vif_result["history"] + pvalue_result["history"]
            pvalue_result["vif_table"] = run_vif_check(
                pvalue_result["X_selected"]
            )
            result = pvalue_result
        else:
            raise ValueError("method phải là 'pvalue', 'vif', hoặc 'both'.")
            
        # Lưu kết quả vào attributes để truy cập từ notebook
        self.X_selected_ = result.get("X_selected")
        self.selected_features_ = result.get("selected_features")
        self.dropped_features_ = result.get("dropped_features")
        self.history_ = result.get("history")
        self.model_ = result.get("model")
        self.coef_table_ = result.get("coef_table")
        self.metrics_ = result.get("metrics")
        self.vif_table_ = result.get("vif_table")
        
        return self

    def transform(self, X, feature_names=None):
        """Trả về DataFrame chỉ chứa các cột đã được chọn sau fit."""
        if self.selected_features_ is None:
            raise ValueError("Mô hình chưa được fit. Hãy gọi fit() trước.")
        X_df = _as_numeric_dataframe(X, feature_names)
        missing = set(self.selected_features_) - set(X_df.columns)
        if missing:
            raise ValueError(f"Dữ liệu transform/predict thiếu các biến: {missing}")
        return X_df[self.selected_features_]

    def fit_transform(self, X, y, feature_names=None):
        """Gọi fit rồi transform trong 1 bước."""
        return self.fit(X, y, feature_names).transform(X, feature_names)

    def predict(self, X, feature_names=None):
        """Dự đoán y_hat trên dữ liệu mới (tự động chọn cột đã fit)."""
        if self.model_ is None:
            raise ValueError("Mô hình chưa được fit. Hãy gọi fit() trước.")
        X_sel = self.transform(X, feature_names)
        return self.model_.predict(X_sel.to_numpy())

    def evaluate(self, X, y, feature_names=None, inverse_transform=False):
        """Tính MAE, RMSE, R² trên tập dữ liệu cho trước."""
        y_pred = self.predict(X, feature_names)
        return evaluate_model(y, y_pred, inverse_transform=inverse_transform)

    def summary(self):
        """In ra tóm tắt thông tin mô hình đã chọn."""
        if self.model_ is None:
            print("Mô hình chưa được fit.")
            return
        
        print("=== KẾT QUẢ CHỌN BIẾN OLS ===")
        print(f"Phương pháp: {self.method}")
        print(f"Số biến ban đầu: {len(self.selected_features_) + len(self.dropped_features_)}")
        print(f"Số biến được giữ lại: {len(self.selected_features_)}")
        print(f"Các biến bị loại: {self.dropped_features_}")
        print("-" * 30)
        print("Các chỉ số đánh giá mô hình:")
        for k, v in self.metrics_.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("-" * 30)
        print("Bảng hệ số:")
        print(self.coef_table_)
        if self.vif_table_ is not None:
            print("-" * 30)
            print("Bảng VIF:")
            print(self.vif_table_)

    def _backward_elimination_pvalue(self, X_df, y_arr):
        """Loại biến có p-value lớn nhất > alpha mỗi vòng, dừng khi tất cả <= alpha."""
        max_iter = X_df.shape[1] if self.max_iter is None else self.max_iter
        selected = X_df.columns.tolist()
        dropped = []
        history = []

        for iteration in range(1, max_iter + 1):
            if len(selected) <= self.min_features:
                break

            X_current = X_df[selected]

            try:
                coef_table = ols_coefficient_table(X_current, y_arr)
            except ValueError:
                # Ma trận suy biến (đa cộng tuyến hoàn hảo) → dùng VIF để loại 1 cột
                vif_df = run_vif_check(X_current)
                worst_vif = vif_df.iloc[0]
                drop_feature = str(worst_vif["feature"])
                dropped.append(drop_feature)
                selected.remove(drop_feature)
                history.append(
                    {
                        "iteration": iteration,
                        "dropped": drop_feature,
                        "criterion": "singular_matrix_vif",
                        "value": float(worst_vif["VIF"]),
                        "n_features": len(selected),
                    }
                )
                if self.verbose:
                    print(
                        f"[p-value iter {iteration}] OLS suy biến, "
                        f"loại '{drop_feature}' theo VIF={worst_vif['VIF']:.4f}"
                    )
                continue

            # Bỏ Intercept, chỉ xét p-value của biến giải thích
            feature_rows = coef_table[coef_table["feature"] != "Intercept"].copy()
            # NaN p-value (fit hoàn hảo, hệ số = 0) → coi như không có ý nghĩa
            feature_rows["_selection_pvalue"] = feature_rows["p_value"].fillna(1.0)
            worst = feature_rows.sort_values("_selection_pvalue", ascending=False).iloc[0]
            worst_pvalue = float(worst["_selection_pvalue"])
            drop_feature = str(worst["feature"])

            if worst_pvalue <= self.alpha:  # Tất cả biến đều có ý nghĩa --> dừng
                history.append(
                    {
                        "iteration": iteration,
                        "dropped": None,
                        "criterion": "p_value",
                        "value": worst_pvalue,
                        "n_features": len(selected),
                    }
                )
                break

            dropped.append(drop_feature)
            selected.remove(drop_feature)
            history.append(
                {
                    "iteration": iteration,
                    "dropped": drop_feature,
                    "criterion": "p_value",
                    "value": worst_pvalue,
                    "n_features": len(selected),
                }
            )

            if self.verbose:
                print(
                    f"[p-value iter {iteration}] Loại '{drop_feature}' "
                    f"(p-value={worst_pvalue:.6f})"
                )

        # Fit lại mô hình cuối cùng với tập biến còn lại
        X_selected = X_df[selected]
        final_model = _fit_ols(X_selected, y_arr)
        y_hat = final_model.predict(X_selected.to_numpy())
        final_table = ols_coefficient_table(X_selected, y_arr)
        metrics = _safe_model_metrics(y_arr, y_hat, p=len(selected))

        return {
            "X_selected": X_selected,
            "selected_features": selected,
            "dropped_features": dropped,
            "history": history,
            "model": final_model,
            "coef_table": final_table,
            "metrics": metrics,
        }

    def _backward_elimination_vif(self, X_df, y_arr=None):
        """Loại biến có VIF cao nhất > threshold mỗi vòng, dừng khi tất cả <= threshold."""
        if self.vif_threshold <= 1:
            raise ValueError("threshold nên > 1. Thường dùng 5 hoặc 10 cho VIF.")

        max_iter = X_df.shape[1] if self.max_iter is None else self.max_iter
        selected = X_df.columns.tolist()
        dropped = []
        history = []

        for iteration in range(1, max_iter + 1):
            if len(selected) <= self.min_features:
                break

            vif_df = run_vif_check(X_df[selected])
            worst = vif_df.iloc[0]
            worst_vif = float(worst["VIF"])
            drop_feature = str(worst["feature"])

            if np.isfinite(worst_vif) and worst_vif <= self.vif_threshold:  # Không còn đa cộng tuyến → dừng
                history.append(
                    {
                        "iteration": iteration,
                        "dropped": None,
                        "criterion": "VIF",
                        "value": worst_vif,
                        "n_features": len(selected),
                    }
                )
                break

            dropped.append(drop_feature)
            selected.remove(drop_feature)
            history.append(
                {
                    "iteration": iteration,
                    "dropped": drop_feature,
                    "criterion": "VIF",
                    "value": worst_vif,
                    "n_features": len(selected),
                }
            )

            if self.verbose:
                print(
                    f"[VIF iter {iteration}] Loại '{drop_feature}' "
                    f"(VIF={worst_vif:.4f})"
                )

        result = {
            "X_selected": X_df[selected],
            "selected_features": selected,
            "dropped_features": dropped,
            "history": history,
            "vif_table": run_vif_check(X_df[selected]),
        }

        # Nếu có y, fit OLS trên tập biến đã loại xong để tính metrics
        if y_arr is not None:
            final_model = _fit_ols(result["X_selected"], y_arr)
            y_hat = final_model.predict(result["X_selected"].to_numpy())
            result["model"] = final_model
            result["coef_table"] = ols_coefficient_table(result["X_selected"], y_arr)
            result["metrics"] = _safe_model_metrics(y_arr, y_hat, p=len(selected))

        return result


# ======================================================================
# MÔ HÌNH 3: RIDGE REGRESSION + K-FOLD CV
# ======================================================================

class RidgeCV:
    """
    Ridge Regression + k-fold Cross-Validation để chọn λ tối ưu.

    Parameters
    ----------
    lambdas  : array-like hoặc None
        Dãy λ cần quét. Mặc định: np.logspace(-4, 4, 50).
    k_folds  : int
        Số fold cho CV (mặc định 5).
    verbose  : bool
        In chi tiết khi gọi summary().

    Attributes (sau khi fit)
    ----------
    best_lambda_    : float — λ tối ưu từ CV.
    cv_results_     : list[dict] — {lam, cv_mse, cv_rmse, std_fold_mse, std_fold_rmse} mỗi λ.
    coef_           : np.ndarray — beta_hat cuối cùng (bao gồm intercept).
    feature_names_  : list[str]
    """

    def __init__(self, lambdas=None, k_folds=5, verbose=True):
        self.lambdas = lambdas if lambdas is not None else np.logspace(-4, 4, 50)
        self.k_folds = k_folds
        self.verbose = verbose

        self.best_lambda_ = None
        self.cv_results_ = None
        self.coef_ = None
        self.feature_names_ = None
        self._fitted = False

    def fit(self, X, y, feature_names=None):
        """
        Chạy k-fold CV để tìm λ tốt nhất, rồi fit lại trên toàn bộ train.
        Dùng lại hàm kfold_cv từ part1 — chọn λ theo CV(k).
        """
        X_df = _as_numeric_dataframe(X, feature_names)
        y_arr = _as_numeric_vector(y)
        _check_xy_shape(X_df, y_arr)

        self.feature_names_ = X_df.columns.tolist()
        X_np = X_df.to_numpy()

        # --- Cross-validation: gọi kfold_cv từ part1 cho mỗi λ ---
        cv_results = []
        for lam in self.lambdas:
            cv_res = kfold_cv(
                X_np, y_arr,
                k=self.k_folds,
                model="ridge",
                lam=lam,
                random_state=42,
            )
            cv_results.append({
                "lam": lam,
                "cv_mse": cv_res["cv_mse"],
                "cv_rmse": cv_res["cv_rmse"],
                "std_fold_mse": cv_res["std_fold_mse"],
                "std_fold_rmse": cv_res["std_fold_rmse"],
            })

        self.cv_results_ = cv_results

        # Chọn λ* có CV MSE thấp nhất
        best_idx = int(np.argmin([r["cv_mse"] for r in cv_results]))
        self.best_lambda_ = cv_results[best_idx]["lam"]

        # Fit lại trên toàn bộ train với λ*
        final_res = ridge_fit(X_np, y_arr, lam=self.best_lambda_)
        self.coef_ = final_res["beta_hat"]
        self._fitted = True

        return self

    def predict(self, X, feature_names=None):
        """Dự đoán y_hat trên dữ liệu mới."""
        if not self._fitted:
            raise ValueError("Mô hình chưa được fit. Hãy gọi fit() trước.")
        X_df = _as_numeric_dataframe(X, feature_names)
        missing = set(self.feature_names_) - set(X_df.columns)
        if missing:
            raise ValueError(f"Dữ liệu predict thiếu các biến: {missing}")
        X_df = X_df[self.feature_names_]
        X_design = np.hstack([np.ones((X_df.shape[0], 1)), X_df.to_numpy()])
        return X_design @ self.coef_

    def evaluate(self, X, y, feature_names=None, inverse_transform=False):
        """Tính MAE, RMSE, R² trên tập dữ liệu cho trước."""
        y_pred = self.predict(X, feature_names)
        return evaluate_model(y, y_pred, inverse_transform=inverse_transform)

    def summary(self):
        """In tóm tắt mô hình Ridge CV."""
        if not self._fitted:
            print("Mô hình chưa được fit.")
            return

        print("=== RIDGE REGRESSION (k-fold CV) ===")
        print(f"Số biến: {len(self.feature_names_)}")
        print(f"Số fold: {self.k_folds}")
        print(f"Số λ đã quét: {len(self.lambdas)}")
        print(f"λ tối ưu: {self.best_lambda_:.6f}")
        best_cv = next(r for r in self.cv_results_ if r["lam"] == self.best_lambda_)
        print(f"CV MSE: {best_cv['cv_mse']:.6f} (RMSE: {best_cv['cv_rmse']:.6f})")
        print("-" * 30)
        print(f"Hệ số (beta_hat): {self.coef_.shape[0]} tham số (bao gồm intercept)")


# ======================================================================
# MÔ HÌNH 4: LASSO REGRESSION + K-FOLD CV
# ======================================================================

class LassoCV:
    """
    Lasso Regression + k-fold Cross-Validation để chọn λ tối ưu.

    Parameters
    ----------
    lambdas   : array-like hoặc None
        Dãy λ cần quét. Mặc định: np.logspace(-6, 0, 50).
    k_folds   : int
        Số fold cho CV (mặc định 5).
    max_iter  : int
        Số vòng lặp tối đa cho coordinate descent (mặc định 1000).
    tol       : float
        Ngưỡng hội tụ (mặc định 1e-6).
    verbose   : bool
        In chi tiết khi gọi summary().

    Attributes (sau khi fit)
    ----------
    best_lambda_    : float — λ tối ưu từ CV.
    cv_results_     : list[dict] — {lam, cv_mse, cv_rmse, std_fold_mse, std_fold_rmse} mỗi λ.
    coef_           : np.ndarray — beta_hat cuối cùng (bao gồm intercept).
    n_nonzero_      : int — số hệ số ≠ 0 (không tính intercept).
    feature_names_  : list[str]
    """

    def __init__(self, lambdas=None, k_folds=5, max_iter=1000, tol=1e-6, verbose=True):
        self.lambdas = lambdas if lambdas is not None else np.logspace(-6, 0, 50)
        self.k_folds = k_folds
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.best_lambda_ = None
        self.cv_results_ = None
        self.coef_ = None
        self.n_nonzero_ = None
        self.feature_names_ = None
        self._fitted = False

    def fit(self, X, y, feature_names=None):
        """
        Chạy k-fold CV để tìm λ tốt nhất, rồi fit lại trên toàn bộ train.
        Dùng lại hàm kfold_cv từ part1 — chọn λ theo CV(k).
        """
        X_df = _as_numeric_dataframe(X, feature_names)
        y_arr = _as_numeric_vector(y)
        _check_xy_shape(X_df, y_arr)

        self.feature_names_ = X_df.columns.tolist()
        X_np = X_df.to_numpy()

        # --- Cross-validation: gọi kfold_cv từ part1 cho mỗi λ ---
        cv_results = []
        for lam in self.lambdas:
            cv_res = kfold_cv(
                X_np, y_arr,
                k=self.k_folds,
                model="lasso",
                lam=lam,
                random_state=42,
            )
            cv_results.append({
                "lam": lam,
                "cv_mse": cv_res["cv_mse"],
                "cv_rmse": cv_res["cv_rmse"],
                "std_fold_mse": cv_res["std_fold_mse"],
                "std_fold_rmse": cv_res["std_fold_rmse"],
            })

        self.cv_results_ = cv_results

        # Chọn λ* có CV MSE thấp nhất
        best_idx = int(np.argmin([r["cv_mse"] for r in cv_results]))
        self.best_lambda_ = cv_results[best_idx]["lam"]

        # Fit lại trên toàn bộ train với λ*
        final_res = lasso_fit(
            X_np, y_arr, lam=self.best_lambda_,
            max_iter=self.max_iter, tol=self.tol,
        )
        self.coef_ = final_res["beta_hat"]
        self.n_nonzero_ = final_res["n_nonzero"]
        self._fitted = True

        return self

    def predict(self, X, feature_names=None):
        """Dự đoán y_hat trên dữ liệu mới."""
        if not self._fitted:
            raise ValueError("Mô hình chưa được fit. Hãy gọi fit() trước.")
        X_df = _as_numeric_dataframe(X, feature_names)
        missing = set(self.feature_names_) - set(X_df.columns)
        if missing:
            raise ValueError(f"Dữ liệu predict thiếu các biến: {missing}")
        X_df = X_df[self.feature_names_]
        X_design = np.hstack([np.ones((X_df.shape[0], 1)), X_df.to_numpy()])
        return X_design @ self.coef_

    def evaluate(self, X, y, feature_names=None, inverse_transform=False):
        """Tính MAE, RMSE, R² trên tập dữ liệu cho trước."""
        y_pred = self.predict(X, feature_names)
        return evaluate_model(y, y_pred, inverse_transform=inverse_transform)

    def summary(self):
        """In tóm tắt mô hình Lasso CV."""
        if not self._fitted:
            print("Mô hình chưa được fit.")
            return

        print("=== LASSO REGRESSION (k-fold CV) ===")
        print(f"Số biến: {len(self.feature_names_)}")
        print(f"Số fold: {self.k_folds}")
        print(f"Số λ đã quét: {len(self.lambdas)}")
        print(f"λ tối ưu: {self.best_lambda_:.6f}")
        best_cv = next(r for r in self.cv_results_ if r["lam"] == self.best_lambda_)
        print(f"CV MSE: {best_cv['cv_mse']:.6f} (RMSE: {best_cv['cv_rmse']:.6f})")
        print(f"Số hệ số ≠ 0: {self.n_nonzero_}/{len(self.feature_names_)}")
        print("-" * 30)
        print(f"Hệ số (beta_hat): {self.coef_.shape[0]} tham số (bao gồm intercept)")

# ======================================================================
# TRÌNH TẠO ĐẶC ĐIỂM 1: POLYNOMIAL FEATURES GENNERATOR
# ======================================================================
class PolynomialFeatureGenerator:
    """
    Polynomial Features — Chỉ sinh đặc trưng đa thức (lũy thừa) từ các cột số.
    """

    def __init__(self, degree=2, top_k=15, use_correlation=True, verbose=True):
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree phải là số nguyên >= 1.")
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("top_k phải là số nguyên dương hoặc None.")
            
        self.degree = degree
        self.top_k = top_k
        self.use_correlation = use_correlation

        self.verbose = verbose

        self.poly_cols_ = None
        self.new_col_names_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        # [TÍCH HỢP HÀM CHECK]: Chuẩn hóa X thành DataFrame số
        X_df = _as_numeric_dataframe(X)

        if self.degree == 1:
            self.poly_cols_ = []
            self.new_col_names_ = []
            self.feature_names_ = X_df.columns.tolist() 
            return self

        # Lọc cột số liên tục (Lúc này X_df đã đảm bảo là số)
        num_cols = X_df.columns.tolist()
        num_cols = [c for c in num_cols if X_df[c].nunique() > 2]

        # Lọc top_k theo Hệ số biến thiên (CV)
        if self.top_k is not None and len(num_cols) > self.top_k:
            if self.use_correlation and y is not None:
                # Cách 1: Dùng hệ số tương quan với y (Ưu tiên)
                y_series = pd.Series(np.asarray(y).ravel())
                correlations = X_df[num_cols].apply(lambda col: col.corr(y_series))
                num_cols = correlations.abs().nlargest(self.top_k).index.tolist()
                
                if self.verbose:
                    print(f"  [Lọc biến] Đã giữ lại top {self.top_k} biến dựa trên Correlation với y.")
            else:
                # Cách 2: Dùng hệ số biến thiên CV (Fallback)
                means = X_df[num_cols].mean().abs() + 1e-8
                cv = X_df[num_cols].std() / means
                num_cols = cv.nlargest(self.top_k).index.tolist()
                
                if self.verbose:
                    reason = "không truyền y" if self.use_correlation else "use_correlation=False"
                    print(f"  [Lọc biến] Do {reason}, đã giữ lại top {self.top_k} biến dựa trên CV.")

        self.poly_cols_ = num_cols
        self.new_col_names_ = []

        if self.poly_cols_:
            # Duyệt từ bậc 2 đến degree để sinh tên biến lũy thừa
            for d in range(2, self.degree + 1):
                for col in self.poly_cols_:
                    self.new_col_names_.append(f"{col}^{d}")

        self.feature_names_ = X_df.columns.tolist() + self.new_col_names_

        if self.verbose and self.new_col_names_:
            print(f"  [Polynomial] {len(self.poly_cols_)} cột gốc → {len(self.new_col_names_)} features mới (degree={self.degree})")
        return self

    def transform(self, X):
        if self.poly_cols_ is None:
            raise ValueError("Mô hình chưa được fit.")
            
        # [TÍCH HỢP HÀM CHECK]: Chuẩn hóa dữ liệu mới truyền vào
        X_df = _as_numeric_dataframe(X)

        missing = set(self.poly_cols_) - set(X_df.columns)
        if missing:
            raise ValueError(f"Transform data thiếu các cột gốc: {missing}. "
                             "Hãy đảm bảo dữ liệu mới có đầy đủ các cột đã dùng khi fit.")
        
        if self.degree == 1 or not self.poly_cols_:
            return X_df.copy()

        new_cols = {}
        for d in range(2, self.degree + 1):
            for col in self.poly_cols_:
                if col in X_df.columns:
                    new_cols[f"{col}^{d}"] = X_df[col] ** d

        new_df = pd.DataFrame(new_cols, index=X_df.index)
        new_df = new_df.reindex(columns=self.new_col_names_, fill_value=0)
        
        return pd.concat([X_df, new_df], axis=1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def summary(self):
        if self.poly_cols_ is None:
            print("Mô hình chưa được fit.")
            return

        width = 55
        print("=" * width)
        print("   POLYNOMIAL FEATURES (ĐA THỨC)".center(width))
        print("=" * width)
        print(f"  {'Bậc đa thức (degree)':<35} {self.degree}")
        print(f"  {'Số cột gốc tham gia':<35} {len(self.poly_cols_)}")
        print(f"  {'Số đặc trưng mới sinh ra':<35} {len(self.new_col_names_)}")
        print(f"  {'Tổng biến sau mở rộng':<35} {len(self.feature_names_)}")
        print("-" * width)

        print("  Cột gốc được chọn:")
        for i, col in enumerate(self.poly_cols_, 1):
            print(f"    {i:>2}. {col}")
        print("-" * width)

        print("  Đặc trưng mới (Top 10):")
        for i, col in enumerate(self.new_col_names_[:10], 1):
            print(f"    {i:>2}. {col}")
        if len(self.new_col_names_) > 10:
            print(f"       ... và {len(self.new_col_names_) - 10} đặc trưng khác.")
        print("=" * width)

# ======================================================================
# TRÌNH TẠO ĐẶC TRƯNG 2: INTERACTION FEATURES GENERATOR
# ======================================================================

class InteractionFeatureGenerator:
    """
    Interaction Features — Chỉ sinh đặc trưng tương tác (nhân chéo) giữa các cột số khác nhau.
    """

    def __init__(self, degree=2, top_k=15, use_correlation=True, verbose=True):
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree phải là số nguyên >= 1.")
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("top_k phải là số nguyên dương hoặc None.")
            
        self.degree = degree
        self.top_k = top_k
        self.use_correlation = use_correlation 
        self.verbose = verbose

        self.interact_cols_ = None
        self.new_col_names_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        # [TÍCH HỢP HÀM CHECK]: Chuẩn hóa X thành DataFrame số
        X_df = _as_numeric_dataframe(X)

        # Lọc cột số liên tục (Lúc này X_df đã đảm bảo là số)
        num_cols = X_df.columns.tolist()
        num_cols = [c for c in num_cols if X_df[c].nunique() > 2]

        # Lọc top_k theo Hệ số biến thiên (CV)
        if self.top_k is not None and len(num_cols) > self.top_k:
            if self.use_correlation and y is not None:
                # Cách 1: Dùng hệ số tương quan với y (Ưu tiên)
                y_series = pd.Series(np.asarray(y).ravel())
                correlations = X_df[num_cols].apply(lambda col: col.corr(y_series))
                num_cols = correlations.abs().nlargest(self.top_k).index.tolist()
                
                if self.verbose:
                    print(f"  [Lọc biến] Đã giữ lại top {self.top_k} biến dựa trên Correlation với y.")
            else:
                # Cách 2: Dùng hệ số biến thiên CV (Fallback)
                means = X_df[num_cols].mean().abs() + 1e-8
                cv = X_df[num_cols].std() / means
                num_cols = cv.nlargest(self.top_k).index.tolist()
                
                if self.verbose:
                    reason = "không truyền y" if self.use_correlation else "use_correlation=False"
                    print(f"  [Lọc biến] Do {reason}, đã giữ lại top {self.top_k} biến dựa trên CV.")

        self.interact_cols_ = num_cols
        self.new_col_names_ = []

        if self.interact_cols_:
            for d in range(2, self.degree + 1):
                for cols in combinations(self.interact_cols_, d):
                    col_name = "_x_".join(cols)
                    self.new_col_names_.append(col_name)

        self.feature_names_ = X_df.columns.tolist() + self.new_col_names_

        if self.verbose and self.new_col_names_:
            print(f"  [Interaction] {len(self.interact_cols_)} cột gốc → {len(self.new_col_names_)} features mới (degree={self.degree})")
        return self

    def transform(self, X):
        if self.interact_cols_ is None:
            raise ValueError("Mô hình chưa được fit.")
            
        # [TÍCH HỢP HÀM CHECK]: Chuẩn hóa dữ liệu mới truyền vào
        X_df = _as_numeric_dataframe(X)
        
        missing = set(self.interact_cols_) - set(X_df.columns)
        if missing:
            raise ValueError(f"Transform data thiếu các cột gốc: {missing}. "
                             "Hãy đảm bảo dữ liệu mới có đầy đủ các cột đã dùng khi fit.")
        
        if not self.interact_cols_:
            return X_df.copy()

        new_cols = {}

        for d in range(2, self.degree + 1):
            for cols in combinations(self.interact_cols_, d):
                # FIX BUGS: Đã định nghĩa lại col_name trong scope này
                col_name = "_x_".join(cols)
                if all(c in X_df.columns for c in cols):
                    new_cols[col_name] = X_df[list(cols)].prod(axis=1)

        new_df = pd.DataFrame(new_cols, index=X_df.index)
        new_df = new_df.reindex(columns=self.new_col_names_, fill_value=0)
        
        return pd.concat([X_df, new_df], axis=1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def summary(self):
        if self.interact_cols_ is None:
            print("Mô hình chưa được fit.")
            return

        width = 55
        print("=" * width)
        print("   INTERACTION FEATURES (TƯƠNG TÁC)".center(width))
        print("=" * width)
        print(f"  {'Bậc tương tác lớn nhất':<35} {self.degree}")
        print(f"  {'Số cột gốc tham gia':<35} {len(self.interact_cols_)}")
        print(f"  {'Số đặc trưng mới sinh ra':<35} {len(self.new_col_names_)}")
        print(f"  {'Tổng biến sau mở rộng':<35} {len(self.feature_names_)}")
        print("-" * width)

        print("  Cột gốc được chọn:")
        for i, col in enumerate(self.interact_cols_, 1):
            print(f"    {i:>2}. {col}")
        print("-" * width)

        print("  Đặc trưng mới (Top 10):")
        for i, col in enumerate(self.new_col_names_[:10], 1):
            print(f"    {i:>2}. {col}")
        if len(self.new_col_names_) > 10:
            print(f"       ... và {len(self.new_col_names_) - 10} đặc trưng khác.")
        print("=" * width)