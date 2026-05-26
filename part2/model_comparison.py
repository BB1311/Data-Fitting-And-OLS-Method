from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from part1.ols_implementation import OLSRegressor, coef_inference, model_metrics
from part2.data_pipeline import run_vif_check


def _as_numeric_dataframe(
    X: pd.DataFrame | pd.Series | np.ndarray,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
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


def _as_numeric_vector(y: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    """Chuẩn hóa y về vector 1 chiều dạng float."""
    y_arr = np.asarray(y, dtype=float).ravel()
    if not np.isfinite(y_arr).all():
        raise ValueError("y có NaN hoặc +/-inf. Hãy xử lý target trước khi fit OLS.")
    return y_arr


def _check_xy_shape(X: pd.DataFrame, y: np.ndarray) -> None:
    """Kiểm tra số dòng của X và y có khớp nhau không."""
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X và y không cùng số dòng: X={X.shape[0]}, y={y.shape[0]}")


def _fit_ols(X: pd.DataFrame, y: np.ndarray) -> OLSRegressor:
    """Fit OLS tự cài đặt ở part 1 trên DataFrame đã chọn cột."""
    return OLSRegressor(fit_intercept=True).fit(X.to_numpy(), y)


def _safe_model_metrics(y: np.ndarray, y_hat: np.ndarray, p: int) -> dict[str, Any]:
    """Tính metric, có xử lý riêng trường hợp fit hoàn hảo làm RSS = 0."""
    try:
        return model_metrics(y, y_hat, p=p, verbose=False)
    except ZeroDivisionError:
        rss = float(np.sum((y - y_hat) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        ess = float(np.sum((y_hat - y.mean()) ** 2))
        n = len(y)
        df_model = p
        df_resid = n - p - 1
        r2 = np.nan if tss == 0 else 1.0 - rss / tss
        r2_adj = np.nan if df_resid <= 0 or tss == 0 else 1.0 - (n - 1) / df_resid * (1.0 - r2)
        return {
            "rss": rss,
            "tss": tss,
            "ess": ess,
            "r2": r2,
            "r2_adj": r2_adj,
            "f_stat": np.inf if rss == 0 and df_model > 0 and df_resid > 0 else np.nan,
            "f_pvalue": 0.0 if rss == 0 and df_model > 0 and df_resid > 0 else np.nan,
            "n": n,
            "p": p,
        }


def ols_coefficient_table(
    X: pd.DataFrame | pd.Series | np.ndarray,
    y: pd.Series | np.ndarray | list[float],
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fit OLS và trả về bảng hệ số, SE, t-stat, p-value cho từng biến.

    Bảng này dùng trực tiếp để quyết định biến nào có p-value lớn nhất.
    """
    X_df = _as_numeric_dataframe(X, feature_names)
    y_arr = _as_numeric_vector(y)
    _check_xy_shape(X_df, y_arr)

    model = _fit_ols(X_df, y_arr)
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


def backward_elimination_pvalue(
    X: pd.DataFrame | pd.Series | np.ndarray,
    y: pd.Series | np.ndarray | list[float],
    alpha: float = 0.05,
    min_features: int = 1,
    max_iter: int | None = None,
    feature_names: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Loại biến theo p-value bằng backward elimination.

    Mỗi vòng lặp:
      1. Fit OLS với tập biến hiện tại.
      2. Lấy p-value của các biến giải thích, không tính Intercept.
      3. Nếu p-value lớn nhất > alpha thì loại biến đó.
      4. Dừng khi tất cả p-value <= alpha hoặc đã chạm min_features.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha phải nằm trong khoảng (0, 1).")

    X_df = _as_numeric_dataframe(X, feature_names)
    y_arr = _as_numeric_vector(y)
    _check_xy_shape(X_df, y_arr)

    max_iter = X_df.shape[1] if max_iter is None else max_iter
    selected = X_df.columns.tolist()
    dropped: list[str] = []
    history: list[dict[str, Any]] = []

    for iteration in range(1, max_iter + 1):
        if len(selected) <= min_features:
            break

        X_current = X_df[selected]

        try:
            coef_table = ols_coefficient_table(X_current, y_arr)
        except ValueError:
            # Nếu OLS lỗi do đa cộng tuyến hoàn hảo, dùng VIF để bỏ bớt 1 cột trước.
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
            if verbose:
                print(
                    f"[p-value iter {iteration}] OLS suy biến, "
                    f"loại '{drop_feature}' theo VIF={worst_vif['VIF']:.4f}"
                )
            continue

        feature_rows = coef_table[coef_table["feature"] != "Intercept"].copy()
        # p-value = NaN thường xảy ra khi fit hoàn hảo và hệ số bằng 0;
        # với chọn biến, xem nó như biến không có ý nghĩa để có thể loại.
        feature_rows["_selection_pvalue"] = feature_rows["p_value"].fillna(1.0)
        worst = feature_rows.sort_values("_selection_pvalue", ascending=False).iloc[0]
        worst_pvalue = float(worst["_selection_pvalue"])
        drop_feature = str(worst["feature"])

        if worst_pvalue <= alpha:
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

        if verbose:
            print(
                f"[p-value iter {iteration}] Loại '{drop_feature}' "
                f"(p-value={worst_pvalue:.6f})"
            )

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


def backward_elimination_vif(
    X: pd.DataFrame | pd.Series | np.ndarray,
    y: pd.Series | np.ndarray | list[float] | None = None,
    threshold: float = 10.0,
    min_features: int = 1,
    max_iter: int | None = None,
    feature_names: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Loại biến theo VIF để giảm đa cộng tuyến.

    Mỗi vòng lặp loại cột có VIF cao nhất nếu VIF đó vượt threshold.
    """
    if threshold <= 1:
        raise ValueError("threshold nên > 1. Thường dùng 5 hoặc 10 cho VIF.")

    X_df = _as_numeric_dataframe(X, feature_names)
    y_arr = None if y is None else _as_numeric_vector(y)
    if y_arr is not None:
        _check_xy_shape(X_df, y_arr)

    max_iter = X_df.shape[1] if max_iter is None else max_iter
    selected = X_df.columns.tolist()
    dropped: list[str] = []
    history: list[dict[str, Any]] = []

    for iteration in range(1, max_iter + 1):
        if len(selected) <= min_features:
            break

        vif_df = run_vif_check(X_df[selected], threshold=threshold)
        worst = vif_df.iloc[0]
        worst_vif = float(worst["VIF"])
        drop_feature = str(worst["feature"])

        if np.isfinite(worst_vif) and worst_vif <= threshold:
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

        if verbose:
            print(
                f"[VIF iter {iteration}] Loại '{drop_feature}' "
                f"(VIF={worst_vif:.4f})"
            )

    result: dict[str, Any] = {
        "X_selected": X_df[selected],
        "selected_features": selected,
        "dropped_features": dropped,
        "history": history,
        "vif_table": run_vif_check(X_df[selected], threshold=threshold),
    }

    if y_arr is not None:
        final_model = _fit_ols(result["X_selected"], y_arr)
        y_hat = final_model.predict(result["X_selected"].to_numpy())
        result["model"] = final_model
        result["coef_table"] = ols_coefficient_table(result["X_selected"], y_arr)
        result["metrics"] = _safe_model_metrics(y_arr, y_hat, p=len(selected))

    return result


class OLSFeatureSelector:
    """
    Mô hình OLS có chọn biến theo p-value, VIF, hoặc kết hợp cả hai.

    method:
      - "pvalue": chỉ loại theo p-value.
      - "vif": chỉ loại theo VIF.
      - "both": loại VIF trước, sau đó loại p-value.
    """

    def __init__(
        self,
        method: str = "both",
        alpha: float = 0.05,
        vif_threshold: float = 10.0,
        min_features: int = 1,
        max_iter: int | None = None,
        verbose: bool = True,
    ):
        if method not in {"pvalue", "vif", "both"}:
            raise ValueError("method phải là 'pvalue', 'vif', hoặc 'both'.")
        self.method = method
        self.alpha = alpha
        self.vif_threshold = vif_threshold
        self.min_features = min_features
        self.max_iter = max_iter
        self.verbose = verbose

        self.selected_features_: list[str] | None = None
        self.dropped_features_: list[str] = []
        self.history_: list[dict[str, Any]] = []
        self.model_: OLSRegressor | None = None
        self.coef_table_: pd.DataFrame | None = None
        self.vif_table_: pd.DataFrame | None = None
        self.metrics_: dict[str, Any] | None = None
        self.feature_names_in_: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray,
        y: pd.Series | np.ndarray | list[float],
        feature_names: list[str] | None = None,
    ) -> "OLSFeatureSelector":
        """Chọn biến và fit OLS cuối cùng trên các biến được giữ lại."""
        X_df = _as_numeric_dataframe(X, feature_names)
        y_arr = _as_numeric_vector(y)
        _check_xy_shape(X_df, y_arr)
        self.feature_names_in_ = X_df.columns.tolist()

        current_X = X_df

        if self.method in {"vif", "both"}:
            vif_result = backward_elimination_vif(
                current_X,
                y=None,
                threshold=self.vif_threshold,
                min_features=self.min_features,
                max_iter=self.max_iter,
                verbose=self.verbose,
            )
            current_X = vif_result["X_selected"]
            self.dropped_features_.extend(vif_result["dropped_features"])
            self.history_.extend(vif_result["history"])
            self.vif_table_ = vif_result["vif_table"]

        if self.method in {"pvalue", "both"}:
            pvalue_result = backward_elimination_pvalue(
                current_X,
                y_arr,
                alpha=self.alpha,
                min_features=self.min_features,
                max_iter=self.max_iter,
                verbose=self.verbose,
            )
            current_X = pvalue_result["X_selected"]
            self.dropped_features_.extend(pvalue_result["dropped_features"])
            self.history_.extend(pvalue_result["history"])

        self.selected_features_ = current_X.columns.tolist()
        self.model_ = _fit_ols(current_X, y_arr)
        y_hat = self.model_.predict(current_X.to_numpy())
        self.coef_table_ = ols_coefficient_table(current_X, y_arr)
        self.vif_table_ = run_vif_check(current_X, threshold=self.vif_threshold)
        self.metrics_ = _safe_model_metrics(y_arr, y_hat, p=len(self.selected_features_))
        return self

    def transform(self, X: pd.DataFrame | pd.Series | np.ndarray) -> pd.DataFrame:
        """Giữ lại đúng các cột đã được chọn trong fit()."""
        if self.selected_features_ is None:
            raise RuntimeError("Hãy gọi fit() trước khi transform().")
        names = None if isinstance(X, (pd.DataFrame, pd.Series)) else self.feature_names_in_
        X_df = _as_numeric_dataframe(X, feature_names=names)
        missing = [col for col in self.selected_features_ if col not in X_df.columns]
        if missing:
            raise ValueError(f"X thiếu các cột đã được chọn khi fit: {missing}")
        return X_df[self.selected_features_]

    def predict(self, X: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """Dự đoán y bằng OLS đã fit trên tập biến được chọn."""
        if self.model_ is None:
            raise RuntimeError("Hãy gọi fit() trước khi predict().")
        X_selected = self.transform(X)
        return self.model_.predict(X_selected.to_numpy())

    def summary(self) -> dict[str, Any]:
        """Trả về tóm tắt kết quả chọn biến để đưa vào notebook/báo cáo."""
        if self.selected_features_ is None:
            raise RuntimeError("Hãy gọi fit() trước khi summary().")
        return {
            "method": self.method,
            "alpha": self.alpha,
            "vif_threshold": self.vif_threshold,
            "selected_features": self.selected_features_,
            "dropped_features": self.dropped_features_,
            "history": self.history_,
            "coef_table": self.coef_table_,
            "vif_table": self.vif_table_,
            "metrics": self.metrics_,
        }


def select_features_ols(
    X: pd.DataFrame | pd.Series | np.ndarray,
    y: pd.Series | np.ndarray | list[float],
    method: str = "both",
    alpha: float = 0.05,
    vif_threshold: float = 10.0,
    min_features: int = 1,
    max_iter: int | None = None,
    feature_names: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Hàm tiện ích một dòng: chọn biến, fit OLS, và trả về summary.
    """
    selector = OLSFeatureSelector(
        method=method,
        alpha=alpha,
        vif_threshold=vif_threshold,
        min_features=min_features,
        max_iter=max_iter,
        verbose=verbose,
    ).fit(X, y, feature_names=feature_names)
    return selector.summary() | {"selector": selector}


# Alias để dễ gọi trong notebook theo nhiều cách đặt tên khác nhau.
ols_pvalue_selection = backward_elimination_pvalue
ols_vif_selection = backward_elimination_vif
vif_feature_selection = backward_elimination_vif
backward_elimination = backward_elimination_pvalue