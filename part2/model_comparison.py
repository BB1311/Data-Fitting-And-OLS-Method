import os
import sys

import numpy as np
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from part1.ols_implementation import OLSRegressor, coef_inference, model_metrics
from part2.data_pipeline import run_vif_check


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


def ols_coefficient_table(X, y, feature_names=None):
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
    X,
    y,
    alpha=0.05,
    min_features=1,
    max_iter=None,
    feature_names=None,
    verbose=True,
):
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
    dropped = []
    history = []

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
    X,
    y=None,
    threshold=10.0,
    min_features=1,
    max_iter=None,
    feature_names=None,
    verbose=True,
):
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
    dropped = []
    history = []

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

    result = {
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


def select_features_ols(
    X,
    y,
    method="both",
    alpha=0.05,
    vif_threshold=10.0,
    min_features=1,
    max_iter=None,
    feature_names=None,
    verbose=True,
):
    """
    Hàm tiện ích một dòng để chọn biến OLS.

    method:
      - "pvalue": chỉ loại theo p-value.
      - "vif": chỉ loại theo VIF.
      - "both": loại VIF trước, sau đó loại p-value.
    """
    X_df = _as_numeric_dataframe(X, feature_names)
    y_arr = _as_numeric_vector(y)
    _check_xy_shape(X_df, y_arr)

    if method == "pvalue":
        return backward_elimination_pvalue(
            X_df,
            y_arr,
            alpha=alpha,
            min_features=min_features,
            max_iter=max_iter,
            verbose=verbose,
        )

    if method == "vif":
        return backward_elimination_vif(
            X_df,
            y_arr,
            threshold=vif_threshold,
            min_features=min_features,
            max_iter=max_iter,
            verbose=verbose,
        )

    if method == "both":
        vif_result = backward_elimination_vif(
            X_df,
            y=None,
            threshold=vif_threshold,
            min_features=min_features,
            max_iter=max_iter,
            verbose=verbose,
        )
        pvalue_result = backward_elimination_pvalue(
            vif_result["X_selected"],
            y_arr,
            alpha=alpha,
            min_features=min_features,
            max_iter=max_iter,
            verbose=verbose,
        )

        pvalue_result["dropped_features"] = (
            vif_result["dropped_features"] + pvalue_result["dropped_features"]
        )
        pvalue_result["history"] = vif_result["history"] + pvalue_result["history"]
        pvalue_result["vif_table"] = run_vif_check(
            pvalue_result["X_selected"],
            threshold=vif_threshold,
        )
        return pvalue_result

    raise ValueError("method phải là 'pvalue', 'vif', hoặc 'both'.")