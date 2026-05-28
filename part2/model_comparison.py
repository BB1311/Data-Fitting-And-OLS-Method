"""
ols_full.py
===========
OLS đầy đủ (tất cả biến sau pipeline + VIF) cho Phần 2.
"""

import numpy as np
import pandas as pd
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_ROOT, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from part1.ols_implementation import OLSRegressor, coef_inference, compute_r2
from part1.cross_validation import _mae, _rmse


class OLSFull:
    """
    OLS hồi quy đầy đủ — sử dụng toàn bộ biến sau pipeline và VIF.

    Interface
    ---------
    model = OLSFull().fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    result  = model.to_result(X_test, y_test)

    Parameters
    ----------
    fit_intercept : bool — có thêm hệ số chặn hay không (mặc định True)
    verbose       : bool — in metric khi gọi evaluate() (mặc định True);
                           to_result() luôn tắt verbose để tránh in lặp khi
                           chạy trong pipeline so sánh mô hình
    """

    def __init__(self, fit_intercept: bool = True, verbose: bool = True):
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self._model: OLSRegressor | None = None
        self._feature_names: list = []

    # ── FIT ───────────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "OLSFull":
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X, dtype=float)
            n_cols = X_arr.shape[1] if X_arr.ndim == 2 else 1
            # fallback tên biến khi input là ndarray — không có column names
            # đặt x0, x1, ... để coef_table() vẫn hoạt động nhưng kém readable
            # khuyến nghị luôn truyền DataFrame để giữ tên biến có nghĩa
            self._feature_names = [f"x{i}" for i in range(n_cols)]

        y_arr = np.asarray(y, dtype=float).ravel()
        self._model = OLSRegressor(fit_intercept=self.fit_intercept).fit(X_arr, y_arr)
        return self

    # ── PREDICT ───────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        return self._model.predict(X_arr)

    # ── EVALUATE ──────────────────────────────────────────────────────
    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        verbose: bool | None = None,
    ) -> dict:
        # dùng instance default nếu không truyền vào
        if verbose is None:
            verbose = self.verbose
        self._check_fitted()
        y_arr  = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        metrics = {
            "mae"  : _mae(y_arr, y_pred),
            "rmse" : _rmse(y_arr, y_pred),
            "r2"   : compute_r2(y_arr, y_pred),
            "label": "OLS Full",
        }
        if verbose:
            print(f"  {'OLS Full':<25} MAE = {metrics['mae']:.4f} | "
                  f"RMSE = {metrics['rmse']:.4f} | R² = {metrics['r2']:.4f}")
        return metrics

    # ── COEF TABLE ────────────────────────────────────────────────────
    def coef_table(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Bảng hệ số với SE, t-stat, p-value."""
        self._check_fitted()
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()

        inf = coef_inference(
            X_arr, y_arr,               # truyền X gốc (không có cột 1)
            self._model.beta_hat,       # beta_hat có p+1 phần tử (bao gồm intercept)
            self._model.sigma2_hat,
            verbose=False,
            # coef_inference tự detect mismatch và thêm cột 1 vào X_arr
            # xem ols_implementation.py: if X.shape[1] == len(beta_hat) - 1
        )

        # Tên cột nhất quán với số lượng hệ số thực tế
        names = (["Intercept"] if self.fit_intercept else []) + self._feature_names
        # Đảm bảo độ dài khớp (phòng thủ)
        n_coef = len(self._model.beta_hat)
        if len(names) != n_coef:
            raise ValueError(
                f"Mismatch tên biến: expected {n_coef} names, got {len(names)}. "
                f"fit_intercept={self.fit_intercept}, n_features={len(self._feature_names)}"
            )

        return pd.DataFrame({
            "feature"   : names,
            "coef"      : self._model.beta_hat,
            "std_error" : inf["se"],
            "t_stat"    : inf["t_stats"],
            "p_value"   : inf["p_values"],
            "ci_lower"  : inf["ci_lower"],
            "ci_upper"  : inf["ci_upper"],
        })

    # ── TO_RESULT ─────────────────────────────────────────────────────
    def to_result(self, X_test: pd.DataFrame | np.ndarray, y_test: np.ndarray) -> dict:
        self._check_fitted()
        return {
            "model"        : self._model,
            "metrics"      : self.evaluate(X_test, y_test, verbose=False),
            "feature_names": self._feature_names,
            "lam": None,
        }

    # ── HELPER ────────────────────────────────────────────────────────
    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("Gọi fit() trước.")

    @property
    def coef_(self) -> np.ndarray:
        self._check_fitted()
        return self._model.beta_hat

    @property
    def n_features(self) -> int:
        return len(self._feature_names)