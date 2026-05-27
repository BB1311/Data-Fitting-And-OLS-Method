"""
model_comparison.py
===================
Class ModelComparison — So sánh Ridge và Lasso trên Ames Housing (Phần 2).

Quy trình:
  1. Đọc & làm sạch dữ liệu (clean_data)
  2. Train/Test Split (80/20, stratified theo SalePrice)
  3. Tiền xử lý (DataPipeline)
  4. Loại bỏ đa cộng tuyến (VIF)
  5. Ridge & Lasso — chọn λ tối ưu qua k-fold CV
  6. Đánh giá trên test set: MAE, RMSE, R²
  7. Biểu đồ: CV curve, coefficient path, feature importance,
             actual vs predicted, residual analysis

Ghi chú: SalePrice đã log1p bên trong DataPipeline.
         Mọi metric tính trên không gian log.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ── sys.path ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_ROOT, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from part2.clean_data        import clean_data
from part2.data_pipeline     import DataPipeline
from part1.ols_implementation import compute_r2
from part1.ridge_lasso        import RidgeRegressor, LassoRegressor
from part2.cross_validation   import compare_models_cv
from part2.residual_analysis  import residual_plots


# ══════════════════════════════════════════════════════════════════════
# CLASS ModelComparison
# ══════════════════════════════════════════════════════════════════════
class ModelComparison:
    """
    So sánh Ridge và Lasso trên bộ dữ liệu Ames Housing.

    Parameters
    ----------
    data_path    : str   — đường dẫn tới AmesHousing.csv
    test_size    : float — tỉ lệ tập test (mặc định 0.20)
    cv_k         : int   — số fold CV (mặc định 5)
    vif_thresh   : float — ngưỡng VIF để loại cột (mặc định 10.0)
    lam_grid     : list  — danh sách λ cần thử (mặc định log-spaced 30 điểm)
    random_state : int   — seed reproducible (mặc định 42)
    figure_dir   : str   — thư mục lưu biểu đồ

    Workflow nhanh
    --------------
        mc = ModelComparison(data_path="data/AmesHousing.csv")
        mc.run()                   # chạy toàn bộ pipeline
        mc.summary()               # bảng so sánh
        mc.plot_all()              # vẽ tất cả biểu đồ
    """

    # ------------------------------------------------------------------
    # KHỞI TẠO
    # ------------------------------------------------------------------
    def __init__(
        self,
        data_path    : str   = os.path.join(_HERE, "data", "AmesHousing.csv"),
        test_size    : float = 0.20,
        cv_k         : int   = 5,
        vif_thresh   : float = 10.0,
        lam_grid     : list  = None,
        random_state : int   = 42,
        figure_dir   : str   = os.path.join(_HERE, "figures"),
    ):
        self.data_path    = data_path
        self.test_size    = test_size
        self.cv_k         = cv_k
        self.vif_thresh   = vif_thresh
        self.lam_grid     = lam_grid or list(np.logspace(-3, 4, 30))
        self.random_state = random_state
        self.figure_dir   = figure_dir
        os.makedirs(self.figure_dir, exist_ok=True)

        # Các thuộc tính được gán sau khi chạy run()
        self.pipe           : DataPipeline = None
        self.X_train        : pd.DataFrame = None
        self.X_test         : pd.DataFrame = None
        self.y_train        : np.ndarray   = None
        self.y_test         : np.ndarray   = None
        self.results        : dict         = {}   # {"ridge": {...}, "lasso": {...}}
        self.cv_full        : dict         = {}   # output đầy đủ từ compare_models_cv
        self._fitted        : bool         = False

    # ------------------------------------------------------------------
    # TIỆN ÍCH NỘI BỘ
    # ------------------------------------------------------------------
    @staticmethod
    def _mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _r2(y_true, y_pred):
        return float(compute_r2(y_true, y_pred))

    def _evaluate(self, y_true, y_pred, label="") -> dict:
        mae  = self._mae(y_true, y_pred)
        rmse = self._rmse(y_true, y_pred)
        r2   = self._r2(y_true, y_pred)
        if label:
            print(f"  {label:<35}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
        return {"mae": mae, "rmse": rmse, "r2": r2, "label": label}

    def _savefig(self, fig, filename: str):
        path = os.path.join(self.figure_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Saved] {path}")

    # ------------------------------------------------------------------
    # BƯỚC 1: ĐỌC & LÀM SẠCH
    # ------------------------------------------------------------------
    def load_and_clean(self) -> pd.DataFrame:
        """Đọc CSV gốc và chạy clean_data pipeline."""
        print("=" * 62)
        print("BƯỚC 1: ĐỌC & LÀM SẠCH DỮ LIỆU")
        print("=" * 62)
        df = clean_data(self.data_path)
        print(f"\n  Shape sau clean_data: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # BƯỚC 2: TRAIN/TEST SPLIT (stratified theo quantile SalePrice)
    # ------------------------------------------------------------------
    def split(self, X: pd.DataFrame, y: pd.Series):
        """
        Phân tầng đơn giản theo quantile y.
        Không dùng sklearn để giữ tinh thần tự cài đặt của đồ án.
        """
        print("\n" + "=" * 62)
        print("BƯỚC 2: TRAIN/TEST SPLIT (80/20, stratified)")
        print("=" * 62)
        rng    = np.random.default_rng(self.random_state)
        n_bins = 5
        bins   = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        test_idx = []
        for b in range(n_bins):
            group  = np.where(bins == b)[0]
            n_test = max(1, int(len(group) * self.test_size))
            test_idx.extend(rng.choice(group, size=n_test, replace=False).tolist())

        test_idx  = np.array(sorted(test_idx))
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

        X_arr, y_arr = X.values, y.values
        X_tr = pd.DataFrame(X_arr[train_idx], columns=X.columns)
        X_te = pd.DataFrame(X_arr[test_idx],  columns=X.columns)
        y_tr = pd.Series(y_arr[train_idx], name=y.name)
        y_te = pd.Series(y_arr[test_idx],  name=y.name)

        print(f"  Train: {X_tr.shape}  |  Test: {X_te.shape}")
        return X_tr, X_te, y_tr, y_te

    # ------------------------------------------------------------------
    # BƯỚC 3: TIỀN XỬ LÝ (DataPipeline)
    # ------------------------------------------------------------------
    def preprocess(self, X_train_raw, X_test_raw, y_train_raw, y_test_raw):
        """Fit pipeline trên train, transform cả hai tập."""
        print("\n" + "=" * 62)
        print("BƯỚC 3: TIỀN XỬ LÝ (DataPipeline)")
        print("=" * 62)
        self.pipe = DataPipeline(
            outlier_method      = "winsorize",
            outlier_threshold   = 0.02,
            encoding            = "auto",
            scale               = True,
            drop_first          = True,
            log_target          = True,   # log1p(SalePrice)
            engineer_features   = True,
            log_skewed_features = True,
            add_interactions    = True,
        )
        X_train, y_train = self.pipe.fit_transform(X_train_raw, y_train_raw)
        X_test           = self.pipe.transform(X_test_raw)
        y_test           = np.log1p(y_test_raw.values)   # đồng nhất với y_train
        y_train_np       = y_train.values if hasattr(y_train, "values") else np.array(y_train)

        print(f"\n  Shape: Train {X_train.shape} | Test {X_test.shape}")
        return X_train, X_test, y_train_np, y_test

    # ------------------------------------------------------------------
    # BƯỚC 4: LOẠI ĐA CỘNG TUYẾN (VIF)
    # ------------------------------------------------------------------
    def remove_multicollinearity(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Iterative VIF — loại cột cho đến khi VIF ≤ vif_thresh."""
        print("\n" + "=" * 62)
        print(f"BƯỚC 4: LOẠI ĐA CỘNG TUYẾN (VIF > {self.vif_thresh})")
        print("=" * 62)
        X_train_vif, dropped = self.pipe.drop_high_vif(X_train, threshold=self.vif_thresh)
        X_test_vif  = X_test.drop(columns=dropped, errors="ignore")
        print(f"  Shape sau VIF: Train {X_train_vif.shape} | Test {X_test_vif.shape}")
        return X_train_vif, X_test_vif

    # ------------------------------------------------------------------
    # BƯỚC 5: CHỌN λ QUA CV VÀ FIT MÔ HÌNH
    # ------------------------------------------------------------------
    def fit(self):
        """
        Chọn λ tốt nhất cho Ridge và Lasso qua k-fold CV,
        sau đó fit mô hình cuối trên toàn bộ tập train.

        Kết quả lưu vào self.results["ridge"] và self.results["lasso"].
        """
        self._check_fitted(require=False)
        print("\n" + "=" * 62)
        print(f"BƯỚC 5: CHỌN λ QUA {self.cv_k}-FOLD CV VÀ FIT MÔ HÌNH")
        print("=" * 62)

        X_tr = self.X_train.values
        y_tr = self.y_train.ravel()

        print(f"  Lưới λ: {len(self.lam_grid)} điểm "
              f"[{self.lam_grid[0]:.2e} … {self.lam_grid[-1]:.2e}]")
        self.cv_full = compare_models_cv(
            X_tr, y_tr,
            k            = self.cv_k,
            lam_grid     = self.lam_grid,
            random_state = self.random_state,
        )

        # ── Ridge ──────────────────────────────────────────────────
        best_lam_r = self.cv_full["best_lam_ridge"]
        print(f"\n  Ridge — λ* = {best_lam_r:.6f}  "
              f"(CV MSE = {self.cv_full['best_ridge']['cv_mse']:.4f})")
        ridge = RidgeRegressor(lam=best_lam_r, fit_intercept=True).fit(X_tr, y_tr)
        pred_r = ridge.predict(self.X_test.values)
        m_r = self._evaluate(self.y_test, pred_r, f"Ridge (λ={best_lam_r:.4f})")

        self.results["ridge"] = {
            "model"        : ridge,
            "lam"          : best_lam_r,
            "metrics"      : m_r,
            "cv_result"    : self.cv_full["best_ridge"],
            "all_cv"       : self.cv_full["ridge_results"],
            "feature_names": list(self.X_train.columns),
            "y_pred"       : pred_r,
        }

        # ── Lasso ──────────────────────────────────────────────────
        best_lam_l = self.cv_full["best_lam_lasso"]
        print(f"\n  Lasso — λ* = {best_lam_l:.6f}  "
              f"(CV MSE = {self.cv_full['best_lasso']['cv_mse']:.4f})")
        lasso = LassoRegressor(lam=best_lam_l, fit_intercept=True).fit(X_tr, y_tr)
        pred_l = lasso.predict(self.X_test.values)
        m_l = self._evaluate(self.y_test, pred_l, f"Lasso (λ={best_lam_l:.4f})")

        n_nonzero = int(np.sum(np.abs(lasso.coef_[1:]) > 1e-6))
        print(f"  Lasso: {n_nonzero}/{len(self.X_train.columns)} hệ số ≠ 0 "
              f"(sparsity {100*(1 - n_nonzero/len(self.X_train.columns)):.1f}%)")

        self.results["lasso"] = {
            "model"        : lasso,
            "lam"          : best_lam_l,
            "metrics"      : m_l,
            "cv_result"    : self.cv_full["best_lasso"],
            "all_cv"       : self.cv_full["lasso_results"],
            "feature_names": list(self.X_train.columns),
            "y_pred"       : pred_l,
            "n_nonzero"    : n_nonzero,
        }

        self._fitted = True

    # ------------------------------------------------------------------
    # RUN — chạy toàn bộ pipeline
    # ------------------------------------------------------------------
    def run(self):
        """
        Chạy đầy đủ pipeline:
          load_and_clean → split → preprocess → remove_multicollinearity → fit
        """
        sns.set_theme(style="whitegrid")

        df = self.load_and_clean()
        y  = df["SalePrice"]
        X  = df.drop(columns=["SalePrice"])

        X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = self.split(X, y)

        X_tr, X_te, y_tr, y_te = self.preprocess(
            X_tr_raw, X_te_raw, y_tr_raw, y_te_raw
        )
        X_tr_vif, X_te_vif = self.remove_multicollinearity(X_tr, X_te)

        self.X_train = X_tr_vif
        self.X_test  = X_te_vif
        self.y_train = y_tr
        self.y_test  = y_te

        self.fit()
        return self

    # ------------------------------------------------------------------
    # BẢNG TỔNG HỢP
    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """In và trả về bảng so sánh MAE / RMSE / R² trên test set."""
        self._check_fitted()
        rows = []
        for key in ["ridge", "lasso"]:
            r = self.results[key]
            row = {
                "Mô hình": f"{key.capitalize()} (λ={r['lam']:.4f})",
                "λ*"     : round(r["lam"], 6),
                "CV MSE" : round(r["cv_result"]["cv_mse"],  4),
                "MAE"    : round(r["metrics"]["mae"],  4),
                "RMSE"   : round(r["metrics"]["rmse"], 4),
                "R²"     : round(r["metrics"]["r2"],   4),
            }
            if key == "lasso":
                row["Hệ số ≠ 0"] = r["n_nonzero"]
            rows.append(row)

        df_cmp = pd.DataFrame(rows).set_index("Mô hình")
        print("\n" + "═" * 70)
        print("BẢNG SO SÁNH RIDGE vs LASSO TRÊN TẬP TEST")
        print("═" * 70)
        print(df_cmp.to_string())
        print("═" * 70)

        best = min(["ridge", "lasso"],
                   key=lambda k: self.results[k]["metrics"]["rmse"])
        print(f"\n✓ Mô hình tốt nhất (RMSE thấp nhất): {best.capitalize()}")
        self._best_key = best
        return df_cmp

    # ------------------------------------------------------------------
    # BIỂU ĐỒ
    # ------------------------------------------------------------------
    def plot_cv_lambda(self, save: bool = True) -> plt.Figure:
        """CV MSE theo λ cho Ridge và Lasso (2 panel)."""
        self._check_fitted()
        lam_grid  = np.array(self.lam_grid)
        ridge_mse = [r["cv_mse"] for r in self.results["ridge"]["all_cv"]]
        lasso_mse = [r["cv_mse"] for r in self.results["lasso"]["all_cv"]]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Cross-Validation MSE theo λ", fontsize=14, fontweight="bold")

        cfg = [
            (axes[0], ridge_mse, self.results["ridge"]["lam"], "Ridge", "royalblue"),
            (axes[1], lasso_mse, self.results["lasso"]["lam"], "Lasso", "firebrick"),
        ]
        for ax, mse_vals, lam_best, title, color in cfg:
            ax.semilogx(lam_grid, mse_vals, color=color, lw=2,
                        marker="o", markersize=4, label="CV MSE")
            ax.axvline(lam_best, color="black", linestyle="--",
                       label=f"λ* = {lam_best:.4f}")
            ax.set_xlabel("λ (log scale)", fontsize=11)
            ax.set_ylabel("CV MSE", fontsize=11)
            ax.set_title(f"{title} Regression", fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.4)

        plt.tight_layout()
        if save:
            self._savefig(fig, "cv_lambda_curve.png")
        return fig

    def plot_lasso_coef_path(self, top_n: int = 10, save: bool = True) -> plt.Figure:
        """
        Lasso coefficient path: hệ số từng feature theo λ.
        Trực quan hóa quá trình các hệ số co dần về 0.
        """
        self._check_fitted()
        feat_names = self.results["lasso"]["feature_names"]
        X_tr = self.X_train.values
        y_tr = self.y_train.ravel()

        coef_matrix = []
        for lam in self.lam_grid:
            m = LassoRegressor(lam=lam, fit_intercept=True).fit(X_tr, y_tr)
            c = m.coef_[1:] if len(m.coef_) == len(feat_names) + 1 else m.coef_
            coef_matrix.append(c)
        coef_matrix = np.array(coef_matrix)   # (n_lam, n_feat)

        top_idx = np.argsort(np.abs(coef_matrix[0]))[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in top_idx:
            ax.semilogx(self.lam_grid, coef_matrix[:, i],
                        label=feat_names[i], lw=1.5)
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.axvline(self.results["lasso"]["lam"], color="grey",
                   linestyle=":", lw=1.2, label=f"λ* = {self.results['lasso']['lam']:.4f}")
        ax.set_xlabel("λ (log scale)", fontsize=11)
        ax.set_ylabel("Hệ số hồi quy", fontsize=11)
        ax.set_title(f"Lasso Coefficient Path — Top {top_n} features",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.19, 0.5))
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        if save:
            self._savefig(fig, "lasso_coef_path.png")
        return fig

    def plot_ridge_trace(self, top_n: int = 10, save: bool = True) -> plt.Figure:
        """
        Ridge trace: hệ số shrinkage theo λ.
        Khác Lasso — hệ số co dần về 0 nhưng không bằng 0 hẳn.
        """
        self._check_fitted()
        feat_names = self.results["ridge"]["feature_names"]
        X_tr = self.X_train.values
        y_tr = self.y_train.ravel()

        coef_matrix = []
        for lam in self.lam_grid:
            m = RidgeRegressor(lam=lam, fit_intercept=True).fit(X_tr, y_tr)
            c = m.coef_[1:] if len(m.coef_) == len(feat_names) + 1 else m.coef_
            coef_matrix.append(c)
        coef_matrix = np.array(coef_matrix)

        top_idx = np.argsort(np.abs(coef_matrix[0]))[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in top_idx:
            ax.semilogx(self.lam_grid, coef_matrix[:, i],
                        label=feat_names[i], lw=1.5)
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.axvline(self.results["ridge"]["lam"], color="grey",
                   linestyle=":", lw=1.2, label=f"λ* = {self.results['ridge']['lam']:.4f}")
        ax.set_xlabel("λ (log scale)", fontsize=11)
        ax.set_ylabel("Hệ số hồi quy", fontsize=11)
        ax.set_title(f"Ridge Trace — Top {top_n} features",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.19, 0.5))
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        if save:
            self._savefig(fig, "ridge_trace.png")
        return fig

    def plot_feature_importance(
        self, model_key: str = None, top_n: int = 20, save: bool = True
    ) -> plt.Figure:
        """
        Horizontal bar chart hệ số hồi quy (đã chuẩn hóa) của Ridge hoặc Lasso.

        Parameters
        ----------
        model_key : "ridge" | "lasso" | None — None → dùng mô hình tốt nhất
        """
        self._check_fitted()
        key   = model_key or getattr(self, "_best_key", "ridge")
        model = self.results[key]["model"]
        feat  = self.results[key]["feature_names"]

        coef = model.coef_
        if len(coef) == len(feat) + 1:
            coef = coef[1:]   # bỏ intercept

        df_coef = (
            pd.DataFrame({"feature": feat, "coef": coef})
            .assign(abs_coef=lambda d: d["coef"].abs())
            .nlargest(top_n, "abs_coef")
            .sort_values("coef")
        )

        colors = ["firebrick" if c < 0 else "steelblue" for c in df_coef["coef"]]
        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
        ax.barh(df_coef["feature"], df_coef["coef"],
                color=colors, edgecolor="k", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Hệ số hồi quy (đã chuẩn hóa z-score)", fontsize=11)
        ax.set_title(
            f"Top {top_n} Feature Importance — {key.capitalize()} "
            f"(λ={self.results[key]['lam']:.4f})",
            fontsize=12, fontweight="bold",
        )
        ax.grid(True, axis="x", alpha=0.4)
        plt.tight_layout()
        if save:
            self._savefig(fig, f"feature_importance_{key}.png")
        return fig

    def plot_actual_vs_predicted(
        self, model_key: str = None, save: bool = True
    ) -> plt.Figure:
        """Scatter: giá trị thực vs dự đoán (log SalePrice)."""
        self._check_fitted()
        key    = model_key or getattr(self, "_best_key", "ridge")
        y_pred = self.results[key]["y_pred"]
        y_true = self.y_test
        label  = self.results[key]["metrics"]["label"]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k",
                   linewidths=0.3, color="steelblue", s=25)
        lims = [min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="y = ŷ")
        ax.set_xlabel("Giá trị thực (log SalePrice)", fontsize=11)
        ax.set_ylabel("Giá trị dự đoán (log SalePrice)", fontsize=11)
        ax.set_title(f"Actual vs Predicted — {label}",
                     fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        if save:
            self._savefig(fig, f"actual_vs_predicted_{key}.png")
        return fig

    def plot_residuals(self, model_key: str = None, save: bool = True):
        """4 biểu đồ phân tích phần dư (dùng residual_analysis.py từ Part 1)."""
        self._check_fitted()
        key   = model_key or getattr(self, "_best_key", "ridge")
        model = self.results[key]["model"]

        try:
            fig, axes = residual_plots(
                self.X_test.values, self.y_test, model.coef_
            )
            if save:
                self._savefig(fig, f"residual_analysis_{key}.png")
            return fig, axes
        except Exception as e:
            print(f"  [CẢNH BÁO] Không vẽ được residual analysis: {e}")
            return None, None

    def plot_all(self, save: bool = True):
        """Vẽ toàn bộ biểu đồ: CV curve, traces, importance, scatter, residuals."""
        self._check_fitted()
        print("\n" + "=" * 62)
        print("VẼ BIỂU ĐỒ")
        print("=" * 62)
        self.plot_cv_lambda(save=save)
        self.plot_ridge_trace(save=save)
        self.plot_lasso_coef_path(save=save)
        self.plot_feature_importance("ridge", save=save)
        self.plot_feature_importance("lasso", save=save)
        self.plot_actual_vs_predicted("ridge", save=save)
        self.plot_actual_vs_predicted("lasso", save=save)
        self.plot_residuals(save=save)
        plt.close("all")
        print(f"\n  Tất cả biểu đồ đã lưu vào: {self.figure_dir}")

    # ------------------------------------------------------------------
    # KIỂM TRA TRẠNG THÁI
    # ------------------------------------------------------------------
    def _check_fitted(self, require: bool = True):
        if require and not self._fitted:
            raise RuntimeError("Gọi .run() hoặc .fit() trước.")

    # ------------------------------------------------------------------
    # REPR
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (f"ModelComparison(cv_k={self.cv_k}, vif_thresh={self.vif_thresh}, "
                f"lam_grid=[{len(self.lam_grid)} pts], status={status})")


