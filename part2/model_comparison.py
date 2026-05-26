"""
model_comparison.py
===================
So sánh các mô hình hồi quy có kiểm soát (Regularization) trên bộ dữ liệu Ames Housing.

Pipeline tập trung vào:
  1. Đọc & làm sạch dữ liệu (clean_data)
  2. Train/Test Split (80/20, phân tầng theo giá trị SalePrice)
  3. Tiền xử lý (DataPipeline: outlier, encoding, z-score scale)
  4. Loại bỏ đa cộng tuyến (VIF)
  5. Xây dựng & tinh chỉnh siêu tham số cho 2 mô hình:
       - Ridge Regression (chọn λ tối ưu qua k-fold CV)
       - Lasso Regression (chọn λ tối ưu qua k-fold CV)
  6. Đánh giá hiệu suất trên tập test: MAE, RMSE, R²
  7. Trực quan hóa: Feature Importance, Lasso Coef Path, Residual Analysis

Ghi chú: Target SalePrice đã được chuyển sang log1p trong DataPipeline 
         để giảm độ lệch (skewness).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Tắt cảnh báo để output console sạch sẽ
warnings.filterwarnings("ignore")

# ── THIẾT LẬP ĐƯỜNG DẪN ĐỂ IMPORT MODULE TỪ PART 1 VÀ PART 2 ─────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_ROOT, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import các hàm từ project (đảm bảo file part1/ và part2/ có sẵn)
from part2.clean_data         import clean_data
from part2.data_pipeline      import DataPipeline
from part1.ols_implementation import compute_r2
from part1.ridge_lasso        import RidgeRegressor, LassoRegressor
from part1.cross_validation   import compare_models_cv
from part1.residual_analysis  import residual_plots

# ══════════════════════════════════════════════════════════════════════
# 0. HẰNG SỐ CẤU HÌNH & HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════
DATA_PATH    = os.path.join(_HERE, "data", "AmesHousing.csv")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_K         = 5            # Số fold để Cross-Validation
VIF_THRESH   = 10.0         # Ngưỡng VIF để loại đa cộng tuyến

# Lưới không gian tìm kiếm siêu tham số λ (30 điểm log-spaced từ 10^-3 đến 10^4)
LAM_GRID = list(np.logspace(-3, 4, 30))

# Thư mục lưu biểu đồ
FIGURE_DIR = os.path.join(_HERE, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 1. HÀM TIỆN ÍCH ĐÁNH GIÁ & CHIA DỮ LIỆU
# ══════════════════════════════════════════════════════════════════════
def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _r2(y_true, y_pred):
    return float(compute_r2(y_true, y_pred))

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Tính toán và in các metric (MAE, RMSE, R²)."""
    mae  = _mae(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)
    r2   = _r2(y_true, y_pred)
    if label:
        print(f"  {label:<25} MAE = {mae:.4f} | RMSE = {rmse:.4f} | R² = {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "label": label}

def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Tự code phân tầng (Stratified Split) dựa trên phân vị của target (y)
    giúp phân phối giá nhà ở tập train/test tương đồng nhau.
    """
    rng = np.random.default_rng(random_state)
    n_bins = 5
    bins   = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    
    test_idx = []
    for b in range(n_bins):
        group = np.where(bins == b)[0]
        n_test = max(1, int(len(group) * test_size))
        chosen = rng.choice(group, size=n_test, replace=False)
        test_idx.extend(chosen.tolist())

    test_idx  = np.array(sorted(test_idx))
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series)    else y

    return (
        pd.DataFrame(X_arr[train_idx], columns=X.columns),
        pd.DataFrame(X_arr[test_idx],  columns=X.columns),
        pd.Series(y_arr[train_idx], name=y.name),
        pd.Series(y_arr[test_idx],  name=y.name),
    )


# ══════════════════════════════════════════════════════════════════════
# 2. XÂY DỰNG MÔ HÌNH RIDGE & LASSO
# ══════════════════════════════════════════════════════════════════════
def build_regularized_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict:
    """
    Chạy Cross-Validation để tìm λ tốt nhất cho Ridge và Lasso.
    Huấn luyện mô hình cuối cùng trên toàn bộ tập train và đánh giá trên tập test.
    """
    results = {}
    X_tr = X_train.values
    X_te = X_test.values
    y_tr = y_train.ravel() if hasattr(y_train, "ravel") else np.array(y_train)
    y_te = y_test.ravel()  if hasattr(y_test,  "ravel") else np.array(y_test)

    print("\n── 5-FOLD CROSS VALIDATION (TÌM λ TỐI ƯU) ───────────────────────")
    print(f"  Đang dò tìm trên lưới {len(LAM_GRID)} giá trị λ...")
    
    cv_result = compare_models_cv(
        X_tr, y_tr,
        k=CV_K,
        lam_grid=LAM_GRID,
        random_state=RANDOM_STATE,
    )

    # ── MÔ HÌNH 1: RIDGE REGRESSION
    best_lam_ridge = cv_result["best_lam_ridge"]
    print(f"\n  [Ridge] Đã tìm thấy λ* = {best_lam_ridge:.4f} (CV MSE: {cv_result['best_ridge']['cv_mse']:.4f})")
    
    ridge = RidgeRegressor(lam=best_lam_ridge, fit_intercept=True).fit(X_tr, y_tr)
    pred_r = ridge.predict(X_te)
    m_ridge = evaluate(y_te, pred_r, f"Ridge (λ={best_lam_ridge:.4f})")

    results["ridge"] = {
        "model"        : ridge,
        "metrics"      : m_ridge,
        "cv_result"    : cv_result["best_ridge"],
        "lam"          : best_lam_ridge,
        "feature_names": list(X_train.columns),
        "all_cv"       : cv_result["ridge_results"],
    }

    # ── MÔ HÌNH 2: LASSO REGRESSION
    best_lam_lasso = cv_result["best_lam_lasso"]
    print(f"\n  [Lasso] Đã tìm thấy λ* = {best_lam_lasso:.4f} (CV MSE: {cv_result['best_lasso']['cv_mse']:.4f})")
    
    lasso = LassoRegressor(lam=best_lam_lasso, fit_intercept=True).fit(X_tr, y_tr)
    pred_l = lasso.predict(X_te)
    m_lasso = evaluate(y_te, pred_l, f"Lasso (λ={best_lam_lasso:.4f})")
    
    # Tính số biến thực sự được Lasso giữ lại (hệ số khác 0)
    n_nonzero = int(np.sum(np.abs(lasso.coef_[1:]) > 1e-6))
    print(f"  [Lasso] Thu gọn mô hình: giữ lại {n_nonzero} / {len(X_train.columns)} biến.")

    results["lasso"] = {
        "model"        : lasso,
        "metrics"      : m_lasso,
        "cv_result"    : cv_result["best_lasso"],
        "lam"          : best_lam_lasso,
        "feature_names": list(X_train.columns),
        "all_cv"       : cv_result["lasso_results"],
    }

    results["cv_full"] = cv_result
    results["lam_grid"] = LAM_GRID

    return results


# ══════════════════════════════════════════════════════════════════════
# 3. TỔNG HỢP VÀ CHỌN MÔ HÌNH TỐT NHẤT
# ══════════════════════════════════════════════════════════════════════
def summary_table(results: dict) -> pd.DataFrame:
    """Lập bảng so sánh 2 mô hình trên tập test và xác định mô hình vô địch."""
    rows = []
    keys = ["ridge", "lasso"]
    display_names = {
        "ridge": f"Ridge (λ={results['ridge']['lam']:.4f})",
        "lasso": f"Lasso (λ={results['lasso']['lam']:.4f})",
    }
    
    for k in keys:
        m = results[k]["metrics"]
        rows.append({
            "Mô hình" : display_names[k],
            "MAE"     : round(m["mae"],  4),
            "RMSE"    : round(m["rmse"], 4),
            "R²"      : round(m["r2"],   4),
        })

    df_cmp = pd.DataFrame(rows).set_index("Mô hình")

    print("\n" + "═" * 60)
    print(" BẢNG SO SÁNH HIỆU SUẤT TRÊN TẬP TEST")
    print("═" * 60)
    print(df_cmp.to_string())
    print("═" * 60)

    # Chọn mô hình có RMSE nhỏ nhất
    best_key = min(keys, key=lambda k: results[k]["metrics"]["rmse"])
    print(f"\n🏆 MÔ HÌNH TỐT NHẤT: {display_names[best_key]}")
    
    return df_cmp, best_key


# ══════════════════════════════════════════════════════════════════════
# 4. HÀM TRỰC QUAN HÓA (VISUALIZATION)
# ══════════════════════════════════════════════════════════════════════
def plot_cv_lambda(results: dict, save: bool = True):
    """Vẽ đường cong Cross-Validation MSE theo sự thay đổi của λ."""
    lam_grid  = np.array(results["lam_grid"])
    ridge_mse = [r["cv_mse"] for r in results["ridge"]["all_cv"]]
    lasso_mse = [r["cv_mse"] for r in results["lasso"]["all_cv"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-Validation MSE vs Lambda (λ)", fontsize=14, fontweight="bold")

    configs = zip(
        axes, 
        [ridge_mse, lasso_mse], 
        [results["ridge"]["lam"], results["lasso"]["lam"]],
        ["Ridge Regression", "Lasso Regression"],
        ["royalblue", "firebrick"]
    )

    for ax, mse_vals, lam_best, title, color in configs:
        ax.semilogx(lam_grid, mse_vals, color=color, lw=2, marker="o", markersize=4, label="CV MSE")
        ax.axvline(lam_best, color="black", linestyle="--", label=f"λ tối ưu = {lam_best:.4f}")
        ax.set_xlabel("Giá trị λ (log scale)", fontsize=11)
        ax.set_ylabel("Mean Squared Error (CV)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURE_DIR, "cv_lambda_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Đã lưu] {path}")

def plot_feature_importance(results: dict, best_key: str, top_n: int = 20, save: bool = True):
    """Trực quan hóa top các biến có hệ số hồi quy lớn nhất (tác động mạnh nhất)."""
    model = results[best_key]["model"]
    feat  = results[best_key]["feature_names"]

    # Bỏ phần tử đầu tiên nếu nó là hệ số tự do (intercept)
    coef = model.coef_[1:] if len(model.coef_) == len(feat) + 1 else model.coef_

    df_coef = pd.DataFrame({"feature": feat, "coef": coef})
    df_coef["abs_coef"] = df_coef["coef"].abs()
    
    # Lấy top N biến ảnh hưởng lớn nhất
    df_top = df_coef.nlargest(top_n, "abs_coef").sort_values("coef")

    colors = ["firebrick" if c < 0 else "steelblue" for c in df_top["coef"]]
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    
    ax.barh(df_top["feature"], df_top["coef"], color=colors, edgecolor="k", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Trọng số (Hệ số hồi quy đã chuẩn hóa)", fontsize=11)
    ax.set_title(
        f"Top {top_n} Biến Quan Trọng Nhất — {results[best_key]['metrics']['label']}",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="x", alpha=0.4)
    plt.tight_layout()
    
    if save:
        path = os.path.join(FIGURE_DIR, "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Đã lưu] {path}")

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, label: str = "", save: bool = True):
    """Vẽ biểu đồ phân tán so sánh giá dự đoán và giá trị thực tế."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="white", linewidths=0.5, color="steelblue", s=30)
    
    # Đường chéo y = x
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Đường y = ŷ (Hoàn hảo)")
    
    ax.set_xlabel("Giá trị thực tế (log1p SalePrice)", fontsize=11)
    ax.set_ylabel("Giá trị dự đoán (log1p SalePrice)", fontsize=11)
    ax.set_title(f"Thực tế vs Dự đoán — {label}", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    
    if save:
        path = os.path.join(FIGURE_DIR, "actual_vs_predicted.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Đã lưu] {path}")

def plot_lasso_coef_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    lam_grid: list,
    top_n: int = 10,
    save: bool = True,
):
    """Vẽ quá trình các hệ số Lasso bị co rút về 0 khi λ tăng."""
    coef_matrix = []
    for lam in lam_grid:
        m = LassoRegressor(lam=lam, fit_intercept=True).fit(X_train, y_train)
        c = m.coef_[1:] if len(m.coef_) == len(feature_names) + 1 else m.coef_
        coef_matrix.append(c)

    coef_matrix = np.array(coef_matrix) # shape (n_lam, n_feat)

    # Chọn top biến có hệ số tuyệt đối lớn nhất ở mức λ nhỏ (ít penalty)
    abs_at_small = np.abs(coef_matrix[0])
    top_idx = np.argsort(abs_at_small)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in top_idx:
        ax.semilogx(lam_grid, coef_matrix[:, i], label=feature_names[i], lw=1.8)
    
    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.set_xlabel("Giá trị λ (log scale)", fontsize=11)
    ax.set_ylabel("Hệ số hồi quy", fontsize=11)
    ax.set_title(f"Đường co rút hệ số Lasso (Lasso Path) — Top {top_n} biến", fontsize=12, fontweight="bold")
    
    # Dời legend ra ngoài để không che biểu đồ
    ax.legend(fontsize=9, loc="center right", bbox_to_anchor=(1.25, 0.5))
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    
    if save:
        path = os.path.join(FIGURE_DIR, "lasso_coef_path.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Đã lưu] {path}")


# ══════════════════════════════════════════════════════════════════════
# 5. PIPELINE CHÍNH (MAIN EXECUTION)
# ══════════════════════════════════════════════════════════════════════
def run_model_comparison(data_path: str = DATA_PATH) -> dict:
    """Chạy toàn bộ pipeline xử lý, huấn luyện và đánh giá mô hình."""
    sns.set_theme(style="whitegrid")

    print("=" * 60)
    print(" BƯỚC 1: ĐỌC VÀ LÀM SẠCH DỮ LIỆU (CLEAN DATA)")
    print("=" * 60)
    df = clean_data(data_path)
    print(f"  ✓ Kích thước sau khi clean: {df.shape}")

    print("\n" + "=" * 60)
    print(" BƯỚC 2: TRAIN/TEST SPLIT (80/20 STRATIFIED)")
    print("=" * 60)
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split_stratified(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  ✓ Kích thước Train: {X_train_raw.shape}")
    print(f"  ✓ Kích thước Test : {X_test_raw.shape}")

    print("\n" + "=" * 60)
    print(" BƯỚC 3: TIỀN XỬ LÝ (DATA PIPELINE)")
    print("=" * 60)
    pipe = DataPipeline(
        outlier_method       = "winsorize",
        outlier_threshold    = 0.02,   # Chặn giá trị ngoại lai ở biên 2%
        encoding             = "auto",
        scale                = True,
        drop_first           = True,
        log_target           = True,   # Lấy log1p cho y để phân phối chuẩn hơn
        engineer_features    = True,
        log_skewed_features  = True,
        add_interactions     = True,
    )
    
    X_train, y_train = pipe.fit_transform(X_train_raw, y_train_raw)
    X_test           = pipe.transform(X_test_raw)

    # Đưa y_test về không gian log để đánh giá đồng nhất với mô hình
    y_test = np.log1p(y_test_raw.values)
    y_train_np = y_train.values if hasattr(y_train, "values") else np.array(y_train)

    print(f"  ✓ Output Feature Matrix: Train {X_train.shape} | Test {X_test.shape}")

    print("\n" + "=" * 60)
    print(f" BƯỚC 4: LOẠI BỎ ĐA CỘNG TUYẾN (NGƯỠNG VIF = {VIF_THRESH})")
    print("=" * 60)
    X_train_vif, dropped_cols = pipe.drop_high_vif(X_train, threshold=VIF_THRESH)
    X_test_vif = X_test.drop(columns=dropped_cols, errors="ignore")
    print(f"  ✓ Output sau VIF: Train {X_train_vif.shape} | Test {X_test_vif.shape}")

    print("\n" + "=" * 60)
    print(" BƯỚC 5: HUẤN LUYỆN RIDGE & LASSO")
    print("=" * 60)
    results = build_regularized_models(X_train_vif, y_train_np, X_test_vif, y_test)

    print("\n" + "=" * 60)
    print(" BƯỚC 6 & 7: KẾT QUẢ VÀ TRỰC QUAN HÓA")
    print("=" * 60)
    
    # 6.1 Lập bảng so sánh
    summary_df, best_key = summary_table(results)

    # 6.2 Vẽ biểu đồ
    plot_cv_lambda(results)
    
    best_model  = results[best_key]["model"]
    X_te_best   = X_test_vif.values
    y_pred_best = best_model.predict(X_te_best)
    
    plot_actual_vs_predicted(y_test, y_pred_best, results[best_key]["metrics"]["label"])
    plot_feature_importance(results, best_key, top_n=20)
    plot_lasso_coef_path(X_train_vif.values, y_train_np, list(X_train_vif.columns), LAM_GRID, top_n=10)

    # 6.3 Phân tích phần dư cho mô hình xuất sắc nhất
    print("\n  Đang vẽ Residual Analysis cho mô hình vô địch...")
    try:
        fig_res, _ = residual_plots(X_te_best, y_test, best_model.coef_)
        path_res = os.path.join(FIGURE_DIR, "residual_analysis_best.png")
        fig_res.savefig(path_res, dpi=150, bbox_inches="tight")
        print(f"  [Đã lưu] {path_res}")
    except Exception as e:
        print(f"  [Cảnh báo] Lỗi khi vẽ residual analysis từ module part2: {e}")

    # Giải phóng bộ nhớ đồ họa
    plt.close("all")

    print("\n" + "★" * 60)
    print(" HOÀN TẤT PIPELINE SO SÁNH!")
    print(f" Toàn bộ biểu đồ phân tích đã được lưu vào: {FIGURE_DIR}")
    print("★" * 60)

    return {
        "results"   : results,
        "summary_df": summary_df,
        "best_key"  : best_key,
        "X_train"   : X_train_vif,
        "X_test"    : X_test_vif,
        "y_train"   : y_train_np,
        "y_test"    : y_test,
        "pipe"      : pipe,
    }

# ══════════════════════════════════════════════════════════════════════
# ĐIỂM BẮT ĐẦU CHẠY SCRIPT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    output = run_model_comparison()