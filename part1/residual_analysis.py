import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
from typing import Tuple, List

def compute_leverage_and_cooks(
    X_design: np.ndarray, 
    y: np.ndarray, 
    y_hat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Returns:
        Tuple: (residuals, internally_stud_res, externally_stud_res, cooks_d, high_leverage_indices)
    """
    n, k = X_design.shape
    residuals = y - y_hat
    
    # 1. Tính Leverage
    XtX = X_design.T @ X_design
    try:
        W = np.linalg.solve(XtX, X_design.T)
    except np.linalg.LinAlgError:
        warnings.warn("Ma trận X^TX bị suy biến.", RuntimeWarning)
        W = np.linalg.pinv(XtX) @ X_design.T
        
    leverage = np.sum(X_design * W.T, axis=1)
    leverage_threshold = (2 * k) / n
    high_leverage_indices = np.where(leverage > leverage_threshold)[0].tolist()
    
    # 2. Tính Internally Studentized Residuals
    leverage_safe = np.clip(leverage, 0, 0.999)
    sigma2_hat = np.sum(residuals**2) / (n - k)
    internally_stud_res = residuals / np.sqrt(sigma2_hat * (1 - leverage_safe))
    
    # Tính Externally Studentized Residuals
    # Tránh căn số âm hoặc chia cho 0 do sai số máy tính bằng np.clip
    denom = (n - k) - internally_stud_res**2
    denom_safe = np.clip(denom, 0.001, None)
    externally_stud_res = internally_stud_res * np.sqrt((n - k - 1) / denom_safe)
    
    # 3. Tính Cook's Distance (Dùng công thức chuẩn với Internally Studentized)
    cooks_d = (internally_stud_res**2 / k) * (leverage_safe / (1 - leverage_safe))
    
    return residuals, internally_stud_res, externally_stud_res, leverage, cooks_d, high_leverage_indices

def residual_plots(X: np.ndarray, y: np.ndarray, beta_hat: np.ndarray):
    """
    Vẽ 4 biểu đồ chẩn đoán phần dư theo chuẩn Thống kê (Residual Analysis).
    """
    sns.set_theme(style="whitegrid")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.set_layout_engine('constrained')
    fig.suptitle('Phân Tích Phần Dư (Residual Analysis)', fontsize=16, fontweight='bold')

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    beta_hat = np.asarray(beta_hat, dtype=float).ravel()
    
    # Kiểm tra và thêm intercept nếu cần (giả định X gốc chưa có)
    if X.shape[1] == len(beta_hat) - 1:
        X_design = np.column_stack([np.ones(len(X)), X])
    else:
        X_design = X
        
    # Tính y dự đoán
    y_hat = X_design @ beta_hat
    
    # Lấy các thông số thống kê
    # Cập nhật dòng này trong hàm residual_plots:
    residuals, int_stud_res, ext_stud_res, leverage, cooks_d, high_leverage = compute_leverage_and_cooks(X_design, y, y_hat)

    # 1. Residuals vs Fitted (Kiểm tra tuyến tính & phương sai)
    sns.residplot(x=y_hat, y=residuals, lowess=True, 
                  scatter_kws={'alpha': 0.6, 'edgecolor': 'k'}, 
                  line_kws={'color': 'red', 'lw': 2}, ax=ax1)
    ax1.axhline(0, color='grey', linestyle='dashed')
    ax1.set_title('1. Residuals vs Fitted', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Giá trị dự đoán (Fitted values)')
    ax1.set_ylabel('Phần dư (Residuals)')

    # 2. Normal Q-Q Plot (Kiểm tra phân phối chuẩn)
    ax2 = ax2
    stats.probplot(ext_stud_res, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_alpha(0.6)
    ax2.get_lines()[0].set_markeredgecolor('k')
    ax2.get_lines()[1].set_color('red')
    ax2.get_lines()[1].set_linewidth(2)
    ax2.set_title('2. Normal Q-Q (Externally Studentized)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Phân vị lý thuyết (Theoretical Quantiles)')
    ax2.set_ylabel('Phần dư Student hóa ngoài (Externally Studentized)')

    # 3. Scale-Location (Kiểm tra phương sai đồng đều)
    sqrt_abs_std_res = np.sqrt(np.abs(int_stud_res))
    sns.regplot(x=y_hat, y=sqrt_abs_std_res, lowess=True,
                scatter_kws={'alpha': 0.6, 'edgecolor': 'k'}, 
                line_kws={'color': 'red', 'lw': 2}, ax=ax3)
    ax3.set_title('3. Scale-Location', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Giá trị dự đoán (Fitted values)')
    ax3.set_ylabel(r'$\sqrt{|Studentized\ Residuals|}$')
    
    # 4. Cook's Distance (Xác định điểm ảnh hưởng lớn)
    markerline, stemlines, baseline = ax4.stem(range(len(cooks_d)), cooks_d, basefmt=" ", markerfmt="o")
    plt.setp(stemlines, 'color', 'steelblue', 'alpha', 0.6)
    plt.setp(markerline, 'color', 'steelblue', 'alpha', 0.6, 'markeredgecolor', 'k')
    
    # Thêm ngưỡng cảnh báo Cook's distance (Thường là 4/n)
    threshold = 4 / len(y)
    ax4.axhline(threshold, color='red', linestyle='dashed', label=f'Ngưỡng (4/n) = {threshold:.3f}')
    
    ax4.set_title("4. Cook's Distance", fontsize=13, fontweight='bold')
    ax4.set_xlabel('Chỉ số quan sát (Observation Index)')
    ax4.set_ylabel("Cook's Distance")
    ax4.legend()
    
    if high_leverage:
        print(f"[CẢNH BÁO] Phát hiện các điểm có độ đòn bẩy (Leverage) cao vượt ngưỡng tiêu chuẩn (2k/n): {high_leverage}")
    else:
        print("[THÔNG BÁO] Không có điểm nào có độ đòn bẩy (Leverage) vượt ngưỡng tiêu chuẩn.")

    return fig, ((ax1, ax2), (ax3, ax4))

# KHỐI DEMO (Chạy độc lập để test hàm)
if __name__ == "__main__":
    # Tạo dữ liệu giả lập có chủ ý (có điểm outlier)
    np.random.seed(42)
    n_samples = 100
    X_demo = np.random.uniform(0, 10, size=(n_samples, 2))
    
    # y = 3 + 2*X1 - 1.5*X2 + nhiễu
    true_beta = np.array([3.0, 2.0, -1.5])
    X_design_demo = np.column_stack([np.ones(n_samples), X_demo])
    y_demo = X_design_demo @ true_beta + np.random.normal(0, 2, n_samples)
    
    # Tạo ra 2 điểm outlier (dị thường) cực mạnh để Cook's Distance bắt được
    X_demo[95] = [9.5, 9.5]
    y_demo[95] = 50.0  # Outlier 1
    
    X_demo[98] = [0.5, 0.5]
    y_demo[98] = -30.0 # Outlier 2
    
    # Tính toán OLS (dùng np.linalg.lstsq cho nhanh trong khối demo)
    X_design_fit = np.column_stack([np.ones(n_samples), X_demo])
    beta_hat_demo, _, _, _ = np.linalg.lstsq(X_design_fit, y_demo, rcond=None)
    
    # Gọi hàm vẽ
    print("Đang tạo biểu đồ Phân tích phần dư...")
    fig, axes = residual_plots(X_demo, y_demo, beta_hat_demo)
    plt.show()