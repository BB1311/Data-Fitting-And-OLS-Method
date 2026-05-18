import numpy as np
from part1.residual_analysis import compute_leverage_and_cooks

def _assert_close(actual, expected, msg=""):
    """Hàm helper để test mảng float"""
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, err_msg=msg)

def test_leverage_and_cooks_known_values():
    """Test 1: Kiểm tra kết quả Leverage và Phần dư trên dữ liệu tính tay được."""
    # Bộ dữ liệu siêu nhỏ n=3, k=2 (1 biến + 1 intercept)
    X = np.array([[1.0], [2.0], [3.0]])
    ones = np.ones((3, 1))
    X_design = np.hstack([ones, X])
    
    y = np.array([2.0, 4.0, 5.0])
    
    # Kết quả giải OLS chuẩn (beta_0 = 2/3, beta_1 = 1.5)
    y_hat = np.array([2.16666667, 3.66666667, 5.16666667])
    
    # Hứng residuals và leverage
    residuals, _, leverage, _ = compute_leverage_and_cooks(X_design, y, y_hat)
    
    # 1. Test Leverage: Công thức SLR là 1/n + (x_i - mean_x)^2 / sum((x - mean_x)^2)
    # mean_x = 2. Mẫu số = (-1)^2 + 0 + 1^2 = 2
    # h_11 = 1/3 + 1/2 = 5/6 = 0.8333...
    # h_22 = 1/3 + 0 = 1/3 = 0.3333...
    # h_33 = 1/3 + 1/2 = 5/6 = 0.8333...
    expected_leverage = [5/6, 1/3, 5/6]
    _assert_close(leverage, expected_leverage, msg="Sai giá trị Leverage (Đường chéo Hat Matrix)")
    
    # 2. Test Residuals: e = y - y_hat
    expected_residuals = [-1/6, 2/6, -1/6]
    _assert_close(residuals, expected_residuals, msg="Sai giá trị Phần dư (Residuals)")

def test_leverage_properties():
    """Test 2: Kiểm tra tính chất toán học của Leverage (Tổng Leverage phải bằng k)."""
    # Bộ dữ liệu tĩnh n=5, k=3 (2 biến + 1 intercept)
    X_design = np.array([
        [1.0, 2.5, 3.1],
        [1.0, 1.2, 8.4],
        [1.0, 5.6, 2.2],
        [1.0, 7.8, 1.1],
        [1.0, 4.4, 5.5]
    ])
    y = np.array([10, 20, 30, 40, 50]) # y không quan trọng trong test này
    
    # Tính y_hat qua np.linalg.lstsq để test hàm
    beta_hat, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta_hat
    
    _, _, leverage, _ = compute_leverage_and_cooks(X_design, y, y_hat)
    
    # Tổng các đòn bẩy (sum of leverage) LUÔN LUÔN bằng số lượng tham số k
    k = X_design.shape[1] # Ở đây k = 3
    _assert_close(np.sum(leverage), float(k), msg="Tổng Leverage phải đúng bằng số tham số k")