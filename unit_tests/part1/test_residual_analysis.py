import numpy as np
import pytest
from part1.residual_analysis import compute_leverage_and_cooks

def _assert_close(actual, expected, msg=""):
    """Hàm helper để test mảng float"""
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, err_msg=msg)

def test_leverage_and_cooks_known_values():
    """Test 1: Kiểm tra kết quả Leverage và Phần dư trên dữ liệu tính tay được."""
    X = np.array([[1.0], [2.0], [3.0]])
    ones = np.ones((3, 1))
    X_design = np.hstack([ones, X])
    y = np.array([2.0, 4.0, 5.0])
    y_hat = np.array([2.16666667, 3.66666667, 5.16666667])

    residuals, int_stud_res, ext_stud_res, leverage, cooks_d, high_leverage = compute_leverage_and_cooks(X_design, y, y_hat)
    
    # 1. Test Leverage 
    expected_leverage = [5/6, 1/3, 5/6]
    _assert_close(leverage, expected_leverage, msg="Sai giá trị Leverage (Đường chéo Hat Matrix)")
    
    # 2. Test Residuals: e = y - y_hat
    expected_residuals = [-1/6, 2/6, -1/6]
    _assert_close(residuals, expected_residuals, msg="Sai giá trị Phần dư (Residuals)")
    
    # 3. Kiểm tra Internally Studentized Residuals
    expected_int_stud = [-1.0, 1.0, -1.0]
    _assert_close(int_stud_res, expected_int_stud, msg="Sai giá trị Internally Studentized Residuals")
    
    # 4. Kiểm tra Cook's Distance
    expected_cooks = [2.5, 0.25, 2.5]
    _assert_close(cooks_d, expected_cooks, msg="Sai giá trị Cook's Distance")

    # 5. Test Externally Studentized Residuals
    expected_ext_stud = [0.0, 0.0, 0.0]
    _assert_close(ext_stud_res, expected_ext_stud, msg="Sai giá trị Externally Studentized Residuals")
    
    # 6. Test High Leverage Indices
    assert high_leverage == [], f"Kỳ vọng list rỗng nhưng lại nhận về: {high_leverage}"

def test_leverage_properties():
    """Test 2: Kiểm tra tính chất toán học của Leverage (Tổng Leverage bằng k, giá trị thuộc khoảng hợp lệ)."""
    # Bộ dữ liệu tĩnh n=5, k=3 (2 biến + 1 intercept)
    X_design = np.array([
        [1.0, 2.5, 3.1],
        [1.0, 1.2, 8.4],
        [1.0, 5.6, 2.2],
        [1.0, 7.8, 1.1],
        [1.0, 4.4, 5.5]
    ])
    y = np.array([10, 20, 30, 40, 50])
    
    # Tính y_hat qua np.linalg.lstsq để test hàm
    beta_hat, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta_hat
    
    _, _, _, leverage, cooks_d, _ = compute_leverage_and_cooks(X_design, y, y_hat)
    
    n, k = X_design.shape # n = 5, k = 3
    
    # Tính chất 1: Tổng các đòn bẩy (sum of leverage) LUÔN LUÔN bằng số lượng tham số k
    _assert_close(np.sum(leverage), float(k), msg="Tổng Leverage phải đúng bằng số tham số k")
    
    # Tính chất 2: Kiểm tra h_ii phải thuộc khoảng [1/n, 1] theo lý thuyết chuẩn của Intercept Model
    assert np.all(leverage >= (1 / n) - 1e-7), f"Phát hiện h_ii nhỏ hơn 1/n. Mảng leverage: {leverage}"
    assert np.all(leverage <= 1.0 + 1e-7), f"Phát hiện h_ii lớn hơn 1. Mảng leverage: {leverage}"
    
    # Tính chất 3: Kiểm tra cận dưới chặt k/n cho tổng thể (hoặc từng phần tử theo một số tài liệu)
    # Vì bài test này là dữ liệu ngẫu nhiên, ta test thêm tính chất Cook's distance luôn không âm cho chắc chắn
    assert np.all(cooks_d >= 0), "Cook's Distance không được phép âm"

def test_outlier_detected():
    """Test 3: Điểm outlier mạnh (giá trị y dị biệt) phải có Cook's Distance cao nhất."""
    X_design = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0]
    ])
    y = np.array([2.0, 4.0, 6.0, 50.0])  # y = 50 là outlier cực mạnh
    
    beta_hat, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta_hat
    _, _, _, _, cooks_d, _ = compute_leverage_and_cooks(X_design, y, y_hat)

    # Khẳng định: Điểm outlier (index 3) bắt buộc phải có Cook's D lớn nhất toàn bộ tập dữ liệu
    assert np.argmax(cooks_d) == 3, f"Kỳ vọng điểm index 3 có Cook's D cao nhất, nhưng thực tế là điểm {np.argmax(cooks_d)}"


def test_high_leverage_point():
    """Test 4: Điểm có giá trị X cực đoan phải có Leverage lớn nhất và tiệm cận về sát giới hạn lý thuyết."""
    X_design = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 1000.0]  # x cực đại xa trung tâm
    ])
    y = np.array([10.0, 12.0, 11.0, 15.0])
    
    beta_hat, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta_hat

    _, _, _, leverage, _, _ = compute_leverage_and_cooks(X_design, y, y_hat)
    
    # Do x=1000 quá xa trung tâm, leverage của điểm cuối (index 3) phải tiệm cận sát số 1 (> 0.95)
    assert leverage[3] > 0.95, f"Leverage của điểm cực đoan chỉ đạt {leverage[3]}, kỳ vọng > 0.95"
    
    # Vì giới hạn mẫu nhỏ n=4, k=2 khiến ngưỡng 2k/n = 1.0 (không điểm nào vượt qua được).
    # Ta sẽ assert tính chất: Điểm index 3 bắt buộc phải là khứa có độ đòn bẩy CAO NHẤT tập dữ liệu.
    assert np.argmax(leverage) == 3, f"Kỳ vọng điểm index 3 có Leverage lớn nhất, nhưng thực tế là {np.argmax(leverage)}"