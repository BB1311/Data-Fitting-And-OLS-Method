import pytest
import numpy as np
import pandas as pd
from part2.model_comparison import (
    _as_numeric_dataframe,
    _as_numeric_vector,
    _check_xy_shape,
    _fit_ols,
    _safe_model_metrics,
    ols_coefficient_table,
    OLSFeatureSelector
)

# --- Tests for _as_numeric_dataframe ---
def test_as_numeric_dataframe_valid():
    """Kiểm tra xử lý đầu vào hợp lệ và tự động sinh tên cột."""
    X = [[1, 2], [3, 4]]
    df = _as_numeric_dataframe(X, feature_names=["A", "B"])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["A", "B"]
    assert df.dtypes.iloc[0] == float

def test_as_numeric_dataframe_invalid_type():
    """Kiểm tra loại bỏ dữ liệu không phải dạng số."""
    X = pd.DataFrame({"A": [1, 2], "B": ["a", "b"]})
    with pytest.raises(ValueError, match="OLS chỉ nhận biến đã mã hóa thành số"):
        _as_numeric_dataframe(X)

def test_as_numeric_dataframe_nan():
    """Kiểm tra báo lỗi khi có giá trị NaN."""
    X = pd.DataFrame({"A": [1, np.nan]})
    with pytest.raises(ValueError, match="X có NaN hoặc \\+/-inf"):
        _as_numeric_dataframe(X)

# --- Tests for _as_numeric_vector ---
def test_as_numeric_vector_valid():
    """Kiểm tra chuyển y thành vector số hợp lệ."""
    y = [1, 2, 3]
    y_arr = _as_numeric_vector(y)
    assert isinstance(y_arr, np.ndarray)
    assert y_arr.ndim == 1
    assert y_arr.dtype == float

def test_as_numeric_vector_invalid():
    """Kiểm tra báo lỗi khi target có NaN."""
    y = [1, np.nan, 3]
    with pytest.raises(ValueError, match="y có NaN hoặc \\+/-inf"):
        _as_numeric_vector(y)

# --- Tests for _check_xy_shape ---
def test_check_xy_shape_valid():
    """Kiểm tra khớp số lượng sample giữa X và y."""
    X = pd.DataFrame(np.zeros((5, 2)))
    y = np.zeros(5)
    # Should not raise exception
    _check_xy_shape(X, y)

def test_check_xy_shape_invalid():
    """Kiểm tra báo lỗi khi X và y khác số lượng dòng."""
    X = pd.DataFrame(np.zeros((5, 2)))
    y = np.zeros(4)
    with pytest.raises(ValueError, match="X và y không cùng số dòng"):
        _check_xy_shape(X, y)

# --- Tests for _fit_ols ---
def test_fit_ols_basic():
    """Kiểm tra _fit_ols cơ bản, phải trả về beta_hat."""
    X = pd.DataFrame({"X1": [1, 2, 3]})
    y = np.array([2, 4, 6])
    model = _fit_ols(X, y)
    assert hasattr(model, "beta_hat")
    assert len(model.beta_hat) == 2  # Intercept + X1

def test_fit_ols_predict():
    """Kiểm tra mô hình có thể dự đoán sau khi _fit_ols."""
    X = pd.DataFrame({"X1": [1, 2, 3]})
    y = np.array([2, 4, 6])
    model = _fit_ols(X, y)
    y_pred = model.predict(X.to_numpy())
    assert len(y_pred) == 3
    assert np.allclose(y_pred, y)

# --- Tests for _safe_model_metrics ---
def test_safe_model_metrics_normal():
    """Kiểm tra hàm tính metric cho kết quả cơ bản đúng đắn."""
    y = np.array([1, 2, 3, 4, 5])
    y_hat = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    metrics = _safe_model_metrics(y, y_hat, p=1)
    assert "r2" in metrics
    assert metrics["n"] == 5
    assert metrics["p"] == 1
    assert metrics["r2"] > 0.9

def test_safe_model_metrics_perfect_fit():
    """Kiểm tra xử lý riêng trường hợp RSS=0 không bị lỗi ZeroDivisionError."""
    y = np.array([1, 2, 3])
    y_hat = np.array([1, 2, 3])  # perfect fit
    metrics = _safe_model_metrics(y, y_hat, p=1)
    assert metrics["rss"] == 0
    assert metrics["r2"] == 1.0

# --- Tests for ols_coefficient_table ---
def test_ols_coefficient_table_basic():
    """Kiểm tra bảng hệ số phải trả ra dataframe đúng format."""
    X = pd.DataFrame({"X1": [1, 2, 3, 4, 5], "X2": [1, 4, 2, 5, 3]})
    y = [2, 4, 5, 4, 5]
    table = ols_coefficient_table(X, y)
    assert isinstance(table, pd.DataFrame)
    assert "p_value" in table.columns
    assert "coef" in table.columns
    assert len(table) == 3  # Intercept + X1 + X2

def test_ols_coefficient_table_names():
    """Kiểm tra tên cột (feature_names) tùy chỉnh trong bảng hệ số."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = [2, 4, 5, 4, 5]
    table = ols_coefficient_table(X, y, feature_names=["CustomX"])
    assert list(table["feature"]) == ["Intercept", "CustomX"]

# --- Tests for OLSFeatureSelector ---
def test_feature_selector_pvalue():
    """Kiểm tra tính năng chọn biến loại các biến có p_value cao (nhiễu)."""
    # X1 tương quan mạnh với y, X2 là nhiễu không liên quan
    X = pd.DataFrame({
        "X1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "X2": [5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0],
    })
    y = [2.1, 4.0, 5.9, 8.1, 10.0, 11.8, 14.1, 16.0, 17.9, 20.1]
    
    selector = OLSFeatureSelector(method="pvalue", alpha=0.05, verbose=False)
    selector.fit(X, y)
    
    assert "X1" in selector.selected_features_
    assert "X2" not in selector.selected_features_
    assert "X2" in selector.dropped_features_

def test_feature_selector_vif():
    """Kiểm tra loại bỏ biến theo hệ số đa cộng tuyến VIF."""
    # X2 gần như bản sao của X1 (đa cộng tuyến cao), X3 độc lập
    X = pd.DataFrame({
        "X1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "X2": [2.01, 4.00, 5.99, 8.01, 10.00, 11.99, 14.01, 16.00, 17.99, 20.01],
        "X3": [5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0],
    })
    y = [3.1, 5.0, 6.9, 9.1, 11.0, 12.9, 15.1, 17.0, 18.9, 21.1]
    
    selector = OLSFeatureSelector(method="vif", vif_threshold=5.0, verbose=False)
    selector.fit(X, y)
    
    # X1 hoặc X2 phải bị loại do VIF quá cao
    assert len(selector.dropped_features_) >= 1
    assert "X3" in selector.selected_features_

def test_feature_selector_transform():
    """Kiểm tra hàm transform lấy đúng data các cột sau fit."""
    # X1 có tương quan mạnh với y, X2 là nhiễu
    X = pd.DataFrame({
        "X1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "X2": [5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0],
    })
    y = [2.1, 4.0, 5.9, 8.1, 10.0, 11.8, 14.1, 16.0, 17.9, 20.1]
    
    selector = OLSFeatureSelector(method="pvalue", verbose=False)
    selector.fit(X, y)
    
    X_trans = selector.transform(X)
    assert list(X_trans.columns) == selector.selected_features_

def test_feature_selector_invalid_method():
    """Kiểm tra báo lỗi khi method chọn biến không hợp lệ."""
    X = pd.DataFrame({"X1": [1, 2, 3]})
    y = [1, 2, 3]
    selector = OLSFeatureSelector(method="invalid_method", verbose=False)
    with pytest.raises(ValueError, match="method phải là"):
        selector.fit(X, y)
