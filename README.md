# ĐỒ ÁN 2: Data Fitting & Phương pháp OLS
**Trường:** Đại học Khoa học tự nhiên, ĐHQG-HCM  
**Nhóm thực hiện:** Nhóm 6

---

## Tổng quan đồ án
Đồ án này tập trung vào hai nhiệm vụ chính nhằm nắm vững và áp dụng phương pháp Bình phương tối thiểu (Ordinary Least Squares - OLS) cùng các kỹ thuật khớp dữ liệu (Data Fitting):

**1. Lý thuyết và minh họa (Phần 1)**: Củng cố nền tảng toán học của OLS.
* Cài đặt từ đầu thuật toán OLS, ma trận chiếu (Hat Matrix) và kiểm chứng các tính chất.
* Tính toán các chỉ số đánh giá ($R^2$, $\bar{R}^2$, kiểm định $F$) và suy diễn thống kê hệ số (Standard errors, $t$-statistics, $p$-values, khoảng tin cậy 95%, VIF).
* Triển khai kỹ thuật hiệu chỉnh (Ridge, Lasso Regression), phân tích thặng dư (Residual Analysis) và đánh giá chéo ($k$-fold Cross-validation).
* Minh họa định lý **Gauss-Markov** thông qua mô phỏng Monte Carlo để kiểm chứng tính không chệch và phương sai nhỏ nhất (BLUE).

**2. Ứng dụng thực tế (Phần 2)**: Vận dụng OLS vào bộ dữ liệu thực tế.
* Khảo sát dữ liệu (EDA), tiền xử lý, làm sạch dữ liệu (đặc biệt các phương pháp xử lý missing values) và đóng gói thành Data Pipeline.
* Xây dựng, đánh giá và so sánh hiệu suất các mô hình hồi quy trên dữ liệu thực tế (OLS cơ bản, OLS chọn biến, Ridge/Lasso, và các mô hình mở rộng).

## Cấu trúc thư mục
Đồ án được chia thành các phần chính, bám sát các yêu cầu thực hành và phân tích:
```text
Group_6/
|-- requirements.txt        # Danh sách thư viện phụ thuộc
|-- README.md               # File hướng dẫn
|-- pytest.ini              # File cấu hình cho kiểm thử tự động pytest
|-- report/                 # Thư mục chứa báo cáo đồ án
|   `-- report.pdf          # File báo cáo
|
|-- part1/                  # Cài đặt và phân tích OLS
|   |-- ols_implementation.py # Cài đặt thuật toán Bình phương tối thiểu (OLS)
|   |-- cross_validation.py   # Các kỹ thuật đánh giá chéo (Cross-validation)
|   |-- residual_analysis.py  # Phân tích thặng dư (Residual analysis)
|   |-- ridge_lasso.py        # Các kỹ thuật hiệu chỉnh (Ridge & Lasso Regression)
|   |-- part1_notebook.ipynb  # Notebook chạy thử, phân tích và trực quan hóa Part 1
|
|-- part2/                  # Xử lý dữ liệu và Đánh giá mô hình
|   |-- data/                 # Thư mục chứa tập dữ liệu đầu vào
|   |   `-- AmesHousing.csv   # File dữ liệu gốc
|   |-- clean_data.py         # Tiền xử lý và làm sạch dữ liệu
|   |-- data_pipeline.py      # Xây dựng luồng xử lý dữ liệu tự động (Pipeline)
|   |-- advanced_methods.py   # Các phương pháp khớp dữ liệu nâng cao
|   |-- model_comparison.py   # Đánh giá và so sánh hiệu suất các mô hình
|   |-- part2_notebook.ipynb  # Notebook chạy thử, phân tích và trực quan hóa Part 2
|
|-- unit_tests/             # Các kịch bản kiểm thử tự động (Unit Tests)
|   |-- part1/              # Test cases cho mã nguồn phần 1
|   |   |-- test_ols_implementation.py  # Kiểm thử thuật toán OLS cốt lõi
|   |   |-- test_ridge_lasso.py         # Kiểm thử mô hình Ridge, Lasso
|   |   |-- test_residual_analysis.py   # Kiểm thử phân tích thặng dư
|   |   `-- test_cross_validation.py    # Kiểm thử phân chia dữ liệu CV
|   |-- part2/              # Test cases cho mã nguồn phần 2
|   |   |-- test_clean_data.py          # Kiểm thử hàm làm sạch dữ liệu
|   |   |-- test_data_pipeline.py       # Kiểm thử Pipeline tiền xử lý
|   |   |-- test_model_comparison.py    # Kiểm thử các pipeline đánh giá mô hình
|   |   `-- test_advanced_methods.py    # Kiểm thử thuật toán mở rộng
```

## Dữ liệu sử dụng
Đồ án sử dụng bộ dữ liệu **Ames Housing** nguyên bản được biên soạn bởi **Giáo sư Dean De Cock** ([Bài báo học thuật gốc, 2011](https://jse.amstat.org/v19n3/decock.pdf)) cho mục đích giáo dục khoa học dữ liệu. Dữ liệu được lấy từ nền tảng [Kaggle](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset/data) và lưu trữ tại `part2/data/AmesHousing.csv`. Đây là bộ dữ liệu thực tế lý tưởng, thỏa mãn đầy đủ các yêu cầu của đồ án (có chứa missing values, biến mục tiêu liên tục và số lượng quan trắc đủ lớn).

## Yêu cầu hệ thống và Cài đặt
Đồ án yêu cầu **Python 3.10+**. 
Đồ án sử dụng Python cho các thuật toán lõi được cài đặt từ đầu (from scratch) dựa trên công thức toán học. Các thư viện phổ biến trong Data Science được sử dụng với mục đích cụ thể như sau:
* **`pandas`**: Đọc, xử lý và thao tác dữ liệu.
* **`matplotlib`, `seaborn`**: Trực quan hóa dữ liệu và kết quả mô hình.
* **`numpy`, `scipy`, `scikit-learn`**: S o sánh và kiểm chứng kết quả (verification) của thuật toán tự cài đặt, không dùng để thay thế phần cài đặt thuật toán chính.
* **`pytest`**: Cho các kịch bản kiểm thử tự động.

Để chạy source code, cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

**Lưu ý quan trọng về môi trường hệ thống:**  

* **Jupyter Notebook:** Môi trường cần cài đặt sẵn Jupyter để có thể xem và thực thi các file `part1_notebook.ipynb` và `part2_notebook.ipynb`.  
* **Pytest:** Đồ án có tích hợp sẵn `unit_tests`, để chạy kiểm thử tổng quát, hãy gõ lệnh `pytest` tại thư mục gốc.

## Thông tin môn học
**Tên môn học:** Toán ứng dụng và thống kê

**Giảng viên:**
* Ths. Lê Nhựt Nam
* Ths. Võ Nam Thục Đoan

## Sinh viên thực hiện
| STT | Họ và Tên | MSSV |
|:---:|:---|:---:|
| 1 | Tôn Thất Kiên | 24120078 |
| 2 | Phù Yến Nhi | 24120112 |
| 3 | Mai Thảo Vy | 24120160 |
| 4 | Phạm Gia Bảo | 24120170 |
| 5 | Trần Ngô Uyên Nhi | 24120210 |