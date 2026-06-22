# Physics-Aware Electrochemical AI

## Giới thiệu

Physics-Aware Electrochemical AI là hệ thống trí tuệ nhân tạo được xây dựng nhằm nhận diện và phân loại các chất điện hóa dựa trên tín hiệu voltammetry.

Khác với các phương pháp AI truyền thống chỉ học từ dữ liệu, dự án này tích hợp đồng thời:

* Signal Features
* Nonlinear Features
* Physics Features

nhằm giúp mô hình khai thác được cả đặc trưng tín hiệu và tri thức điện hóa học.

Mục tiêu của dự án là xây dựng một hệ thống AI có khả năng:

* Nhận diện chất điện hóa.
* Khai thác các đặc trưng vật lý điện hóa.
* Nâng cao khả năng giải thích mô hình.
* Hỗ trợ phát triển các hệ thống Electrochemical AI trong tương lai.

---

## Ý tưởng chính

Tín hiệu voltammetry chứa nhiều thông tin quan trọng như:

* Vị trí đỉnh phản ứng.
* Dòng điện đỉnh oxy hóa.
* Độ rộng đỉnh.
* Động học truyền electron.
* Hiện tượng khuếch tán.

Các thông tin này được chuyển đổi thành các Physics Features để giúp AI học được bản chất của phản ứng điện hóa thay vì chỉ học từ dữ liệu thuần túy.

---

## Bộ đặc trưng sử dụng

### Signal Features

* Mean
* Standard Deviation
* RMS
* Maximum
* Minimum
* Energy
* Entropy

### Nonlinear Features

* Shannon Entropy
* Skewness
* Kurtosis
* DFA
* Hurst Exponent

### Physics Features

* Number of Peaks
* Oxidation Peak Current
* Oxidation Peak Voltage
* Peak Width
* Peak Prominence
* Peak Area
* Peak Symmetry
* Gradient Features
* Curvature Features
* Diffusion Asymmetry
* Tafel Slope
* Tafel Intercept
* Reaction Activity
* Kinetic Energy
* Signal Entropy
* Equilibrium Potential
* Potential Range
* Current Range
* Baseline Drift
* Electrochemical Stability

---

## Các mô hình được triển khai

### XGBoost

Mô hình Machine Learning sử dụng trên tập đặc trưng đã trích xuất.

### CNN

Khai thác đặc trưng cục bộ của tín hiệu voltammetry.

### Transformer

Mô hình hóa mối quan hệ dài hạn giữa các vùng điện thế.

### Attention Network

Tập trung vào các vùng tín hiệu quan trọng như Peak Region.

### Hybrid Model

Kết hợp:

* Handcrafted Features
* Deep Features
* Physics Features

nhằm nâng cao độ chính xác và khả năng tổng quát hóa.

## Công nghệ sử dụng

* Python
* NumPy
* Pandas
* SciPy
* Matplotlib
* Scikit-learn
* XGBoost
* TensorFlow
* Keras
* SHAP

---

## Kết quả bước đầu

Hệ thống đã triển khai thành công:

* Trích xuất đặc trưng điện hóa.
* Xây dựng Physics-Aware Dataset.
* Huấn luyện mô hình XGBoost.
* Huấn luyện CNN.
* Huấn luyện Transformer.
* Huấn luyện Hybrid Model.
* Đánh giá Accuracy, Precision, Recall, F1-score.
* Phân tích Feature Importance.

---

## Hướng phát triển

### Multi-Analyte Classification

Nhận diện đồng thời nhiều chất trong cùng một mẫu.

### Concentration Prediction

Dự đoán nồng độ chất phân tích.

### Physics-Informed Neural Network (PINN)

Tích hợp trực tiếp các phương trình điện hóa vào quá trình học.

### CNN + Attention + Transformer Fusion

Kiến trúc kết hợp nhiều tầng học sâu.

### Real-Time Electrochemical AI

Triển khai trên Raspberry Pi hoặc hệ thống nhúng.

### Mobile/Web Deployment

Xây dựng nền tảng Electrochemical AI phục vụ nghiên cứu và ứng dụng thực tế.

---

## Tác giả

Đảm Phạm

Ngành: Kỹ thuật Điện tử - Viễn thông

Lĩnh vực nghiên cứu:

* Artificial Intelligence
* Electrochemical Sensors
* Physics-Informed AI
* Biomedical Signal Processing
