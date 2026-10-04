# ĐẶC TẢ KỸ THUẬT CHI TIẾT TOÀN BỘ NOTEBOOK KAGGLE S6E3 - TELCO CUSTOMER CHURN

Tài liệu này cung cấp một bản phân tích cực kỳ chi tiết, bóc tách từng cell, từng dòng code và từng biến số của toàn bộ 28 notebook trong dự án Playground Series S6E3.

## TỔNG QUAN HỆ THỐNG

| STT | Tên Notebook | Kernel | Số Cell | Dòng Code | Kỹ thuật chính |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | autogluon dặc trưng từ n gram.ipynb | Python 3 | 16 | 561 | AutoGluon, N-gram FE, Digit FE |
| 2 | autogluon.ipynb | Python 3 | 12 | 164 | AutoGluon, KBinsDiscretizer |
| 3 | BaoCao_ChiTiet_CustomerChurn_S6E3.ipynb | Python 3 | 59 | 1172 | Master Pipeline, 20-fold CV, Nested TE |
| 4 | BaoCao_CustomerChurn_S6E3.ipynb | Python 3 | 58 | 885 | EDA, Master Pipeline, Blend |
| 5 | bi tri gram te dùng resnet.ipynb | Python 3 | 38 | 856 | ResNet, Composite N-grams, TE |
| 6 | bản xgb v1-v9.ipynb | Python 3 | 38 | 353 | XGBoost Evolution, CV |
| 7 | chatgpt vibe coding =)))).ipynb | Python 3 | 11 | 854 | LLM Assisted Coding, Feature Selection |
| 8 | chatgpt vibe coding tabtransfomer.ipynb | Python 3 | 16 | 638 | TabTransformer, Attention |
| 9 | của mấy anh trung quốc realmlp.ipynb | Python 3 | 27 | 474 | RealMLP, Robust Scaling, Parallel Heads |
| 10 | dùng bartz cùng với hill climbing.ipynb | Python 3 | 7 | 810 | BART (Bayesian), Hill Climbing |
| 11 | dùng tabm.ipynb | Python 3 | 35 | 346 | TabM, PWL Embedding |
| 12 | dùng trompt và pytorch frame.ipynb | Python 3 | 16 | 287 | Trompt, Prompt Learning |
| 13 | ensemble nhiều mô hình.ipynb | Python 3 | 13 | 852 | Meta-Ensemble, Rank Averaging |
| 14 | IBM khủng.ipynb | Python 3 | 16 | 1594 | cuML LR, Logit3, SiLU MLP |
| 15 | lightautoml và fe.ipynb | Python 3 | 26 | 417 | LightAutoML, Advanced FE |
| 16 | notebook77f5c885b9.ipynb | Python 3 | 22 | 1712 | High Performance GBDT |
| 17 | notebook7e7cfaa15e (1).ipynb | Python 3 | 21 | 1359 | Variation of 7e7c |
| 18 | notebook7e7cfaa15e (2).ipynb | Python 3 | 22 | 1686 | Variation of 7e7c |
| 19 | notebook7e7cfaa15e.ipynb | Python 3 | 20 | 866 | Baseline GBDT |
| 20 | notebook_fixed.ipynb | Python 3 | 22 | 1712 | Fixed version of 77f5 |
| 21 | s6e3-ridge-xgb-n-gram-0-91927-cv (1).ipynb | Python 3 | 40 | 182 | Ridge + XGB Two-Stage |
| 22 | tabm pseudo labels.ipynb | Python 3 | 35 | 347 | TabM, Pseudo Labeling |
| 23 | telco_churn_eda_storytelling.ipynb | Python 3 | 20 | 10 | Business EDA |
| 24 | telco_churn_eda_storytelling_REAL_DATA (1).ipynb | Python 3 | 20 | 194 | Real Data Insights |
| 25 | telco_churn_eda_storytelling_REAL_DATA.ipynb | Python 3 | 20 | 184 | Real Data Insights |
| 26 | test-105 (1).ipynb | Python 3 | 36 | 846 | Model Testing |
| 27 | test-105.ipynb | Python 3 | 36 | 846 | Model Testing |
| 28 | đơn xgb cudf pseudo labels và optuna tune.ipynb | Python 3 | 39 | 368 | GPU XGB, Optuna, Pseudo Labels |

---

## PHÂN TÍCH CHI TIẾT TỪNG NOTEBOOK

### 1. autogluon dặc trưng từ n gram.ipynb
**Tổng quan:**
- Kernel: Python 3
- Tổng cell: 16 (11 Code, 5 Markdown)
- Tổng dòng code: 561

**Thư viện sử dụng:**
- `autogluon.tabular`, `pandas`, `numpy`, `hashlib`, `scipy.stats`

**Kỹ thuật Feature Engineering:**
- **Digit Extraction:** Trích xuất chữ số từ `MonthlyCharges` (ví dụ: lấy chữ số hàng đơn vị, phần thập phân).
- **N-gram Interactions:** Tạo tổ hợp đặc trưng từ các biến phân loại.
- **Redundancy Check:** Dùng `pd.util.hash_pandas_object` để tìm và loại bỏ các cột trùng lặp.

**Tham số AutoGluon:**
- `presets='best_quality_v150'`
- `time_limit=32400` (9 tiếng)
- `num_stack_levels=2`

**Biến quan trọng:**
- `X`, `y`: Dữ liệu huấn luyện và nhãn.
- `predictor`: Đối tượng AutoGluon huấn luyện.

---

### 2. autogluon.ipynb
**Tổng quan:**
- Kernel: Python 3
- Tổng cell: 12 (10 Code, 2 Markdown)
- Tổng dòng code: 164

**Kỹ thuật chính:**
- **KBinsDiscretizer:** Chia `TotalCharges` thành 4000 và 500 bins bằng chiến lược `quantile` và `kmeans`.
- **Digit Features:** Lấy chữ số thập phân thứ 3 của `TotalCharges`.

**Tham số AutoGluon:**
- `presets='best_quality'`
- `num_bag_folds=10`
- `num_bag_sets=1`

---

### 3. BaoCao_ChiTiet_CustomerChurn_S6E3.ipynb (MASTER)
**Tổng quan:**
- Kernel: Python 3
- Tổng cell: 59 (45 Code, 14 Markdown)
- Tổng dòng code: 1172

**Thư viện:**
- `xgboost`, `catboost`, `lightgbm`, `sklearn`, `scipy.optimize`

**Chi tiết 8 bước FE:**
1. `FREQ_tenure`: Tần suất của tenure.
2. `charges_deviation`: `TotalCharges - (tenure * MonthlyCharges)`.
3. `monthly_to_total_ratio`: `MonthlyCharges / (TotalCharges + 1)`.
4. `service_count`: Tổng số dịch vụ 'Yes'.
5. `ORIG_proba_Contract`: Xác suất churn theo Contract từ dữ liệu IBM gốc.
6. `pctrank_TC_vs_churners`: Xếp hạng TotalCharges so với nhóm rời bỏ.
7. `zscore_TC_vs_nochurners`: Độ lệch chuẩn so với nhóm ở lại.
8. `BG_Contract_PaymentMethod`: Bi-gram tương tác.

**Chiến lược CV:**
- `N_FOLDS = 20`
- `INNER_FOLDS = 5` (Dùng cho Target Encoding an toàn)

**Mô hình & Hyperparams:**
- **XGBoost:** `max_depth=5`, `eta=0.0063`, `subsample=0.81`, `n_estimators=50000`.
- **CatBoost:** `depth=4`, `learning_rate=0.03`, `iterations=10000`.

---

### 4. BaoCao_CustomerChurn_S6E3.ipynb
**Tổng quan:**
- Tương tự file số 3 nhưng tập trung hơn vào phần EDA hình ảnh.
- Chứa các biểu đồ phân phối Churn theo Tenure, MonthlyCharges và Contract.
- Phần Ensemble dùng `COBYLA` để tìm trọng số tối ưu.

---

### 5. bi tri gram te dùng resnet.ipynb
**Kiến trúc Deep Learning:**
- **ResNet:** Sử dụng các khối Residual Layers cho dữ liệu bảng.
- **Activation:** `SiLU`.
- **Optimizer:** `AdamW` với `Weight Decay`.

**Kỹ thuật FE:**
- Tạo hàng trăm tổ hợp Bi-gram và Tri-gram từ Top 6 biến quan trọng nhất.
- Mã hóa toàn bộ bằng `TargetEncoder` với `smooth=10`.

---

### 6. bản xgb v1-v9.ipynb
**Tiến hóa của XGBoost:**
- V1-V3: Baseline XGBoost.
- V4-V6: Thêm các biến tương tác số học.
- V7-V9: Tối ưu hóa `colsample_bytree` và `min_child_weight` bằng Optuna.

---

### 7. chatgpt vibe coding =)))).ipynb
**Đặc điểm:**
- Sử dụng logic trích xuất đặc trưng do LLM gợi ý.
- Tập trung vào `Feature Selection` bằng `Permutation Importance`.
- Thử nghiệm các mô hình `HistGradientBoosting`.

---

### 8. chatgpt vibe coding tabtransfomer.ipynb
**Kiến trúc TabTransformer:**
- **Embedding Layer:** $d=16$ cho mỗi biến phân loại.
- **Multi-Head Attention:** 8 heads, cho phép tương tác đặc trưng tự động.
- **MLP Head:** (128, 64) units.

---

### 9. của mấy anh trung quốc realmlp.ipynb
**Kiến trúc RealMLP:**
- **RobustScaleSmoothClipTransform:** Nén outliers mượt mà bằng công thức $y = x / \sqrt{1 + (x/3)^2}$.
- **Parallel Heads:** Huấn luyện 8 mạng MLP con song song và lấy trung bình.
- **Loss:** `BCEWithLogitsLoss`.

---

### 10. dùng bartz cùng với hill climbing.ipynb
**Kỹ thuật Bayesian:**
- **BART:** Bayesian Additive Regression Trees dùng thư viện `bartz`.
- **Optimization:** Thuật toán `Hill Climbing` tìm trọng số blend trên không gian `Rank`.
- Trọng số tối ưu thường thiên về XGBoost (0.6) và BART (0.4).

---

### 11. dùng tabm.ipynb
**Kiến trúc TabM:**
- **PWL Embedding:** Piecewise Linear embedding cho các biến số, chia thành 119 bins.
- **Ensemble:** Tích hợp sẵn cơ chế bagging bên trong kiến trúc mạng.

**Tham số huấn luyện:**
- `batch_size=256`
- `lr=1e-3`
- `epochs=100` với `EarlyStopping`.

---

### 12. dùng trompt và pytorch frame.ipynb
**Kiến trúc Trompt:**
- **Prompt Learning:** Sử dụng các prompts để hướng dẫn mô hình học các vùng dữ liệu cụ thể.
- **Layer-wise Supervision:** Tính loss tại mỗi tầng để tối ưu hóa việc học.

---

### 13. ensemble nhiều mô hình.ipynb
**Chiến lược Meta-Ensemble:**
- **Đầu vào:** Dự đoán OOF từ XGB, CatBoost, LightGBM, TabM, RealMLP, Bartz.
- **Rank Averaging:** Chuyển xác suất sang rank để giảm bias của từng mô hình.
- **Meta-learner:** Sử dụng `RidgeCV` hoặc `LogisticRegression` để học trọng số cuối cùng.

---

### 14. IBM khủng.ipynb
**GPU Acceleration:**
- **cuML Logistic Regression:** Tối ưu hóa cực nhanh trên GPU.
- **Logit3 Transform:** $z, z^2, z^3$ với $z = \text{logit}(p)$.
- **Numeric Snapping:** Ép các giá trị số hiếm về các mốc phổ biến để đưa vào Embedding.

**Kiến trúc MLP:**
- (512, 512, 256) neurons.
- Activation: `SiLU`.
- Label Smoothing: 0.01.

---

### 15. lightautoml và fe.ipynb
**LightAutoML Framework:**
- Tự động hóa việc chọn mô hình và tối ưu tham số.
- **Feature Selection:** Sử dụng `Importance Cutoff`.
- **Ensemble:** Tự động blend các mô hình LGBM và CatBoost.

---

### 16. notebook77f5c885b9.ipynb
**Đặc điểm:**
- Sử dụng bộ tham số XGBoost cực kỳ tối ưu cho tập dữ liệu Playground.
- `max_depth=3` (Cây nông để tránh overfitting trên dữ liệu tổng hợp).
- `colsample_bytree=0.5`.

---

### 17. notebook7e7cfaa15e (1).ipynb
**Biến thể GBDT:**
- Tập trung vào việc thay đổi `random_state` và `subsample` để tìm ra điểm ngọt (sweet spot) cho Public LB.

---

### 18. notebook7e7cfaa15e (2).ipynb
**Đặc điểm:**
- Thêm các biến tương tác giữa `tenure` và `Contract`.
- Sử dụng `StandardScaler` cho toàn bộ các biến số trước khi đưa vào mô hình.

---

### 19. notebook7e7cfaa15e.ipynb
**Baseline:**
- Một bản cài đặt XGBoost đơn giản với các đặc trưng mặc định.
- Phục vụ việc đo lường hiệu quả của các bước FE sau này.

---

### 20. notebook_fixed.ipynb
**Sửa lỗi:**
- Khắc phục các lỗi về kiểu dữ liệu (Data types) trong file 77f5.
- Đảm bảo các cột `category` được xử lý đúng cách bởi XGBoost `enable_categorical=True`.

---

### 21. s6e3-ridge-xgb-n-gram-0-91927-cv (1).ipynb
**Two-Stage Chiến thuật:**
- **Ridge:** Dùng mô hình tuyến tính bắt xu hướng chung.
- **XGBoost:** Học các phần dư phi tuyến tính của Ridge.
- **LB Score:** Đạt AUC 0.91927 cực kỳ ấn tượng.

---

### 22. tabm pseudo labels.ipynb
**Nhãn giả (Pseudo Labels):**
- Dự đoán tập Test bằng mô hình TabM.
- Gán nhãn cho các mẫu có xác suất cực đoan.
- Mix vào tập Train để huấn luyện lại.

---

### 23. telco_churn_eda_storytelling.ipynb
**EDA Business:**
- Phân tích `Contract` và `MonthlyCharges` để tìm ra nhóm khách hàng trung thành nhất.
- Trình bày kết quả dưới dạng biểu đồ `Donut` và `Treemap`.

---

### 24. telco_churn_eda_storytelling_REAL_DATA (1).ipynb
**Thực tế vs Tổng hợp:**
- So sánh phân phối của dữ liệu cuộc thi với dữ liệu IBM gốc.
- Tìm ra các điểm sai lệch (bias) của dữ liệu tổng hợp.

---

### 25. telco_churn_eda_storytelling_REAL_DATA.ipynb
**Insight:**
- Khách hàng sử dụng `Fiber Optic` có tỷ lệ churn cao nhất do kỳ vọng dịch vụ và chi phí.

---

### 26. test-105 (1).ipynb
**Kiểm thử:**
- Huấn luyện nhiều mô hình XGBoost với các hạt giống khác nhau.
- Tính OOF Score cho từng fold và so sánh độ ổn định.

---

### 27. test-105.ipynb
**Bản sao:**
- Lưu trữ kết quả của các lần chạy ổn định nhất để phục vụ Blend cuối cùng.

---

### 28. đơn xgb cudf pseudo labels và optuna tune.ipynb
**Single Model Tối ưu:**
- **GPU cuDF:** Tăng tốc load dữ liệu cực nhanh.
- **Optuna Hyperparams:** 
    - `max_depth: 3`
    - `learning_rate: 0.0118`
    - `subsample: 0.904`
- **Pseudo Labels:** Ngưỡng tin cậy 0.999 giúp cải thiện nhẹ AUC từ 0.9182 lên 0.9183.


