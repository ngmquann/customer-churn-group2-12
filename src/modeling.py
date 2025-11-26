import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1) Tải dữ liệu

df = pd.read_csv("Customer_Churn.csv")

# 2) Xử lý dữ liệu bị thiếu

df = df.dropna()  # Xóa các hàng chứa giá trị NaN

# 3) Mã hóa cột nhãn Churn (Yes/No → 1/0)

le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])

# 4) Tách dữ liệu thành X (đặc trưng) và y (nhãn)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# 5) Mã hóa các cột dạng chuỗi

for col in X.columns:
    if X[col].dtype == object:
        X[col] = LabelEncoder().fit_transform(X[col])

# 6) Chia dữ liệu Train/Test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7) Train mô hình Logistic Regression

log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)
log_acc = log_model.score(X_test, y_test)

# 8) Train mô hình Random Forest

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = rf_model.score(X_test, y_test)

# 9) Chọn mô hình tốt nhất

best_model = rf_model if rf_acc > log_acc else log_model

# 10) Lưu mô hình vào file model.pkl

joblib.dump(best_model, "model.pkl")

# 11) In kết quả để kiểm tra
print("Độ chính xác Logistic Regression:", log_acc)
print("Độ chính xác Random Forest:", rf_acc)
print(">> Đã lưu mô hình tốt nhất vào file model.pkl")
