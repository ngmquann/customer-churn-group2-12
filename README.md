Customer Churn Prediction (Dự đoán khách hàng rời bỏ)
Môn học: Khai thác dữ liệu
Nhóm thực hiện: Nhóm 2+12

📖 Giới thiệu (Introduction)
Dự án này xây dựng một hệ thống Machine Learning để dự đoán khả năng rời bỏ dịch vụ (Churn) của khách hàng trong lĩnh vực viễn thông. Quy trình thực hiện tuân thủ theo tiêu chuẩn CRISP-DM (Cross-Industry Standard Process for Data Mining).

Mục tiêu chính là giúp doanh nghiệp nhận diện sớm các khách hàng có nguy cơ rời bỏ để kịp thời đưa ra các chính sách chăm sóc và giữ chân (Customer Retention).

🛠 Công nghệ sử dụng (Tech Stack)

Ngôn ngữ: Python 3.x

Thư viện xử lý dữ liệu: Pandas, NumPy

Trực quan hóa: Matplotlib, Seaborn

Machine Learning: Scikit-learn (Logistic Regression, Random Forest)

Giao diện Demo: Streamlit

⚙️ Hướng dẫn cài đặt (Installation)

Clone dự án về máy:

Bash

git clone [LINK_GITHUB_CUA_BAN]

cd Customer-Churn-Prediction

Cài đặt các thư viện cần thiết: Bạn có thể cài đặt nhanh bằng câu lệnh sau:

Bash

pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib

Hướng dẫn chạy Demo (Usage)

Để khởi chạy ứng dụng dự đoán trên trình duyệt web, chạy lệnh sau tại thư mục gốc của dự án:

Bash

streamlit run src/app.py

(Lưu ý: Đảm bảo file model.pkl nằm cùng thư mục với app.py hoặc cập nhật đúng đường dẫn)

Sau khi chạy, truy cập vào địa chỉ: http://localhost:8501 trên trình duyệt.

