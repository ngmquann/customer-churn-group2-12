"""
=============================================================================
STREAMLIT APP - CUSTOMER CHURN PREDICTION SYSTEM
=============================================================================
Ứng dụng web dự đoán khả năng khách hàng rời bỏ dịch vụ (Customer Churn)
sử dụng Machine Learning model đã được train trước.

Framework: Streamlit
Model: Random Forest Classifier
=============================================================================
"""

import streamlit as st
import pandas as pd
import os
import sys

# =============================================================================
# PHẦN 1: XỬ LÝ IMPORT VÀ ĐƯỜNG DẪN
# =============================================================================

# Lấy đường dẫn tuyệt đối của file hiện tại (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Lấy đường dẫn của thư mục cha (parent directory)
parent_dir = os.path.dirname(current_dir)

# Thêm đường dẫn vào sys.path để Python có thể tìm thấy module
# insert(0, ...) đảm bảo thư mục này được tìm kiếm đầu tiên
sys.path.insert(0, os.path.join(parent_dir, 'src'))  # Thêm thư mục src/
sys.path.insert(0, current_dir)  # Thêm thư mục hiện tại

# Import function predict_churn từ module predict.py
# Sử dụng try-except để xử lý trường hợp file không tồn tại
try:
    from predict import predict_churn
except ImportError as e:
    st.error(f"❌ Không thể import module predict: {e}")
    st.stop()  # Dừng app nếu không import được

# =============================================================================
# PHẦN 2: CẤU HÌNH TRANG WEB
# =============================================================================

# Cấu hình metadata cho trang web
st.set_page_config(
    page_title="Customer Churn Prediction",  # Tiêu đề hiển thị trên tab browser
    page_icon="📊",  # Icon hiển thị trên tab
    layout="wide"  # Sử dụng toàn bộ chiều rộng màn hình
)

# =============================================================================
# PHẦN 3: CUSTOM CSS - CHỈNH SỬA GIAO DIỆN
# =============================================================================

# Inject custom CSS để tùy chỉnh giao diện
st.markdown("""
<style>
    /* Thêm padding cho phần main content */
    .main > div {padding-top: 2rem;}
    
    /* Làm cho button chiếm full width */
    .stButton > button {width: 100%;}
    
    /* Đổi màu tiêu đề chính */
    h1 {color: #1f77b4;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PHẦN 4: HEADER - TIÊU ĐỀ TRANG
# =============================================================================

st.title("📊 Customer Churn Prediction System")
st.markdown("**Dự đoán khả năng khách hàng rời bỏ dịch vụ dựa trên Machine Learning**")

# =============================================================================
# PHẦN 5: SIDEBAR - FORM NHẬP LIỆU
# =============================================================================

with st.sidebar:
    st.header("⚙️ Thông tin khách hàng")
    st.markdown("---")  # Đường phân cách
    
    # ---------------------------------------------------------------------------
    # 5.1. Thông tin cá nhân
    # ---------------------------------------------------------------------------
    st.subheader("👤 Thông tin cá nhân")
    
    # Selectbox: dropdown menu cho phép chọn 1 giá trị
    gender = st.selectbox("Giới tính", ['Male', 'Female'])
    
    # format_func: hàm format cách hiển thị giá trị (0 -> "Không", 1 -> "Có")
    senior = st.selectbox(
        "Người cao tuổi", 
        [0, 1], 
        format_func=lambda x: "Có" if x == 1 else "Không"
    )
    
    partner = st.selectbox("Có bạn đời", ['Yes', 'No'])
    dependents = st.selectbox("Có người phụ thuộc", ['Yes', 'No'])
    
    # Slider: thanh trượt để chọn giá trị số
    # Cú pháp: slider(label, min, max, default)
    tenure = st.slider("Thời gian sử dụng (tháng)", 0, 72, 12)
    
    # ---------------------------------------------------------------------------
    # 5.2. Dịch vụ sử dụng
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📞 Dịch vụ sử dụng")
    
    phone_service = st.selectbox("Dịch vụ điện thoại", ['Yes', 'No'])
    multiple_lines = st.selectbox("Nhiều đường dây", ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox("Dịch vụ Internet", ['DSL', 'Fiber optic', 'No'])
    
    # Logic điều kiện: chỉ hiển thị các dịch vụ internet nếu có dùng internet
    if internet_service != 'No':
        # Nếu có dùng internet, cho phép chọn các add-on services
        online_security = st.selectbox("Bảo mật online", ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox("Sao lưu online", ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox("Bảo vệ thiết bị", ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox("Hỗ trợ kỹ thuật", ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox("TV streaming", ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox("Movies streaming", ['No', 'Yes', 'No internet service'])
    else:
        # Nếu không dùng internet, tự động set các dịch vụ = 'No internet service'
        online_security = 'No internet service'
        online_backup = 'No internet service'
        device_protection = 'No internet service'
        tech_support = 'No internet service'
        streaming_tv = 'No internet service'
        streaming_movies = 'No internet service'
    
    # ---------------------------------------------------------------------------
    # 5.3. Thông tin thanh toán
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💳 Thanh toán")
    
    contract = st.selectbox("Loại hợp đồng", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Hóa đơn điện tử", ['Yes', 'No'])
    payment_method = st.selectbox(
        "Phương thức thanh toán", 
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 
         'Credit card (automatic)']
    )
    
    # number_input: ô nhập số với các tham số (label, min, max, default, step)
    monthly_charges = st.number_input("Chi phí hàng tháng ($)", 0.0, 200.0, 70.0, 0.5)
    total_charges = st.number_input("Tổng chi phí ($)", 0.0, 10000.0, 840.0, 10.0)

# =============================================================================
# PHẦN 6: MAIN CONTENT - NỘI DUNG CHÍNH
# =============================================================================

# Tạo 2 cột với tỷ lệ width 1.5:1
col1, col2 = st.columns([1.5, 1])

# ---------------------------------------------------------------------------
# 6.1. Cột 1: Hiển thị dữ liệu đầu vào
# ---------------------------------------------------------------------------
with col1:
    st.subheader("📋 Dữ liệu đầu vào")
    
    # Tạo dictionary chứa tất cả input data
    # Dictionary này sẽ được truyền vào model để predict
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Chuyển dict thành DataFrame để hiển thị dạng bảng đẹp hơn
    # .T = transpose: đổi hàng thành cột
    df_display = pd.DataFrame([input_data]).T
    df_display.columns = ['Giá trị']
    
    # use_container_width=True: bảng chiếm full width của container
    st.dataframe(df_display, use_container_width=True)

# ---------------------------------------------------------------------------
# 6.2. Cột 2: Hiển thị kết quả dự đoán
# ---------------------------------------------------------------------------
with col2:
    st.subheader("🎯 Kết quả dự đoán")
    
    # Button để trigger prediction
    # type="primary": nút có màu xanh nổi bật
    # use_container_width=True: button chiếm full width
    if st.button("🔮 Dự đoán ngay", type="primary", use_container_width=True):
        
        # Try-except để xử lý các lỗi có thể xảy ra
        try:
            # Hiển thị spinner (loading animation) trong khi xử lý
            with st.spinner("⏳ Đang phân tích dữ liệu..."):
                
                # GỌI HÀM PREDICT - PHẦN QUAN TRỌNG NHẤT
                result = predict_churn(input_data)
                # result là dict chứa: {'prediction': 0/1, 'probability': float, 'churn_label': 'Yes'/'No'}
                
                # ---------------------------------------------------------------
                # Hiển thị kết quả dựa trên prediction
                # ---------------------------------------------------------------
                if result['prediction'] == 1:
                    # Nếu dự đoán khách hàng sẽ churn
                    st.error("⚠️ **CẢNH BÁO: Khách hàng có nguy cơ cao rời bỏ!**")
                    
                    # Metric: hiển thị giá trị số với label và delta
                    st.metric(
                        label="Xác suất Churn", 
                        value=f"{result['probability']:.1%}",  # Format thành phần trăm với 1 chữ số thập phân
                        delta=f"{result['probability']-0.5:.1%} so với ngưỡng",  # Delta so với ngưỡng 50%
                        delta_color="inverse"  # Màu đỏ nếu tăng (vì tăng là xấu trong trường hợp này)
                    )
                else:
                    # Nếu dự đoán khách hàng trung thành
                    st.success("✅ **AN TOÀN: Khách hàng trung thành**")
                    st.metric(
                        label="Xác suất Churn", 
                        value=f"{result['probability']:.1%}",
                        delta=f"{0.5-result['probability']:.1%} dưới ngưỡng",
                        delta_color="normal"  # Màu xanh nếu giảm (giảm là tốt)
                    )
                
                # Thanh progress bar để visualize probability
                # Giá trị từ 0.0 đến 1.0
                st.progress(result['probability'])
                
                # ---------------------------------------------------------------
                # Phân tích chi tiết
                # ---------------------------------------------------------------
                st.markdown("---")
                st.subheader("📊 Phân tích chi tiết")
                
                # Phân loại mức độ rủi ro dựa trên probability
                if result['probability'] > 0.7:
                    risk_level = "🔴 Cao"
                elif result['probability'] > 0.4:
                    risk_level = "🟡 Trung bình"
                else:
                    risk_level = "🟢 Thấp"
                
                st.write(f"**Mức độ rủi ro:** {risk_level}")
                
                # ---------------------------------------------------------------
                # Phân tích các yếu tố ảnh hưởng
                # ---------------------------------------------------------------
                st.markdown("**Các yếu tố chính:**")
                factors = []  # List để chứa các yếu tố
                
                # Logic kiểm tra từng yếu tố rủi ro
                if contract == 'Month-to-month':
                    factors.append("• Hợp đồng ngắn hạn (tăng rủi ro)")
                
                if tenure < 12:
                    factors.append("• Khách hàng mới (tăng rủi ro)")
                
                if internet_service == 'Fiber optic' and online_security == 'No':
                    factors.append("• Không dùng dịch vụ bảo mật")
                
                if payment_method == 'Electronic check':
                    factors.append("• Thanh toán qua séc điện tử")
                
                if monthly_charges > 70:
                    factors.append("• Chi phí tháng cao")
                
                # Hiển thị các yếu tố
                if factors:
                    for f in factors:
                        st.markdown(f)
                else:
                    st.markdown("• Hồ sơ khách hàng ổn định ✓")
        
        # ---------------------------------------------------------------
        # XỬ LÝ CÁC LOẠI LỖI
        # ---------------------------------------------------------------
        except FileNotFoundError as e:
            # Lỗi: không tìm thấy file model hoặc preprocessor
            st.error("❌ **Lỗi:** Không tìm thấy file model hoặc preprocessor!")
            st.info("Hãy chạy `modeling.py` trước để train model.")
            st.code(str(e))
            
        except Exception as e:
            # Lỗi chung (catch-all)
            st.error(f"❌ **Đã xảy ra lỗi:** {str(e)}")
            st.info("Vui lòng kiểm tra lại dữ liệu đầu vào và thử lại.")
            
            # Expander: phần có thể mở rộng để xem chi tiết
            with st.expander("Chi tiết lỗi"):
                st.code(str(e))

# =============================================================================
# PHẦN 7: KHUYẾN NGHỊ HÀNH ĐỘNG
# =============================================================================

st.markdown("---")
st.subheader("💡 Khuyến nghị hành động")

# Tạo 2 cột để hiển thị 2 trường hợp
col3, col4 = st.columns(2)

# ---------------------------------------------------------------------------
# 7.1. Khuyến nghị khi nguy cơ cao
# ---------------------------------------------------------------------------
with col3:
    st.markdown("### 🔴 Nếu nguy cơ cao")
    st.markdown("""
    1. **Liên hệ ngay:** Gọi điện trong 24h
    2. **Ưu đãi đặc biệt:** Giảm 20-30% phí dịch vụ
    3. **Chuyển đổi hợp đồng:** Đề xuất hợp đồng dài hạn
    4. **Tặng quà:** Miễn phí 1 tháng dịch vụ cao cấp
    5. **Phân tích sâu:** Tìm hiểu nguyên nhân không hài lòng
    """)

# ---------------------------------------------------------------------------
# 7.2. Khuyến nghị khi nguy cơ thấp
# ---------------------------------------------------------------------------
with col4:
    st.markdown("### 🟢 Nếu nguy cơ thấp")
    st.markdown("""
    1. **Duy trì chất lượng:** Theo dõi satisfaction score
    2. **Upsell:** Đề xuất gói dịch vụ cao cấp
    3. **Loyalty program:** Thêm điểm thưởng
    4. **Cross-sell:** Giới thiệu dịch vụ mới
    5. **Referral:** Khuyến khích giới thiệu bạn bè
    """)

# =============================================================================
# PHẦN 8: FOOTER - THÔNG TIN MODEL
# =============================================================================

st.markdown("---")

# Tạo 3 cột để hiển thị metrics
col_a, col_b, col_c = st.columns(3)

with col_a:
    # ✅ Hiển thị model: Logistic Regression (theo kết quả thực tế)
    st.metric("Model", "Logistic Regression")

with col_b:
    # ✅ Hiển thị accuracy thực tế: 79.9%
    st.metric("Accuracy", "79.9%")

with col_c:
    # ✅ Hiển thị số features
    st.metric("Features", "19+")

# Caption: text nhỏ màu xám ở cuối trang
st.caption("📝 *Demo system by Streamlit | Model: Logistic Regression | AUC: 0.84*")

# =============================================================================
# KẾT THÚC APP
# =============================================================================