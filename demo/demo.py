"""
=============================================================================
STREAMLIT APP - CUSTOMER CHURN PREDICTION SYSTEM
=============================================================================
Ứng dụng web dự đoán khả năng khách hàng rời bỏ dịch vụ (Customer Churn)
sử dụng Machine Learning model đã được train trước.

Framework: Streamlit
Model: Logistic Regression Classifier
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# =============================================================================
# PHẦN 1: XỬ LÝ IMPORT VÀ ĐƯỜNG DẪN
# =============================================================================

# Lấy đường dẫn tuyệt đối của file hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Lấy đường dẫn của thư mục cha (DATA-MINING/)
parent_dir = os.path.dirname(current_dir)

# Thêm đường dẫn vào sys.path
sys.path.insert(0, os.path.join(parent_dir, 'src'))
sys.path.insert(0, parent_dir)

# Import functions từ module predict.py
try:
    from src.predict import load_model, predict_single
except ImportError:
    try:
        from predict import load_model, predict_single
    except ImportError as e:
        st.error(f"❌ Không thể import module predict: {e}")
        st.stop()

# =============================================================================
# PHẦN 2: CẤU HÌNH TRANG WEB
# =============================================================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# =============================================================================
# PHẦN 3: CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stButton > button {width: 100%;}
    h1 {color: #1f77b4;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PHẦN 4: LOAD MODEL
# =============================================================================

@st.cache_resource
def get_model():
    """Load model từ file pkl"""
    model_path = os.path.join(parent_dir, "models", "model.pkl")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy model tại: `{model_path}`")
        st.info(f"📁 Thư mục hiện tại: `{os.getcwd()}`")
        st.info(f"📂 Project root: `{parent_dir}`")
        st.stop()
    
    return load_model(model_path)

# Load model
try:
    model = get_model()
except Exception as e:
    st.error(f"❌ Lỗi khi load model: {e}")
    st.info("💡 Hãy chạy `modeling.py` hoặc notebook để train model trước!")
    st.stop()

# =============================================================================
# PHẦN 5: HEADER
# =============================================================================

st.title("📡 Dự Đoán Rời Bỏ - Dịch Vụ Viễn Thông")
st.markdown("**Nhập thông tin khách hàng để dự đoán nguy cơ Churn dựa trên Machine Learning**")
st.divider()

# =============================================================================
# PHẦN 6: SIDEBAR - FORM NHẬP LIỆU
# =============================================================================

with st.sidebar:
    st.header("⚙️ Thông tin khách hàng")
    st.markdown("---")
    
    # ---------------------------------------------------------------------------
    # 6.1. Thông tin cá nhân
    # ---------------------------------------------------------------------------
    st.subheader("👤 Thông tin cá nhân")
    
    gender = st.selectbox("Giới tính", ['Female', 'Male'])
    senior_citizen = st.selectbox(
        "Người cao tuổi", 
        ['No', 'Yes']
    )
    partner = st.selectbox("Có bạn đời", ['No', 'Yes'])
    dependents = st.selectbox("Người phụ thuộc", ['No', 'Yes'])
    tenure = st.slider("Thâm niên (tháng)", 0, 72, 12)
    
    # ---------------------------------------------------------------------------
    # 6.2. Dịch vụ đăng ký
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📞 Dịch vụ đăng ký")
    
    phone_service = st.selectbox("Dịch vụ thoại", ['No', 'Yes'])
    multiple_lines = st.selectbox("Nhiều đường dây", ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox("Internet", ['DSL', 'Fiber optic', 'No'])
    
    # Các dịch vụ đi kèm Internet
    if internet_service != 'No':
        online_security = st.selectbox("Bảo mật Online", ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox("Sao lưu Online", ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox("Bảo vệ thiết bị", ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox("Hỗ trợ kỹ thuật", ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox("Truyền hình (Streaming TV)", ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox("Phim ảnh (Streaming Movies)", ['No', 'Yes', 'No internet service'])
    else:
        online_security = 'No internet service'
        online_backup = 'No internet service'
        device_protection = 'No internet service'
        tech_support = 'No internet service'
        streaming_tv = 'No internet service'
        streaming_movies = 'No internet service'
    
    # ---------------------------------------------------------------------------
    # 6.3. Hợp đồng & Thanh toán
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💳 Hợp đồng & Thanh toán")
    
    contract = st.selectbox("Loại hợp đồng", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Hóa đơn điện tử", ['No', 'Yes'])
    payment_method = st.selectbox("Phương thức thanh toán", [
        'Bank transfer (automatic)', 
        'Credit card (automatic)', 
        'Electronic check', 
        'Mailed check'
    ])
    monthly_charges = st.number_input("Cước hàng tháng ($)", min_value=0.0, value=70.0, step=0.5)
    total_charges = st.number_input("Tổng cước tích lũy ($)", min_value=0.0, value=1500.0, step=10.0)

# =============================================================================
# PHẦN 7: MAIN CONTENT - HIỂN THỊ DỮ LIỆU VÀ DỰ ĐOÁN
# =============================================================================

col1, col2 = st.columns([1.5, 1])

# ---------------------------------------------------------------------------
# 7.1. Cột 1: Hiển thị dữ liệu đầu vào
# ---------------------------------------------------------------------------
with col1:
    st.subheader("📋 Dữ liệu đầu vào")
    
    # Hàm encode giá trị categorical thành số
    def get_index(value, options):
        """Trả về index của value trong list options đã sort A-Z"""
        options_sorted = sorted(options)
        return options_sorted.index(value)
    
    # Chuẩn bị dictionary input với encoding
    input_data_display = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
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
    
    # Dictionary cho model (đã encode thành số)
    input_data = {
        'gender': get_index(gender, ['Female', 'Male']),
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': get_index(partner, ['No', 'Yes']),
        'Dependents': get_index(dependents, ['No', 'Yes']),
        'tenure': tenure,
        'PhoneService': get_index(phone_service, ['No', 'Yes']),
        'MultipleLines': get_index(multiple_lines, ['No', 'No phone service', 'Yes']),
        'InternetService': get_index(internet_service, ['DSL', 'Fiber optic', 'No']),
        'OnlineSecurity': get_index(online_security, ['No', 'No internet service', 'Yes']),
        'OnlineBackup': get_index(online_backup, ['No', 'No internet service', 'Yes']),
        'DeviceProtection': get_index(device_protection, ['No', 'No internet service', 'Yes']),
        'TechSupport': get_index(tech_support, ['No', 'No internet service', 'Yes']),
        'StreamingTV': get_index(streaming_tv, ['No', 'No internet service', 'Yes']),
        'StreamingMovies': get_index(streaming_movies, ['No', 'No internet service', 'Yes']),
        'Contract': get_index(contract, ['Month-to-month', 'One year', 'Two year']),
        'PaperlessBilling': get_index(paperless_billing, ['No', 'Yes']),
        'PaymentMethod': get_index(payment_method, ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']),
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Hiển thị dạng bảng (dùng data gốc để dễ đọc)
    df_display = pd.DataFrame([input_data_display]).T
    df_display.columns = ['Giá trị']
    st.dataframe(df_display, use_container_width=True)

# ---------------------------------------------------------------------------
# 7.2. Cột 2: Kết quả dự đoán
# ---------------------------------------------------------------------------
with col2:
    st.subheader("🎯 Kết quả dự đoán")
    
    if st.button("🚀 Phân Tích Ngay", type="primary", use_container_width=True):
        try:
            with st.spinner("⏳ Đang phân tích dữ liệu..."):
                # Đường dẫn scaler
                scaler_path = os.path.join(parent_dir, "models", "scaler.pkl")
                
                # Gọi hàm dự đoán
                result = predict_single(model, input_data, scaler_path=scaler_path)
                
                # Lưu kết quả vào session state
                st.session_state.last_prediction = result
                
                prob = result["probability"]
                is_churn = result["prediction"] == 1
                
                # Hiển thị kết quả
                if is_churn:
                    st.error("⚠️ **CẢNH BÁO: Khách hàng có nguy cơ cao rời bỏ!**")
                    st.metric(
                        label="Xác suất Churn", 
                        value=f"{prob:.1%}",
                        delta=f"{prob-0.5:.1%} so với ngưỡng",
                        delta_color="inverse"
                    )
                else:
                    st.success("✅ **AN TOÀN: Khách hàng trung thành**")
                    st.metric(
                        label="Xác suất Churn", 
                        value=f"{prob:.1%}",
                        delta=f"{0.5-prob:.1%} dưới ngưỡng",
                        delta_color="normal"
                    )
                
                # Progress bar
                st.progress(prob)
                
                # Phân tích mức độ rủi ro
                st.markdown("---")
                st.subheader("📊 Phân tích chi tiết")
                
                if prob > 0.7:
                    risk_level = "🔴 Cao"
                elif prob > 0.4:
                    risk_level = "🟡 Trung bình"
                else:
                    risk_level = "🟢 Thấp"
                
                st.write(f"**Mức độ rủi ro:** {risk_level}")
                
                # Phân tích các yếu tố ảnh hưởng
                st.markdown("**Các yếu tố chính:**")
                factors = []
                
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
                
                if factors:
                    for f in factors:
                        st.markdown(f)
                else:
                    st.markdown("• Hồ sơ khách hàng ổn định ✓")
                    
        except FileNotFoundError as e:
            st.error("❌ Không tìm thấy file model hoặc scaler!")
            st.info("💡 Hãy chạy notebook để train model trước")
            st.code(str(e))
            
        except Exception as e:
            st.error(f"❌ Đã xảy ra lỗi: {str(e)}")
            with st.expander("Chi tiết lỗi"):
                st.code(str(e))

# =============================================================================
# PHẦN 8: KHUYẾN NGHỊ HÀNH ĐỘNG
# =============================================================================

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

st.markdown("---")
st.subheader("💡 Khuyến nghị hành động")

if st.session_state.last_prediction is not None:
    result = st.session_state.last_prediction
    prob = result['probability']
    
    if prob > 0.7:
        st.error("### 🔴 HÀNH ĐỘNG KHẨN CẤP - Nguy cơ cao")
        st.markdown("""
        #### 🚨 Ưu tiên cao nhất (trong 24h):
        1. **📞 Liên hệ trực tiếp:** Gọi điện cho khách hàng ngay
        2. **🎁 Ưu đãi VIP:** Giảm 25-30% phí dịch vụ trong 6 tháng
        3. **📝 Chuyển hợp đồng:** Đề xuất hợp đồng 1-2 năm với ưu đãi đặc biệt
        4. **💰 Retention budget:** Miễn phí 1 tháng dịch vụ cao cấp
        """)
        
    elif prob > 0.4:
        st.warning("### 🟡 THEO DÕI SÁT - Nguy cơ trung bình")
        st.markdown("""
        #### 👀 Hành động trong tuần:
        1. **📧 Email cá nhân hóa:** Gửi ưu đãi dựa trên usage pattern
        2. **💳 Incentive:** Giảm 10-15% nếu chuyển sang hợp đồng dài hạn
        3. **🎓 Education:** Giới thiệu các tính năng chưa sử dụng
        4. **📞 Check-in call:** Gọi điện hỏi thăm satisfaction
        """)
        
    else:
        st.success("### 🟢 DUY TRÌ & PHÁT TRIỂN - Khách hàng trung thành")
        st.markdown("""
        #### ⭐ Chiến lược duy trì:
        1. **🏆 Loyalty rewards:** Tích điểm, ưu đãi sinh nhật
        2. **📈 Upsell thông minh:** Đề xuất gói cao cấp phù hợp
        3. **🤝 Referral program:** Thưởng giới thiệu bạn bè
        4. **💎 VIP treatment:** Priority support, early access features
        """)
    
    st.markdown("---")
    st.info("""
    **📊 Tại sao cần can thiệp?**
    
    - Chi phí tìm khách mới = **5x** giữ khách cũ
    - Giảm churn 5% → Tăng lợi nhuận **25-95%**
    - Khách hàng trung thành chi tiêu nhiều hơn **67%**
    """)
else:
    st.info("👆 **Nhập thông tin khách hàng và nhấn 'Phân Tích Ngay' để nhận khuyến nghị chi tiết**")

# =============================================================================
# PHẦN 9: FOOTER
# =============================================================================

st.markdown("---")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Model", "Logistic Regression")

with col_b:
    st.metric("Accuracy", "79.9%")

with col_c:
    st.metric("Features", "19")

st.caption("📝 *Demo system by Streamlit | Model: Logistic Regression | AUC: 0.84*")