import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
load_model = tf.keras.models.load_model
from PIL import Image
import tensorflow as tf

# --- Cache model loading function ---
@st.cache_resource
def get_ml_model():
    return joblib.load('horse_feature_model.pkl')

@st.cache_resource
def get_nn_model():
    return tf.keras.models.load_model('horse_image_model.h5')

# --- Additional information (Constants) ---
CLASSES = ['Alicorn', 'Horse', 'Pegasus', 'Unicorn', 'Zebra']

# --- Screen settings ---
st.set_page_config(page_title="Horse Family AI Project", layout="wide")

# --- Managing Navigation (Session State) ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def navigate_to(page):
    st.session_state.page = page

# --- Additional information (Constants) ---
CLASSES = ['Alicorn', 'Horse', 'Pegasus', 'Unicorn', 'Zebra']

# --- Homepage ---
if st.session_state.page == 'Home':
    st.title("🐴 Horse Family Classification Project")
    st.subheader("ยินดีต้อนรับสู่โปรเจกต์จำแนกสัตว์ตระกูลม้าด้วย AI")
    
    st.markdown("""
    โปรเจกต์นี้จัดทำขึ้นเพื่อศึกษาและเปรียบเทียบประสิทธิภาพระหว่าง 
    **Machine Learning (Ensemble Model)** และ **Neural Network (Deep Learning)** ในการแยกแยะม้าสายพันธุ์ปกติและสัตว์ในตำนาน
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 🤖 Machine Learning")
        st.write("วิเคราะห์จาก 'ลักษณะทางกายภาพ' (มีเขา, มีปีก, มีลาย)")
        if st.button("ไปที่โมเดล ML"): navigate_to('ML_Info')
        
    with col2:
        st.success("### 🧠 Neural Network")
        st.write("วิเคราะห์จาก 'รูปภาพ' โดยตรงด้วยระบบ Deep Learning")
        if st.button("ไปที่โมเดล NN"): navigate_to('NN_Info')

# --- ML INFO ---
elif st.session_state.page == 'ML_Info':
    st.title("📚 รายละเอียดการพัฒนา Machine Learning")
    
    st.markdown("""
    ### 1. แรงบันดาลใจ (Motivation)
    ต้องการสร้างระบบคัดกรองข้อมูลสัตว์ตระกูลม้าแบบรวดเร็วโดยใช้ข้อมูลลักษณะเด่น (Features) 
    แทนการใช้ภาพถ่าย เพื่อความรวดเร็วในการประมวลผลบนอุปกรณ์ที่สเปกไม่สูง
    
    ### 2. ทฤษฎีและอัลกอริทึ่ม (Ensemble Learning)
    เราใช้เทคนิค **Voting Classifier** ซึ่งเป็นการรวมพลังของ 3 โมเดล:
    * **Random Forest:** ใช้ Decision Tree หลายต้นลดความแปรปรวน
    * **XGBoost:** ใช้ระบบ Boosting เพื่อรีดประสิทธิภาพจากข้อผิดพลาด
    * **Logistic Regression:** ช่วยสร้างเส้นแบ่งข้อมูลเชิงเส้นเพื่อความเสถียร
    
    ### 3. ขั้นตอนการเตรียมข้อมูล
    * สร้าง Dataset จากการสังเกตลักษณะเด่น: `has_horn`, `has_wings`, `is_striped`
    * ทำการ Clean ข้อมูลที่หายไปด้วยวิธี Mode Imputation (เติมค่าที่พบบ่อย)
    * จัดเก็บข้อมูลในรูปแบบ CSV เพื่อใช้เป็นตารางฟีเจอร์
    
    ### 4. แหล่งอ้างอิง
    * ข้อมูลลักษณะสัตว์ตำนานจากสารานุกรมออนไลน์และเทพปกรณัมกรีก
    """)
    
    if st.button("➡️ ไปหน้าทดลอง ML"): navigate_to('ML_Test')

# --- ML TEST ---
elif st.session_state.page == 'ML_Test':
    st.title("🧪 ทดลองใช้งาน ML Model (Features)")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'horse_feature_model.pkl')

    # Input Features Section
    col1, col2, col3 = st.columns(3)
    with col1: horn = st.selectbox("มีเขา?", [0, 1], format_func=lambda x: "มี" if x==1 else "ไม่มี")
    with col2: wings = st.selectbox("มีปีก?", [0, 1], format_func=lambda x: "มี" if x==1 else "ไม่มี")
    with col3: stripes = st.selectbox("มีลาย?", [0, 1], format_func=lambda x: "มี" if x==1 else "ไม่มี")

    if st.button("Predict"):
        try:
            model = joblib.load(model_path)
            input_df = pd.DataFrame([[horn, wings, stripes]], columns=['has_horn', 'has_wings', 'is_striped'])
            
            # Calculate the confidence level (Probability)
            probs = model.predict_proba(input_df)
            max_prob = np.max(probs)
            prediction = model.predict(input_df)[0]

            if max_prob < 0.5:
                st.warning(f"⚠️ ความแม่นยำต่ำเกินไป ({max_prob:.2%})")
                st.error("ฉันไม่คิดว่าข้อมูลที่คุณให้มาจะเป็นสัตว์ใน dataset ของฉันนะ (ม้า, ม้าลาย, ยูนิคอร์น, เพกาซัส หรือเอลิคอร์น)")
            else:
                st.balloons()
                st.success(f"ผลลัพธ์คือ: **{prediction}** (ความมั่นใจ: {max_prob:.2%})")
        except:
            st.error(f"ยังหาไฟล์ไม่เจอ: {model_path}")
            st.write("ไฟล์ที่มีในโฟลเดอร์นี้คือ:", os.listdir(current_dir))

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a: 
        if st.button("🏠 กลับหน้าแรก"): navigate_to('Home')
    with col_b: 
        if st.button("🧠 ไปหน้า NN"): navigate_to('NN_Info')

# --- NN INFO ---
elif st.session_state.page == 'NN_Info':
    st.title("📚 รายละเอียดการพัฒนา Neural Network")
    
    st.markdown("""
    ### 1. แนวทางการพัฒนา
    ใช้เทคนิค **Transfer Learning** โดยใช้โมเดลพื้นฐานระดับโลกอย่าง **MobileNetV2** มาทำการปรับจูนให้เข้ากับรูปภาพสัตว์ในโปรเจกต์ของเรา
    
    ### 2. ทฤษฎีของอัลกอริทึ่ม (Deep Learning)
    * **CNN (Convolutional Neural Network):** ใช้สกัดฟีเจอร์จากพิกเซลภาพ
    * **Freezing Layers:** แช่แข็งความรู้เดิมจาก ImageNet ไว้เพื่อไม่ให้สูญเสียความแม่นยำพื้นฐาน
    * **Softmax Activation:** ใช้ในเลเยอร์สุดท้ายเพื่อแปลงค่าเป็นความน่าจะเป็นของ 5 สายพันธุ์
    
    ### 3. การเตรียมข้อมูลรูปภาพ
    * รวบรวมภาพรวม 490+ ภาพ แบ่งเป็น 5 หมวดหมู่
    * ทำ **Image Scaling (1/255)** เพื่อปรับค่าสีให้อยู่ในช่วง 0-1
    * ทำ **Image Augmentation** (หมุน, พลิกภาพ) เพื่อให้โมเดลมองภาพได้หลายมุมมอง
    
    ### 4. แหล่งที่มาข้อมูล
    * รวบรวมจาก Google Images และ Dataset สาธารณะ
    """)
    
    if st.button("➡️ ไปหน้าทดลอง NN"): navigate_to('NN_Test')

# --- NN TEST ---
elif st.session_state.page == 'NN_Test':
    st.title("🧪 ทดลองใช้งาน NN Model (Image)")
    
    uploaded_file = st.file_uploader("อัปโหลดรูปภาพเพื่อทำนาย...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="ภาพที่อัปโหลด", width=300)
        
        if st.button("Analyze Image"):
            try:
                model_nn = get_nn_model()
                
                # Prepare an image that matches the training size (224x224)
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                
                # predictions
                predictions = model_nn.predict(img_array)
                max_score = np.max(predictions)
                class_idx = np.argmax(predictions)
                result_label = CLASSES[class_idx]

                if max_score < 0.5:
                    st.warning(f"⚠️ ค่าความมั่นใจต่ำ ({max_score:.2%})")
                    st.error("ฉันไม่คิดว่าข้อมูลที่คุณให้มาจะเป็นสัตว์ใน dataset ของฉันนะ (ม้า, ม้าลาย, ยูนิคอร์น, เพกาซัส หรือเอลิคอร์น)")
                else:
                    st.balloons()
                    st.success(f"นี่คือ: **{result_label}** (ความแม่นยำ: {max_score:.2%})")
            except:
                st.error("ไม่พบไฟล์โมเดล NN หรือรูปแบบภาพไม่ถูกต้อง")

    st.divider()
    col_x, col_y = st.columns(2)
    with col_x: 
        if st.button("🏠 กลับหน้าแรก"): navigate_to('Home')
    with col_y: 
        if st.button("🤖 ไปหน้า ML"): navigate_to('ML_Info')