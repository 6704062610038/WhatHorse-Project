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
from tensorflow.keras.models import load_model

# --- Cache model loading function ---
@st.cache_resource
def get_ml_model():
    model_path = os.path.join(os.getcwd(), 'horse_feature_model.pkl')
    return joblib.load(model_path)

@st.cache_resource
def get_nn_model():
    model_path = os.path.join(os.getcwd(), 'horse_image_model.h5')
    return tf.keras.models.load_model(model_path, compile=False)

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
    st.subheader("This project aims to test how AI models—both feature-based (ML) and image-based (NN)—can accurately differentiate between these closely related biological and mythical forms.")
    
    st.markdown("""
    **The inspiration for this project** began with a fundamental challenge in Computer Vision: object classification. While classifying distinct objects is a standard task, I was intrigued by the challenge of Fine-Grained Classification—distinguishing between entities with highly similar physical structures.
    """)
    st.markdown("""
    **Starting from a base of common equines** like Horses and Zebras, I expanded the scope into the mythical realm. This led to a fascinating progression: if a horse has a horn, it's a Unicorn; if it has wings, it's a Pegasus. But what if it possesses both? This curiosity introduced the Alicorn into our dataset. This project aims to test how AI models—both feature-based (ML) and image-based (NN)—can accurately differentiate between these closely related biological and mythical forms.
    """)
    st.markdown(""" """)
    st.markdown("""
    ### Model Development Process
    * **Phase 1: Problem Definition & Scope**
        * **Objective:** To design a system capable of distinguishing between 5 closely related equine species: Horse, Zebra, Unicorn, Pegasus, and Alicorn.
        * **Approach:** Implementing a hybrid solution using Feature-based ML and Image-based Neural Networks.
    * **Phase 2: Data Acquisition & Preprocessing**
        * **Machine Learning Data Generation & Simulation:** I developed a script to extract physical attributes (Horns, Wings, Stripes) into a structured CSV format. To demonstrate robust data handling, I intentionally introduced missing values via a randomization function, simulating real-world incomplete data scenarios, which were then resolved using Mode Imputation.
        * **Neural Network Data Curation:** A specialized image dataset of over 490+ samples was curated. I implemented Image Augmentation techniques (such as rotation and flipping) to diversify the training data and enhance the model's ability to generalize to new images.
        * **Normalization & Encoding:** All inputs were standardized. I applied Pixel Scaling for image data and Label Encoding for categorical targets, ensuring the data was in the optimal format for model consumption.
    * **Phase 3: Model Architecture & Training**
        * **Machine Learning Implementation:** Developed an Ensemble Voting Classifier (Random Forest, XGBoost, and Logistic Regression) to ensure robust logic-based predictions.
        * **Neural Network Implementation:** Leveraged Transfer Learning with the MobileNetV2 architecture, fine-tuning the top layers for our specific 5-class classification task.
    * **Phase 4: Evaluation & Optimization**
        * **Performance Metrics:** Evaluated models based on Accuracy and Confidence Scores.
        * **Confidence Thresholding:** Implemented a 70% Threshold to filter out uncertain predictions (Out-of-Distribution data), significantly reducing false positives.
    * **Phase 5: Web Integration & Deployment**
        * **UI/UX Design:** Built an interactive web application using Streamlit to provide a seamless user experience for both ML and NN testing.
        * **Cloud Deployment:** Uploaded the project to GitHub and deployed via Streamlit Cloud for global accessibility.
    """)
    st.markdown(""" """)
    st.markdown(""" """)

    col1, col2 = st.columns([2,2])
    with col1:
        st.info("### 🤖  Machine Learning (feature-based)")
        c1, c2 ,c3= st.columns([0.6,1.5,0.2])
        with c2:
            if st.button("Go to Machine Learning model"): navigate_to('ML_Info')
        
    with col2:
        st.success("### 🧠  Neural Network (image-based)")
        c1, c2 ,c3= st.columns([0.6,1.5,0.2])
        with c2:
            if st.button("Go to Neural Network model"): navigate_to('NN_Info')
    
    st.markdown(""" """)
    st.markdown(""" """)

    st.markdown("""
    ### References
    * **Some horses and zebra Image :** Curated from [Donkeys, Horses & Zebra Images Dataset](https://www.kaggle.com/datasets/ifeanyinneji/donkeys-horses-zebra-images-dataset?resource=download).
    * **My little pony characters :** Curated from [My little pony character](https://mlp.fandom.com/wiki/Characters#Main_characters).
    * **Other Data Sourcing:** Supplementary images were curated via [Google Images](https://images.google.com) under Creative Commons licenses, specifically searching for diverse equine variations.
    """)
    

# --- ML INFO ---
elif st.session_state.page == 'ML_Info':
    st.title("📚 Details on Machine Learning Development")
    st.markdown(""" """)
    st.markdown("""
    ### 1. Why Machine Learning?
    While images provide a rich data source, I wanted to explore a logic-based approach. By extracting specific physical attributes—such as the presence of horns, wings, or stripes I developed an Ensemble Learning model. This allows for a lightweight and interpretable system that can classify species based on descriptive data rather than pixels, mimicking how a taxonomist might categorize a new discovery.
    """)
    st.markdown("""
    ### 2. Feature Engineering
    * **Automated Dataset Construction:** I developed a specialized script to automate the extraction of key physical attributes, including has_horn, has_wings, and is_striped, accurately mapping them to each of the 5 equine categories.
    * **Missing Data Simulation:** To demonstrate advanced data handling, I intentionally introduced randomized missing values (NaN) within the feature matrix. This process was designed to simulate the challenges of real-world incomplete datasets.
    * **Data Integrity & Cleaning:** I resolved these simulated gaps using Mode Imputation, replacing null values with the most frequent data points to maintain consistent logic and ensure high-quality input for the models.
    * **Data Serialization:** The final cleaned feature set was structured and exported in CSV format, serving as the primary source of truth for our Machine Learning ensemble.
    """)
    st.markdown("""
    ### 3. Ensemble Learning Approach
    For this project, I implemented an **Ensemble Learning** technique specifically using a **Voting Classifier (Soft Voting)**. This approach acts like a "committee of experts," where three distinct models collaborate to provide the most accurate prediction by leveraging their individual strengths.

    * **Random Forest:** 
        * **Concept:** An ensemble of many **Decision Trees** operating as a committee. It uses a technique called **Bagging** (Bootstrap Aggregating).
        * **Why I use it:** Since our equine dataset relies on specific binary features (e.g., *has_horn*, *has_wings*), Random Forest is excellent for **reducing variance**. By training multiple trees on different subsets of data and averaging their results, it prevents **Overfitting** and ensures the model doesn't get confused by "noise" or outliers in the data.
    * **XGBoost:**
        * **Concept:** A powerful implementation of **Gradient Boosting** that builds trees sequentially. Each new tree focuses on **correcting the errors** made by the previous ones.
        * **Why I use it:** For complex mythological creatures like the **Alicorn** (which shares features with both Unicorns and Pegasi), XGBoost excels at capturing subtle patterns. It "squeezes" out maximum performance and optimization, making it highly effective at distinguishing between classes that have overlapping physical traits.
    * **Logistic Regression:**
        * **Concept:** Despite its name, this is a fundamental **Classification** algorithm that uses a Sigmoid function to establish a "Decision Boundary" between classes.
        * **Why I use it:** I included Logistic Regression as a **Stability Anchor**. While tree-based models (like RF and XGB) are great for non-linear patterns, they can sometimes become overly complex. Logistic Regression provides a simple, linear perspective that balances the ensemble, ensuring the overall prediction remains grounded and reliable.
    """)
    
    st.markdown(""" """)
    st.markdown(""" """)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Go to Machine Learning test", use_container_width=True):
            navigate_to('ML_Test')

# --- ML TEST ---
elif st.session_state.page == 'ML_Test':
    st.title("🧪 Machine Learning model testing (Feature-based)")
    
    import pathlib
    current_dir = pathlib.Path(__file__).parent.resolve()
    model_path = current_dir / 'horse_feature_model.pkl'
    st.markdown(""" """)

    # Input Features Section
    col1, col2, col3 = st.columns(3)
    with col1: horn = st.selectbox("Has horn ?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with col2: wings = st.selectbox("Has wings ?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with col3: stripes = st.selectbox("Is striped ?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    st.markdown(""" """)
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
        if st.button("Predict", use_container_width=True):
            if not model_path.exists():
                st.error(f"❌ The system could not find the file at {model_path}")
                st.write("Please double-check the file name.")
            else:
                try:
                    with open(model_path, 'rb') as f:
                        model = joblib.load(f)
                
                    input_df = pd.DataFrame([[horn, wings, stripes]], 
                                     columns=['has_horn', 'has_wings', 'is_striped'])
                
                    probs = model.predict_proba(input_df)
                    max_prob = np.max(probs)
                    prediction = model.predict(input_df)[0]
                    
                    if max_prob < 0.5:
                        st.warning(f"⚠️ Low accuracy level: {max_prob:.2%}")
                        st.error("Unknown species in our dataset.")
                    else:
                        st.balloons()
                        st.success(f"The result is: **{prediction}** (accuracy: {max_prob:.2%})")

                except Exception as e:
                    st.error(f"Error: {e}")
    st.divider()
    a1, a2, a3, a4, a5 = st.columns([1, 0.8, 0.2, 0.8, 1])
    with a2: 
        if st.button("🏠 Go to Homepage", use_container_width=True): navigate_to('Home')
    with a4: 
        if st.button("🧠 Go to Neural Network", use_container_width=True): navigate_to('NN_Info')

# --- NN INFO ---
elif st.session_state.page == 'NN_Info':
    st.title("📚 Details on Neural Network Development")
    st.markdown(""" """)
    st.markdown("""
    ### 1. Why Neural Network?
    To push the boundaries of this project, I implemented a Deep Learning approach using Transfer Learning (MobileNetV2). The goal was to see if a neural network could 'see' the subtle visual differences that define each creature. By training the model on hundreds of images, I aimed to create a system capable of autonomous visual recognition, identifying an Alicorn from a Pegasus with the same visual intuition as a human expert.
    """)
    st.markdown("""
    ### 2. Image Feature Engineering (NN)
    * **Automated Image Resizing:**
        * **Concept:** I developed a preprocessing pipeline to automatically transform all raw images into a standardized **224x224 pixel** resolution.
        * **Why I use it:** Neural Networks like **MobileNetV2** require a fixed input shape. This automation ensures that every image—regardless of its original size—is perfectly formatted for the model's first layer, preventing spatial distortion during analysis.
    * **Min-Max Pixel Scaling:**
        * **Concept:** I implemented a scaling script to divide all pixel values (0-255) by **255**, effectively normalizing the data into a floating-point range between **0.0 and 1.0**.
        * **Why I use it:** Raw pixels are too large for efficient weight updates. Scaling prevents **Gradient Explosion** and ensures the model converges faster during training, making the mathematical operations within the network more stable and accurate.
    * **Dynamic Image Augmentation:**
        * **Concept:** I utilized a real-time augmentation script to apply random **Rotations, Horizontal Flips, and Zooms** to the training images.
        * **Why I use it:** Since our dataset is specialized (490+ images), this technique "simulates" a much larger dataset. It teaches the model to recognize a **Unicorn** or **Pegasus** from any angle or perspective, significantly improving **Generalization** and reducing **Overfitting**.
    * **Label Encoding for Classes:**
        * **Concept:** I mapped the 5 equine categories (Horse, Zebra, Unicorn, Pegasus, and Alicorn) into a numeric format that the **Softmax** layer can interpret.
        * **Why I use it:** Deep Learning models cannot process text strings. By converting species names into a **Probability Vector**, the model can calculate the specific **Confidence Score** for each category, which is essential for our system's final decision logic.
    """)
    st.markdown("""
    ### 3. Deep Learning Theory & Architecture (NN)
    * **MobileNetV2 (CNN Architecture):**
        * **Concept:** A specialized **Convolutional Neural Network (CNN)** designed for mobile and web-based efficiency. It uses **Depthwise Separable Convolutions** to process image pixels (edges, shapes, and textures) with significantly fewer parameters than standard CNNs.
        * **Why I use it:** Since we are deploying this on a web interface (Streamlit), we need a model that is **Lightweight and Fast**. MobileNetV2 provides high accuracy for image classification while maintaining low latency, ensuring users get instant predictions without heavy server loads.
    * **Transfer Learning & Freezing Layers:**
        * **Concept:** A technique where a model developed for one task (ImageNet) is reused as the starting point for a second task. **Freezing** means locking the weights of the initial layers so they don't change during our specific training.
        * **Why I use it:** Our dataset is relatively small (approx. 500 images). Training a deep network from scratch would lead to **Overfitting**. By freezing the "knowledge" of 1.4 million images from ImageNet, we retain the ability to recognize basic visual features (like animal legs or fur) and only need to train the final layers to distinguish our 5 specific equine classes.
    * **Softmax Activation Function:**
        * **Concept:** A mathematical function used in the final **Output Layer** that turns raw numerical scores (logits) into a **Probability Distribution** (ranging from 0 to 1).
        * **Why I use it:** We need to know not just "what" the animal is, but "how sure" the model is. Softmax allows us to see the **Confidence Percentage** for all 5 classes (Horse, Zebra, Unicorn, Pegasus, and Alicorn). This is crucial for our **70% Threshold** logic—if the highest Softmax probability is too low, the system can safely say it "doesn't know," preventing false guesses.
    """)
    st.markdown(""" """)
    st.markdown(""" """)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Go to Neural Network test", use_container_width=True):
            navigate_to('NN_Test')

# --- NN TEST ---

elif st.session_state.page == 'NN_Test':
    st.title("🧪 Neural Network model testing (Image-based)")
    
    uploaded_file = st.file_uploader("Upload an image to perform a prediction...", type=['jpg', 'jpeg', 'png'])

    st.markdown(""" """)
    col1, col2, col3, col4, col5 = st.columns([1, 1.5, 1.5, 1.5, 1])
    with col3:
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded images")
        
        if st.button("Analyze Image", use_container_width=True):
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

                if max_score < 0.7:
                    st.warning(f"⚠️ Low accuracy level ({max_score:.2%})")
                    st.error("No matching species found. The system is unable to categorize the input based on the existing Horse, Zebra, Unicorn, Pegasus, and Alicorn datasets.")
                else:
                    st.balloons()
                    st.success(f"This is: **{result_label}** (accuracy: {max_score:.2%})")
            except:
                st.error("The Neural Network model file was not found or the image format is incorrect.")

    st.divider()
    a1, a2, a3, a4, a5 = st.columns([1, 0.8, 0.2, 0.8, 1])
    with a2: 
        if st.button("🏠 Go to Homepage", use_container_width=True): navigate_to('Home')
    with a4: 
        if st.button("🤖 Go to Machine Learning", use_container_width=True): navigate_to('ML_Info')
