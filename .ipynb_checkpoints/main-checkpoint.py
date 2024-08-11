from data import ProstateDataModeling
import streamlit as st
import pandas as pd

# Instantiate ProstateDataModeling class
pipeline = ProstateDataModeling()

# Read dataset
df_prostate = pipeline.read_data("prostate-cancer-data.csv")

# Three pages app
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Exploration', 'Prediction'])  # three pages

# First Page
if app_mode == 'Home':
    st.title("Welcome to ProstateCare+")
    st.image('prostate_image.png')

    # Subheader
    st.write("\n\nProstateCare+ is an innovative tool that uses machine learning model to help you assess your risk for prostate cancer with ease and accuracy.")


elif app_mode == "Exploration":
    st.subheader("Exploratore Prostate Cancer Data")
    st.write("")
    st.write("Here, we present the data behind ProstateCare+ model. We will walk you through the key features of our dataset and you'll find visualizations that highlight the distribution and relationships within the data, helping to make ProstateCare+ more explainable.")
    
    st.subheader('\n\nKey Features of our data')
    # Feature descriptions
    st.write("Click on the features to learn more.")
    # First row with four columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.expander("**lcavol**"):
            st.write("lcavol (log cancer volume): This represents the logarithm of the cancer volume. Cancer volume is a measurement of the size of the tumor in the prostate.")
    
    with col2:
        with st.expander("**lweight**"):
            st.write("lweight (log prostate weight): This is the logarithm of the weight of the prostate.")
    
    with col3:
        with st.expander("**Age**"):
            st.write("Age: The age of the patient.")
    
    with col4:
        with st.expander("**lbph**"):
            st.write("lbph (log benign prostatic hyperplasia): This represents the logarithm of the volume of benign prostatic hyperplasia (BPH). BPH is a noncancerous enlargement of the prostate that can affect urinary function and PSA levels.")
    
    # Second row with four columns
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        with st.expander("**lcp**"):
            st.write("lcp (log capsular penetration): This denotes the logarithm of the extent to which the cancer has penetrated the capsule of the prostate. Capsular penetration indicates a more advanced stage of cancer.")
    
    with col6:
        with st.expander("**Gleason**"):
            st.write("Gleason: The Gleason score is a grading system used to evaluate the prognosis of prostate cancer based on microscopic appearance. Higher Gleason scores indicate more aggressive cancer.")
    
    with col7:
        with st.expander("**pgg45**"):
            st.write("pgg45 (percentage of Gleason scores 4 or 5): This is the percentage of tissue samples that have a Gleason grade of 4 or 5. Higher percentages indicate a more aggressive and advanced cancer.")
    
    with col8:
        with st.expander("**Target**"):
            st.write("Target: Presence or the absence of prostate")

    st.text("")

    # Sub header
    st.subheader("Prevalence of Prostate Cancer by Age groups")
    st.text("")
    st.write("\n\n\nWe have found that prostate cancer data revealed an age-related disparity in cancer incidence. It shows that elderly individuals have a higher prevalence of prostate cancer compared to younger age groups.")
    st.write("")
    # Make a chart of prostate cancer prevalence
    fig = pipeline.stacked_barchart(df_prostate, 'age_group','age','Target')
    st.pyplot(fig)


else:
    st.subheader("Predict Your Prostate Cancer Risk")
    st.write("This tool is designed to help you can get an estimate of your risk for developing prostate cancer based on our model in order to help you make informed decisions.")
    st.write("")
    st.write("Please enter the required clinical parameters in the sidebar. These inputs will be used to generate your risk prediction. Ensure all fields are filled in accurately for the most reliable results.")

    # Sidebar for user input
    st.sidebar.header("Input Features for Prediction")
    
    
    # Collecting input data from the user
    lcavol = st.sidebar.number_input("lcavol", min_value=-100.00, max_value=105.0, value=0.0, step=0.1, format="%.6f")
    lweight = st.sidebar.number_input("lweight", min_value=-100.0, max_value=105.0, value=0.0, step=0.1, format="%.6f")
    age = st.sidebar.number_input("Age", min_value=0.0, max_value=100.0, value=0.0, step=1.0, format="%.6f")
    lbph = st.sidebar.number_input("lbph", min_value=-100.0, max_value=105.0, value=0.0, step=1.0, format="%.6f")
    lcp = st.sidebar.number_input("lcp", min_value=-100.0, max_value=105.0, value=0.0, step=1.0, format="%.6f")
    gleason = st.sidebar.number_input("gleason", min_value=-10.0, max_value=105.0, value=0.0, step=1.0, format="%.6f")
    pgg45 = st.sidebar.number_input("ppg45", min_value=-100.0, max_value=105.0, value=0.0, step=1.0, format="%.6f")
    lpsa = st.sidebar.number_input("lpsa", min_value=-100.0, max_value=105.0, value=0.0, step=1.0, format="%.6f")

    
    # Prepare Input Features as a dictionary
    input_features = pd.DataFrame({
        "lcavol": [lcavol],
        "lweight": [lweight],
        "age": [age],
        "lbph": [lbph],
        "lcp": [lcp],
        "gleason": [gleason],
        "pgg45": [pgg45],
        "lpsa": [lpsa]
    })

    
    # Data modeling pipeline
    @st.cache_data
    def predict(_model, new_data):
        
        # Split data
        _model.split_data(df_prostate, target_column="Target")
    
        # Scale data
        _model.scale_data()
    
        # Train model
        _model.train_model()
    
        # Evaluate model
        accuracy = _model.evaluate_model()
    
        # Make prediction
        prediction = _model.make_predictions(new_data)
        
        return prediction
    
    # Assuming `pipeline` is your model instance and `input_features` is the new data
    prediction = predict(pipeline, input_features)

    # Add a button to trigger the prediction
    if st.button("Make Prediction"):
            
        # Format output
        if prediction == 1:
            st.write("**Prediction Result:**", prediction)
            st.write("The model predicts a **high likelihood** of prostate cancer based on the input features. It's recommended to consult with a healthcare professional for further evaluation and confirmation.")
        else:
            st.write("**Prediction Result**:", prediction)
            st.write("The model predicts a **low likelihood** of prostate cancer based on the input features. While this is a positive result, it's important to continue regular check-ups and screenings as recommended by your healthcare provider.")
        


    