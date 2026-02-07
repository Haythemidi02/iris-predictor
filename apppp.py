"""
app.py - Streamlit Web Interface for Iris Flower Prediction

This file creates an interactive web application for predicting iris flower species.
Run with: streamlit run app.py
"""

# Import required libraries
import streamlit as st  # Main Streamlit library for creating the web app
import pandas as pd     # For data manipulation and display
import numpy as np      # For numerical operations
import joblib          # For loading the saved model
import matplotlib.pyplot as plt  # For creating visualizations

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
# This MUST be the first Streamlit command
# It sets up the page title, icon, and layout
st.set_page_config(
    page_title="Iris Flower Predictor",  # Shows in browser tab
    page_icon="🌸",                       # Icon in browser tab
    layout="wide",                        # Use full width of the page
    initial_sidebar_state="expanded"      # Sidebar open by default
)

# ============================================================================
# LOAD MODEL (with caching for performance)
# ============================================================================
@st.cache_resource  # This decorator caches the model so it's only loaded once
def load_model():
    """
    Load the trained model and model information from disk.
    The @st.cache_resource decorator ensures this only runs once,
    not every time the user interacts with the app.
    """
    try:
        model = joblib.load('iris_model.pkl')
        model_info = joblib.load('model_info.pkl')
        return model, model_info
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run 'python train_model.py' first.")
        st.stop()

# Load the model when the app starts
model, model_info = load_model()

# ============================================================================
# CUSTOM CSS STYLING (optional but makes the app look better)
# ============================================================================
st.markdown("""
    <style>
    /* Style for the main title */
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Style for prediction result box */
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
    }
    
    /* Style for info boxes */
    /* Style for info boxes */
    .info-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #e3f2fd;  /* Light blue background */
        border-left: 4px solid #1f77b4;  /* Blue left border */
        margin: 10px 0;
        color: #1a1a1a;  /* Dark text */
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown('<h1 class="main-title">🌸 Iris Flower Species Predictor</h1>', 
            unsafe_allow_html=True)

# Add description
st.markdown("""
<div class="info-box">
This application uses a <b>Random Forest Classifier</b> to predict the species of an Iris flower 
based on its physical measurements. Simply adjust the sliders below and click <b>Predict</b> 
to see which species your flower belongs to!
</div>
""", unsafe_allow_html=True)

# Add a separator
st.markdown("---")

# ============================================================================
# MAIN LAYOUT - Two Columns
# ============================================================================
# Create two columns: left for inputs, right for predictions
col1, col2 = st.columns([1, 1])

# ============================================================================
# LEFT COLUMN - INPUT FEATURES
# ============================================================================
with col1:
    st.header("📊 Input Features")
    st.markdown("Adjust the sliders to match your flower's measurements:")
    
    # Create a slider for Sepal Length
    # st.slider(label, min_value, max_value, default_value, step)
    sepal_length = st.slider(
        'Sepal Length (cm)',
        min_value=4.0,        # Minimum possible value
        max_value=8.0,        # Maximum possible value
        value=5.8,            # Default starting value
        step=0.1,             # How much the slider moves each step
        help="The length of the sepal (outer part of the flower) in centimeters"
    )
    
    # Create a slider for Sepal Width
    sepal_width = st.slider(
        'Sepal Width (cm)',
        min_value=2.0,
        max_value=4.5,
        value=3.0,
        step=0.1,
        help="The width of the sepal in centimeters"
    )
    
    # Create a slider for Petal Length
    petal_length = st.slider(
        'Petal Length (cm)',
        min_value=1.0,
        max_value=7.0,
        value=4.3,
        step=0.1,
        help="The length of the petal (inner colored part) in centimeters"
    )
    
    # Create a slider for Petal Width
    petal_width = st.slider(
        'Petal Width (cm)',
        min_value=0.1,
        max_value=2.5,
        value=1.3,
        step=0.1,
        help="The width of the petal in centimeters"
    )
    
    # Display current input values in a nice table
    st.subheader("Current Measurements:")
    input_df = pd.DataFrame({
        'Feature': model_info['feature_names'],
        'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
    })
    
    # Display the table with custom styling
    st.dataframe(
        input_df.style.highlight_max(axis=0, color='lightgreen'),
        use_container_width=True,
        hide_index=True
    )

# ============================================================================
# RIGHT COLUMN - PREDICTIONS
# ============================================================================
with col2:
    st.header("🎯 Prediction Results")
    
    # Prepare the input data for the model
    # The model expects a 2D array with shape (n_samples, n_features)
    # We have 1 sample and 4 features, so shape is (1, 4)
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Create a predict button
    # The prediction only happens when this button is clicked
    if st.button('🔮 Predict Species', type="primary", use_container_width=True):
        
        # Make prediction using the loaded model
        prediction = model.predict(input_features)
        
        # Get prediction probabilities for each class
        # This shows how confident the model is about each species
        prediction_proba = model.predict_proba(input_features)
        
        # Get the name of the predicted species
        predicted_species = model_info['target_names'][prediction[0]]
        
        # Display the prediction in a styled box
        st.markdown(
            f'<div class="prediction-box">Predicted Species: {predicted_species.upper()}</div>',
            unsafe_allow_html=True
        )
        
        # ========================================================================
        # CONFIDENCE VISUALIZATION
        # ========================================================================
        st.subheader("📊 Prediction Confidence")
        
        # Create a DataFrame with probabilities
        prob_df = pd.DataFrame({
            'Species': model_info['target_names'],
            'Probability (%)': prediction_proba[0] * 100  # Convert to percentage
        }).sort_values('Probability (%)', ascending=False)
        
        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Create bars with different colors
        colors = ['#2ecc71' if species == predicted_species else '#3498db' 
                  for species in prob_df['Species']]
        
        bars = ax.barh(prob_df['Species'], prob_df['Probability (%)'], color=colors)
        
        # Customize the chart
        ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Confidence for Each Species', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 100])
        ax.grid(axis='x', alpha=0.3)
        
        # Add percentage labels on the bars
        for i, (species, prob) in enumerate(zip(prob_df['Species'], prob_df['Probability (%)'])):
            ax.text(prob + 2, i, f'{prob:.1f}%', va='center', fontweight='bold')
        
        # Display the chart in Streamlit
        st.pyplot(fig)
        
        # Also display the probabilities as a table
        st.subheader("📋 Detailed Probabilities")
        st.dataframe(
            prob_df.style.background_gradient(cmap='RdYlGn', subset=['Probability (%)']),
            use_container_width=True,
            hide_index=True
        )
        
        # ========================================================================
        # INTERPRETATION HELP
        # ========================================================================
        st.subheader("💡 What does this mean?")
        
        max_prob = prob_df['Probability (%)'].max()
        
        if max_prob > 90:
            confidence_text = "The model is **very confident** about this prediction!"
        elif max_prob > 70:
            confidence_text = "The model is **fairly confident** about this prediction."
        else:
            confidence_text = "The model is **somewhat uncertain**. The measurements might be borderline between species."
        
        st.info(confidence_text)

# ============================================================================
# SIDEBAR - Additional Information and Features
# ============================================================================
with st.sidebar:
    # About section
    st.header("ℹ️ About This App")
    st.markdown("""
    This application demonstrates how to deploy a machine learning model 
    using **Streamlit**.
    
    **Model Information:**
    - **Algorithm:** Random Forest Classifier
    - **Dataset:** Iris Flower Dataset (150 samples)
    - **Features:** 4 measurements
    - **Classes:** 3 species
    - **Accuracy:** {:.1f}%
    """.format(model_info.get('accuracy', 0.967) * 100))
    
    # Species information
    st.header("🌺 Iris Species Guide")
    
    # Expandable sections for each species
    with st.expander("🌸 Setosa"):
        st.markdown("""
        **Characteristics:**
        - Smallest flowers
        - Short, wide petals
        - Very distinct from other species
        - Easy to identify
        """)
    
    with st.expander("🌼 Versicolor"):
        st.markdown("""
        **Characteristics:**
        - Medium-sized flowers
        - Moderate petal length and width
        - Can overlap with Virginica
        - Intermediate features
        """)
    
    with st.expander("🌺 Virginica"):
        st.markdown("""
        **Characteristics:**
        - Largest flowers
        - Long, narrow petals
        - Can overlap with Versicolor
        - Distinctive when fully grown
        """)
    
    # Divider
    st.markdown("---")
    
    # ========================================================================
    # BATCH PREDICTION FEATURE
    # ========================================================================
    st.header("📁 Batch Prediction")
    st.markdown("Upload a CSV file to predict multiple flowers at once:")
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV should have 4 columns: sepal_length, sepal_width, petal_length, petal_width"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read the CSV file
            batch_data = pd.read_csv(uploaded_file)
            
            st.success(f"✓ File uploaded! Found {len(batch_data)} samples.")
            
            # Check if the file has the correct number of columns
            if batch_data.shape[1] == 4:
                # Make predictions for all rows
                batch_predictions = model.predict(batch_data.values)
                batch_probabilities = model.predict_proba(batch_data.values)
                
                # Add predictions to the dataframe
                batch_data['Predicted Species'] = [
                    model_info['target_names'][p] for p in batch_predictions
                ]
                
                # Add confidence scores
                batch_data['Confidence (%)'] = [
                    max(probs) * 100 for probs in batch_probabilities
                ]
                
                # Display results
                st.subheader("Results:")
                st.dataframe(batch_data, use_container_width=True)
                
                # Provide download button for results
                csv = batch_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name='iris_predictions.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                
                # Show summary statistics
                st.subheader("Summary:")
                species_counts = batch_data['Predicted Species'].value_counts()
                st.bar_chart(species_counts)
                
            else:
                st.error(f"❌ Error: CSV file should have exactly 4 columns, but found {batch_data.shape[1]}.")
                st.info("Expected columns: sepal_length, sepal_width, petal_length, petal_width")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 20px;">
    <p><b>Built with Streamlit 🎈</b></p>
    <p>A tutorial on deploying ML models with interactive web interfaces</p>
    <p><i>Learn more at <a href="https://docs.streamlit.io" target="_blank">docs.streamlit.io</a></i></p>
</div>
""", unsafe_allow_html=True)
