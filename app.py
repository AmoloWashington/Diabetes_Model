# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Diabetes Symptoms Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4169E1;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
    }
    .section {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("This app analyzes diabetes symptoms data and predicts the likelihood of diabetes based on input parameters.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes_symptoms_data.csv')
    return df

# Function for EDA
def explore_data(df):
    st.markdown("<h2 class='sub-header'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    
    # Show first few rows
    if st.checkbox("Show sample data"):
        st.write(df.head())
    
    # Basic info
    if st.checkbox("Dataset information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset shape:**", df.shape)
            st.write("**Missing values:**", df.isnull().sum().sum())
            
        with col2:
            # Class distribution
            class_counts = df['class'].value_counts()
            st.write("**Class distribution:**")
            st.write(f"Positive: {class_counts['Positive']} ({class_counts['Positive']/len(df):.1%})")
            st.write(f"Negative: {class_counts['Negative']} ({class_counts['Negative']/len(df):.1%})")
    
    # Gender distribution
    if st.checkbox("Gender distribution"):
        fig = px.pie(df, names='gender', title='Gender Distribution', 
                     color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig)
    
    # Age distribution
    if st.checkbox("Age distribution"):
        fig = px.histogram(df, x="age", color="class", marginal="box", 
                           title="Age Distribution by Diabetes Status",
                           color_discrete_map={"Positive": "#1E90FF", "Negative": "#D3D3D3"})
        st.plotly_chart(fig)
    
    # Symptoms distribution
    if st.checkbox("Symptoms distribution"):
        # Create a melted dataframe for symptoms
        symptom_cols = ['polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness', 
                         'polyphagia', 'genital_thrush', 'visual_blurring', 'itching', 
                         'irritability', 'delayed_healing', 'partial_paresis', 
                         'muscle_stiffness', 'alopecia', 'obesity']
        
        symptoms_by_class = pd.DataFrame()
        for symptom in symptom_cols:
            positive_yes = df[df['class'] == 'Positive'][symptom].value_counts().get('Yes', 0)
            positive_total = len(df[df['class'] == 'Positive'])
            positive_pct = positive_yes / positive_total * 100
            
            negative_yes = df[df['class'] == 'Negative'][symptom].value_counts().get('Yes', 0)
            negative_total = len(df[df['class'] == 'Negative'])
            negative_pct = negative_yes / negative_total * 100
            
            temp_df = pd.DataFrame({
                'Symptom': [symptom, symptom],
                'Class': ['Positive', 'Negative'],
                'Percentage': [positive_pct, negative_pct]
            })
            symptoms_by_class = pd.concat([symptoms_by_class, temp_df])
        
        fig = px.bar(symptoms_by_class, x='Symptom', y='Percentage', color='Class', barmode='group',
                     title='Percentage of "Yes" Responses by Symptom and Diabetes Status',
                     color_discrete_map={"Positive": "#1E90FF", "Negative": "#D3D3D3"})
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig)
    
    # Correlation heatmap
    if st.checkbox("Feature correlation"):
        # Convert Yes/No to 1/0 for correlation analysis
        df_encoded = df.copy()
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'gender':  # Skip gender for now
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
        
        # One-hot encode gender
        df_encoded = pd.get_dummies(df_encoded, columns=['gender'], drop_first=True)
        
        # Map class to 1/0
        df_encoded['class'] = df_encoded['class'].map({'Positive': 1, 'Negative': 0})
        
        # Calculate correlation matrix
        corr = df_encoded.corr()
        
        # Plot heatmap
        fig = px.imshow(corr, text_auto=True, aspect="auto", 
                       title="Feature Correlation Heatmap",
                       color_continuous_scale='Blues')
        st.plotly_chart(fig)
    
    # Feature importance by chi-square test
    if st.checkbox("Feature importance"):
        # Prepare data for chi-square test
        X = df.drop('class', axis=1)
        y = df['class']
        
        # Encode categorical variables
        X_encoded = pd.get_dummies(X)
        y_encoded = LabelEncoder().fit_transform(y)
        
        # Select features with chi-square test
        selector = SelectKBest(chi2, k='all')
        selector.fit(X_encoded, y_encoded)
        feature_scores = pd.DataFrame({
            'Feature': X_encoded.columns,
            'Score': selector.scores_
        })
        
        # Sort by importance
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        
        # Plot importance
        fig = px.bar(feature_scores, x='Feature', y='Score', 
                    title='Feature Importance (Chi-Square Test)',
                    color='Score', color_continuous_scale='Blues')
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig)

# Function for model building
def build_model(df):
    st.markdown("<h2 class='sub-header'>🔍 Model Building</h2>", unsafe_allow_html=True)
    
    # Data preprocessing
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Convert categorical variables to numerical
    X_encoded = pd.get_dummies(X)
    y_encoded = LabelEncoder().fit_transform(y)  # Convert Positive/Negative to 1/0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.25, random_state=42)
    
    # Select model
    model_option = st.selectbox(
        "Select a model to train:",
        ("Random Forest", "Logistic Regression", "Gradient Boosting")
    )
    
    if model_option == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    elif model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
    else:  # Gradient Boosting
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    
    # Train model with hyperparameter tuning
    with st.spinner("Training model... Please wait."):
        cv = st.slider("Number of cross-validation folds:", 3, 10, 5)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        st.write(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Hyperparameter tuning
        if st.checkbox("Perform hyperparameter tuning (may take time)"):
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            st.write(f"Best parameters: {grid_search.best_params_}")
            st.write(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
            
            # Use the best model
            model = grid_search.best_estimator_
        else:
            # Train with default parameters
            model.fit(X_train, y_train)
    
    # Evaluate model
    st.markdown("<h3>Model Evaluation</h3>", unsafe_allow_html=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Test Accuracy", f"{accuracy:.4f}")
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("Classification Report:")
        st.dataframe(report_df.style.format('{:.4f}'))
    
    with col2:
        # Confusion Matrix
        fig = px.imshow(cm, 
                      x=['Predicted Negative', 'Predicted Positive'],
                      y=['Actual Negative', 'Actual Positive'],
                      text_auto=True,
                      title="Confusion Matrix",
                      color_continuous_scale='Blues')
        st.plotly_chart(fig)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc:.4f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.7, y=0.05),
        width=700,
        height=500
    )
    st.plotly_chart(fig)
    
    return model, X_encoded.columns

# Function for prediction
def make_prediction(model, feature_names):
    st.markdown("<h2 class='sub-header'>🔮 Diabetes Prediction</h2>", unsafe_allow_html=True)
    st.write("Enter the patient's information to get a prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 0, 100, 45)
        gender = st.radio("Gender", ["Male", "Female"])
        polyuria = st.radio("Polyuria (Excessive urination)", ["Yes", "No"])
        polydipsia = st.radio("Polydipsia (Excessive thirst)", ["Yes", "No"])
        sudden_weight_loss = st.radio("Sudden weight loss", ["Yes", "No"])
        weakness = st.radio("Weakness", ["Yes", "No"])
        polyphagia = st.radio("Polyphagia (Excessive hunger)", ["Yes", "No"])
        genital_thrush = st.radio("Genital thrush", ["Yes", "No"])
    
    with col2:
        visual_blurring = st.radio("Visual blurring", ["Yes", "No"])
        itching = st.radio("Itching", ["Yes", "No"])
        irritability = st.radio("Irritability", ["Yes", "No"])
        delayed_healing = st.radio("Delayed healing", ["Yes", "No"])
        partial_paresis = st.radio("Partial paresis", ["Yes", "No"])
        muscle_stiffness = st.radio("Muscle stiffness", ["Yes", "No"])
        alopecia = st.radio("Alopecia (Hair loss)", ["Yes", "No"])
        obesity = st.radio("Obesity", ["Yes", "No"])
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'polyuria': [polyuria],
        'polydipsia': [polydipsia],
        'sudden_weight_loss': [sudden_weight_loss],
        'weakness': [weakness],
        'polyphagia': [polyphagia],
        'genital_thrush': [genital_thrush],
        'visual_blurring': [visual_blurring],
        'itching': [itching],
        'irritability': [irritability],
        'delayed_healing': [delayed_healing],
        'partial_paresis': [partial_paresis],
        'muscle_stiffness': [muscle_stiffness],
        'alopecia': [alopecia],
        'obesity': [obesity]
    })
    
    # One-hot encode the input data
    input_encoded = pd.get_dummies(input_data)
    
    # Ensure input_encoded has the same columns as training data
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[feature_names]
    
    # Make prediction
    if st.button("Predict"):
        with st.spinner("Calculating prediction..."):
            # Get prediction probability
            prediction_proba = model.predict_proba(input_encoded)[0][1]
            prediction = "Positive" if prediction_proba > 0.5 else "Negative"
            
            # Display result
            st.markdown("<div style='background-color:#F0F8FF; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
            st.subheader("Prediction Result:")
            
            if prediction == "Positive":
                st.markdown(f"<h3 style='color:#FF5733'>Diabetes Risk: {prediction}</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:#33A1FF'>Diabetes Risk: {prediction}</h3>", unsafe_allow_html=True)
            
            # Display probability
            st.write(f"Probability of diabetes: {prediction_proba:.2%}")
            
            # Risk level
            if prediction_proba < 0.2:
                risk_level = "Low Risk"
                color = "#33FF57"
            elif prediction_proba < 0.5:
                risk_level = "Moderate Risk"
                color = "#FFD733"
            elif prediction_proba < 0.8:
                risk_level = "High Risk"
                color = "#FF8C33"
            else:
                risk_level = "Very High Risk"
                color = "#FF5733"
            
            st.markdown(f"<h4 style='color:{color}'>Risk Level: {risk_level}</h4>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("Recommendations:")
            if prediction == "Positive":
                st.write("- Consult with a healthcare provider for a comprehensive diabetes assessment")
                st.write("- Consider getting an HbA1c test and fasting blood glucose test")
                st.write("- Monitor blood sugar levels regularly")
                st.write("- Maintain a healthy diet and exercise routine")
            else:
                st.write("- Continue maintaining a healthy lifestyle")
                st.write("- Regular check-ups with healthcare provider")
                st.write("- Be aware of diabetes symptoms and risk factors")
            
            # Disclaimer
            st.info("Disclaimer: This prediction is based on a machine learning model and should not be considered as medical advice. Always consult with a healthcare professional for proper diagnosis and treatment.")

# Information section
def show_info():
    st.markdown("<h2 class='sub-header'>ℹ️ About Diabetes</h2>", unsafe_allow_html=True)
    
    tabs = st.tabs(["What is Diabetes?", "Risk Factors", "Symptoms", "Prevention"])
    
    with tabs[0]:
        st.write("""
        **Diabetes** is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood sugar.
        
        There are two main types of diabetes:
        - **Type 1 Diabetes**: The body does not produce insulin. People with type 1 diabetes need daily insulin injections to control their blood glucose levels.
        - **Type 2 Diabetes**: The body does not use insulin effectively. This is the most common type of diabetes and is largely the result of excess body weight and physical inactivity.
        """)
    
    with tabs[1]:
        st.write("""
        **Risk factors for Type 2 Diabetes include:**
        - Family history of diabetes
        - Overweight or obesity
        - Physical inactivity
        - Age (risk increases with age)
        - High blood pressure
        - History of gestational diabetes
        - Polycystic ovary syndrome
        - History of heart disease or stroke
        - Certain ethnicities (including African American, Latino, Native American, and Asian American)
        """)
    
    with tabs[2]:
        st.write("""
        **Common symptoms of diabetes include:**
        - Polyuria (frequent urination)
        - Polydipsia (excessive thirst)
        - Polyphagia (excessive hunger)
        - Unexpected weight loss
        - Fatigue
        - Blurred vision
        - Slow-healing wounds
        - Frequent infections
        - Tingling or numbness in hands or feet
        
        Note that many people with Type 2 diabetes may not experience symptoms for years.
        """)
    
    with tabs[3]:
        st.write("""
        **Prevention strategies for Type 2 Diabetes:**
        - Maintain a healthy weight
        - Regular physical activity (at least 30 minutes per day)
        - Healthy diet rich in fruits, vegetables, and whole grains
        - Limit sugar and saturated fat intake
        - Don't smoke
        - Limit alcohol consumption
        - Regular health check-ups
        """)

# Main function
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/diabetes.png", width=100)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training", "Make Prediction", "About Diabetes"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This app is for educational purposes only and should not be used as a substitute for professional medical advice.
    """)
    
    # Load data
    df = load_data()
    
    # Home page
    if page == "Home":
        st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=150)
        st.markdown("<h1 class='main-header'>Welcome to the Diabetes Prediction App</h1>", unsafe_allow_html=True)
        
        st.write("""
        This application uses machine learning to predict the likelihood of diabetes based on various symptoms and risk factors.
        
        ### Features:
        - **Data Exploration**: Visualize and understand the diabetes symptoms dataset
        - **Model Training**: Train and evaluate machine learning models for diabetes prediction
        - **Make Prediction**: Input patient information to get a diabetes risk assessment
        - **About Diabetes**: Learn about diabetes, its symptoms, risk factors, and prevention strategies
        
        ### How to use:
        1. Navigate through the app using the sidebar
        2. Explore the data to understand diabetes risk factors
        3. Train a model to see how accurately we can predict diabetes
        4. Input patient information to get a prediction
        
        ### Dataset:
        The application uses a dataset containing information about various diabetes symptoms and their correlation with diabetes diagnosis.
        """)
        
        st.markdown("---")
        st.markdown("<h3>Quick Statistics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Diabetes Positive", len(df[df['class'] == 'Positive']))
        with col3:
            st.metric("Diabetes Negative", len(df[df['class'] == 'Negative']))
    
    # Data Exploration page
    elif page == "Data Exploration":
        explore_data(df)
    
    # Model Training page
    elif page == "Model Training":
        model, feature_names = build_model(df)
        # Save the model and feature names in session state
        st.session_state['model'] = model
        st.session_state['feature_names'] = feature_names
    
    # Make Prediction page
    elif page == "Make Prediction":
        if 'model' not in st.session_state:
            st.warning("Please train a model first on the 'Model Training' page.")
            if st.button("Go to Model Training"):
                st.session_state['page'] = "Model Training"
                st.experimental_rerun()
        else:
            make_prediction(st.session_state['model'], st.session_state['feature_names'])
    
    # About Diabetes page
    # About Diabetes page
    elif page == "About Diabetes":
        show_info()
        
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        color: #2E86C1;
        font-size: 42px;
    }
    .sub-header {
        color: #3498DB;
    }
    </style>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()