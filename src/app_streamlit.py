"""Interactive Streamlit app for spam classification with advanced visualizations.

Run with:
    streamlit run src\app_streamlit.py
"""
import os
import time
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

from data import load_sms_dataset, normalize_labels
from models import (
    load_pipeline, train_model, plot_evaluation, save_pipeline,
    evaluate_model, train_multiple_models
)


MODEL_PATH = os.environ.get("SPAM_MODEL_PATH", "models/spam_classifier.joblib")


@st.cache_resource
def load_model(path: str):
    """Load cached model pipeline."""
    if not os.path.exists(path):
        return None
    return load_pipeline(path)


def plot_token_distribution(texts, labels, top_n=20):
    """Plot token frequency distribution by class using Plotly for interactive visualization."""
    def get_token_counts(text_list):
        tokens = ' '.join(text_list).lower().split()
        return Counter(tokens)
    
    # Split by class
    ham_texts = [text for text, label in zip(texts, labels) if label == 0]
    spam_texts = [text for text, label in zip(texts, labels) if label == 1]
    
    # Get counts
    ham_counts = get_token_counts(ham_texts)
    spam_counts = get_token_counts(spam_texts)
    
    # Create DataFrames
    ham_df = pd.DataFrame(ham_counts.most_common(top_n), columns=['token', 'frequency'])
    spam_df = pd.DataFrame(spam_counts.most_common(top_n), columns=['token', 'frequency'])
    
    # Create subplots using plotly
    fig = go.Figure()
    
    # Ham tokens
    fig.add_trace(go.Bar(
        name='Ham',
        x=ham_df['frequency'],
        y=ham_df['token'],
        orientation='h',
        marker_color='rgba(55, 128, 191, 0.7)',
        text=ham_df['frequency'],
        textposition='auto',
    ))
    
    # Spam tokens
    fig.add_trace(go.Bar(
        name='Spam',
        x=spam_df['frequency'],
        y=spam_df['token'],
        orientation='h',
        marker_color='rgba(219, 64, 82, 0.7)',
        text=spam_df['frequency'],
        textposition='auto',
    ))
    
    # Update layout
    fig.update_layout(
        title='Token Frequency by Class',
        barmode='group',
        xaxis_title='Frequency',
        yaxis_title='Token',
        template='plotly_dark',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def plot_threshold_metrics(y_true, y_prob):
    """Create interactive plots for model performance metrics."""
    # Calculate metrics
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create metrics table
    thresholds_range = np.linspace(0.3, 0.75, 10)
    metrics = []
    for threshold in thresholds_range:
        y_pred = (y_prob >= threshold).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics.append({
            'threshold': threshold,  # Store as float instead of string
            'accuracy': float(accuracy),
            'precision': float(precision_val),
            'recall': float(recall_val),
            'f1': float(f1)
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create ROC curve
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        mode='lines',
        line=dict(color='cyan', width=2)
    ))
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random',
        mode='lines',
        line=dict(color='gray', width=2, dash='dash')
    ))
    roc_fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark',
        height=500,
        showlegend=True
    )
    
    # Create metrics visualization
    metrics_fig = go.Figure()
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        metrics_fig.add_trace(go.Scatter(
            x=metrics_df['threshold'],
            y=metrics_df[metric],
            name=metric.capitalize(),
            mode='lines+markers'
        ))
    metrics_fig.update_layout(
        title='Performance Metrics vs Threshold',
        xaxis_title='Threshold',
        yaxis_title='Score',
        template='plotly_dark',
        height=500,
        showlegend=True
    )
    
    return metrics_df, roc_fig, metrics_fig

def apply_custom_css():
    """Apply custom CSS styling to make the interface more modern and beautiful."""
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to bottom right, #1a1a2e, #16213e);
        }
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 20px;
            padding: 0.5rem 2rem;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        h1, h2, h3 {
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
            margin-bottom: 1rem;
        }
        .stAlert {
            border-radius: 10px;
            padding: 1rem;
        }
        .sidebar .sidebar-content {
            background-color: rgba(26, 26, 46, 0.9);
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)

def show_loading_progress(message="Processing..."):
    """Show a progress bar with a custom message."""
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    progress_bar.empty()

def main():
    st.set_page_config(
        page_title="Spam/Ham Classifier - Phase 4 Visualizations",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    
    # Header with modern design
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3rem; margin-bottom: 1rem;'>
                🔍 Spam/Ham Classifier
            </h1>
            <p style='font-size: 1.2rem; color: #a0a0a0;'>
                Advanced visualization and analysis dashboard for spam detection
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar inputs
    st.sidebar.title("Inputs")
    
    # Dataset selection
    st.sidebar.subheader("Dataset CSV")
    dataset_path = st.sidebar.selectbox(
        "Dataset path",
        ["sms_spam_no_header.csv"],
        key="dataset_path"
    )
    
    # Column selections
    st.sidebar.subheader("Label column")
    label_col = st.sidebar.selectbox("", ["label", "col_0"], key="label_col")
    
    st.sidebar.subheader("Text column")
    text_columns = [
        "message", "col_1",
        "text_clean", "text_lower", "text_contacts_masked",
        "text_numbers", "text_stripped", "text_whitespace"
    ]
    text_col = st.sidebar.selectbox("", text_columns, key="text_col")
    
    # Model directory
    st.sidebar.subheader("Models dir")
    model_dir = st.sidebar.text_input("Models directory", "models", key="model_dir", label_visibility="collapsed")
    
    # Model selection dropdown
    st.sidebar.subheader("Model Selection")
    available_models = ["SVM", "Random Forest", "Gradient Boosting", "Naive Bayes"]
    selected_model = st.sidebar.selectbox(
        "Select model for analysis",
        available_models,
        key="selected_model",
        help="Choose which model to analyze and use for predictions"
    )
    
    # Test size slider
    test_size = st.sidebar.slider("Test size", 0.1, 0.3, 0.2, 0.01)
    
    # Random seed
    seed = st.sidebar.number_input("Seed", value=42)
    
    # Decision threshold
    threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)

    # Load data and model
    with st.spinner("Loading data and model..."):
        try:
            # Make sure the file exists
            if not os.path.exists(dataset_path):
                st.error(
                    "⚠️ Dataset not found! Please check the following:\n"
                    "1. The file exists in the correct location\n"
                    "2. The file name is correct\n"
                    f"3. The path '{dataset_path}' is accessible"
                )
                return
            
            # Load and process data
            with st.spinner("Loading dataset..."):
                show_loading_progress("Reading data...")
                # Load and process data
                df = load_sms_dataset(dataset_path)  # This sets column names to 'label' and 'message'
                df = normalize_labels(df)  # This adds 'label_num' column
                
                st.success("✅ Dataset loaded successfully!")
                st.markdown(f"""
                    **Dataset Summary:**
                    - Total messages: {len(df):,}
                    - Spam messages: {df['label_num'].sum():,} ({df['label_num'].mean()*100:.1f}%)
                    - Ham messages: {(1-df['label_num']).sum():,} ({(1-df['label_num']).mean()*100:.1f}%)
                    
                    **Columns:**
                    - Label column: 'label' (text labels: spam/ham)
                    - Text column: 'message' (SMS content)
                """)
            
            # Load or train models (once per session)
            if "all_models" not in st.session_state:
                with st.spinner("Training all models (once per session)..."):
                    show_loading_progress("Training in progress...")
                    
                    # Train multiple models
                    results = train_multiple_models(
                        df["message"].tolist(),
                        df["label_num"].tolist(),
                        cross_validate=True
                    )
                    
                    # Find the best model
                    best_model_name = max(
                        results['models'].keys(),
                        key=lambda x: results['models'][x]['metrics']['roc']['auc']
                    )
                    
                    st.success(f"✅ Models trained successfully! Best model: {best_model_name}")
                    
                    # Store all models and results in session state
                    st.session_state.all_models = results['models']
                    st.session_state.vectorizer = results['vectorizer']
                    
                    # Set the default model to the best one
                    st.session_state.model = results['models'][best_model_name]['model']
                    
                    # Save the best model to disk
                    save_pipeline(results['vectorizer'], st.session_state.model, MODEL_PATH)
                    st.success(f"✅ Best model ({best_model_name}) saved to disk.")
                
        except Exception as e:
            st.error("🚫 An error occurred!")
            with st.expander("Error Details"):
                st.error(f"Error Type: {type(e).__name__}")
                st.error(f"Error Message: {str(e)}")
                st.error("""
                    Please check:
                    1. The dataset file format is correct (CSV)
                    2. The file has the expected columns
                    3. You have sufficient permissions
                    4. The Python environment is properly configured
                """)
            return

    # Data Overview with tabs
    st.header("📊 Data Analysis & Model Performance")
    
    tabs = st.tabs(["📈 Data Overview", "🔤 Token Analysis", "🎯 Model Performance", "🤖 Live Testing"])
    
    with tabs[0]:
        st.subheader("Class Distribution")
        
        # Create a more attractive distribution plot using plotly
        dist_fig = px.pie(
            df, 
            names="label",
            title="Message Distribution",
            color="label",
            color_discrete_map={"ham": "rgb(55, 128, 191)", "spam": "rgb(219, 64, 82)"},
            hole=0.4
        )
        dist_fig.update_layout(template="plotly_dark")
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # Add some statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", f"{len(df):,}", "100%")
        with col2:
            st.metric("Spam Messages", f"{df['label_num'].sum():,}", 
                     f"{df['label_num'].mean()*100:.1f}%")
        with col3:
            st.metric("Ham Messages", f"{(1-df['label_num']).sum():,}", 
                     f"{(1-df['label_num']).mean()*100:.1f}%")
    
    with tabs[1]:
        st.subheader("Token Analysis")
        n_tokens = st.slider("Number of top tokens to show", 10, 50, 20)
        token_fig = plot_token_distribution(df["message"].tolist(), df["label_num"].tolist(), top_n=n_tokens)
        st.plotly_chart(token_fig, use_container_width=True)
        
        with st.expander("📝 Token Analysis Explanation"):
            st.markdown("""
                This visualization shows the most frequent words (tokens) in spam and ham messages.
                - Longer bars indicate more frequent usage
                - Compare patterns between spam and ham to understand key differences
                - Common words like 'to', 'the', etc. are expected in both classes
            """)
    
    with tabs[2]:
        st.subheader("Model Performance Metrics")

        if "model" in st.session_state:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            # Always use the model selected in the sidebar
            if "all_models" in st.session_state and selected_model in st.session_state.all_models:
                st.session_state.model = st.session_state.all_models[selected_model]['model']

            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                df["message"], df["label_num"],
                test_size=test_size,
                random_state=seed,
                stratify=df["label_num"]
            )

            # Calculate and display key metrics based on the selected threshold
            st.markdown(f"### {selected_model} Performance (at 0.5 Threshold)")

            metrics = st.session_state.all_models[selected_model]['metrics']['classification_report']
            acc = metrics['accuracy']
            precision = metrics['1']['precision']
            recall = metrics['1']['recall']
            f1 = metrics['1']['f1-score']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{acc:.4f}")
            with col2:
                st.metric("Precision", f"{precision:.4f}")
            with col3:
                st.metric("Recall", f"{recall:.4f}")
            with col4:
                st.metric("F1 Score", f"{f1:.4f}")

            # Add model comparison section if multiple models are available
            if "all_models" in st.session_state:
                st.markdown("### 📊 Model Comparison")

                # Create comparison dataframe
                model_metrics = []
                for model_name, model_info in st.session_state.all_models.items():
                    metrics = model_info['metrics']
                    model_metrics.append({
                        'Model': model_name,
                        'ROC AUC': metrics['roc']['auc'],
                        'PR AUC': metrics['pr']['auc'],
                        'Accuracy': metrics['classification_report']['accuracy'],
                        'Precision': metrics['classification_report']['1']['precision'],
                        'Recall': metrics['classification_report']['1']['recall'],
                        'F1-Score': metrics['classification_report']['1']['f1-score']
                    })

                comparison_df = pd.DataFrame(model_metrics)

                # Display comparison table
                st.dataframe(
                    comparison_df.style.background_gradient(cmap='viridis')
                        .format({col: '{:.4f}' for col in comparison_df.columns if col != 'Model'})
                )

                # Create ROC curve comparison
                roc_fig_comp = go.Figure()
                for model_name, model_info in st.session_state.all_models.items():
                    metrics = model_info['metrics']
                    roc_fig_comp.add_trace(go.Scatter(
                        x=metrics['roc']['fpr'],
                        y=metrics['roc']['tpr'],
                        name=f"{model_name} (AUC={metrics['roc']['auc']:.3f})",
                        mode='lines'
                    ))

                roc_fig_comp.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='Random',
                    mode='lines',
                    line=dict(color='gray', width=2, dash='dash')
                ))

                roc_fig_comp.update_layout(
                    title='ROC Curves Comparison',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    template='plotly_dark',
                    height=500,
                    showlegend=True
                )

                st.plotly_chart(roc_fig_comp, use_container_width=True)

                with st.expander("📖 Understanding Model Comparison"):
                    st.markdown("""
                        **Model Comparison Metrics:**
                        - **ROC AUC**: Overall model discrimination ability (higher is better)
                        - **PR AUC**: Performance on imbalanced datasets (higher is better)
                        - **Accuracy**: Overall prediction accuracy
                        - **Precision**: Accuracy of spam predictions
                        - **Recall**: Ability to find all spam messages
                        - **F1-Score**: Harmonic mean of precision and recall

                        The ROC curves show each model's performance across different classification thresholds.
                        Models with curves closer to the top-left corner perform better.
                    """)

            # Transform test data to get probabilities for plots
            X_test_vec = st.session_state.vectorizer.transform(X_test)
            y_prob = st.session_state.model.predict_proba(X_test_vec)[:, 1]

            # Get metrics and plots for different thresholds
            metrics_df, roc_fig, metrics_fig = plot_threshold_metrics(y_test, y_prob)

            # Display metrics in an attractive way
            st.markdown("### 📊 Performance at Different Thresholds")
            styled_df = metrics_df.style.background_gradient(cmap='viridis')
            styled_df.format({
                'threshold': lambda x: '{:.2f}'.format(x),
                'precision': lambda x: '{:.4f}'.format(x),
                'recall': lambda x: '{:.4f}'.format(x),
                'f1': lambda x: '{:.4f}'.format(x),
                'accuracy': lambda x: '{:.4f}'.format(x)
            })
            st.dataframe(styled_df)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(roc_fig, use_container_width=True)
            with col2:
                st.plotly_chart(metrics_fig, use_container_width=True)

            with st.expander("📖 Understanding the Metrics"):
                st.markdown("""
                    **Metrics Explanation:**
                    - **Accuracy**: Overall correctness of predictions
                    - **Precision**: Accuracy of spam predictions
                    - **Recall**: Ability to find all spam messages
                    - **F1 Score**: Balance between precision and recall

                    **ROC Curve:**
                    - Shows tradeoff between true and false positive rates
                    - Closer to top-left corner is better
                    - AUC (Area Under Curve) of 1.0 is perfect

                    **Threshold Selection:**
                    - Higher threshold = fewer spam predictions but more confident
                    - Lower threshold = more spam predictions but less confident
                    - Choose based on your tolerance for false positives
                """)
    
    with tabs[3]:
        st.subheader("🔍 Live Message Classification")
        
        # Model selection for classification
        if "all_models" in st.session_state:
            model_names = list(st.session_state.all_models.keys())
            selected_model_classify = st.selectbox(
                "Select model for classification",
                model_names,
                key="selected_model_classify"
            )
            active_model = st.session_state.all_models[selected_model_classify]['model']
        else:
            active_model = st.session_state.model
        
        # Example messages section
        st.markdown("### Try an Example")
        example_col1, example_col2 = st.columns(2)
        with example_col1:
            if st.button("📨 Load Spam Example", use_container_width=True):
                text = "URGENT! You've won a 1 week FREE membership in our £100,000 Prize Jackpot! Text CLAIM to 81010 T&C www.dbuk.net"
                st.session_state.message = text
        with example_col2:
            if st.button("💌 Load Ham Example", use_container_width=True):
                text = "Hey, can we meet tomorrow at 3pm for coffee? Let me know if that works for you!"
                st.session_state.message = text
        
        # Message input section
        st.markdown("### 📝 Enter Your Message")
        message = st.text_area(
            "",
            value=st.session_state.get("message", ""),
            height=100,
            placeholder="Type or paste a message here to classify..."
        )
        
        # Prediction section
        if st.button("🔍 Classify Message", type="primary", use_container_width=True) and message and "model" in st.session_state:
            with st.spinner("Analyzing message..."):
                # Transform input
                X = st.session_state.vectorizer.transform([message])
                proba = active_model.predict_proba(X)[0]
                prediction = "SPAM" if proba[1] >= threshold else "HAM"
                
                # Display prediction with custom styling
                prediction_color = "red" if prediction == "SPAM" else "green"
                st.markdown(f"""
                    <div style='background-color: rgba(0,0,0,0.2); padding: 20px; border-radius: 10px; margin: 20px 0;'>
                        <h3 style='color: {prediction_color}; text-align: center; margin: 0;'>
                            {prediction} {'🚫' if prediction == 'SPAM' else '✅'}
                        </h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence scores
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Spam Probability", f"{proba[1]:.3f}", 
                             f"{'+' if proba[1] > 0.5 else ''}{(proba[1]-0.5)*200:.1f}%")
                with col2:
                    st.metric("Ham Probability", f"{proba[0]:.3f}", 
                             f"{'+' if proba[0] > 0.5 else ''}{(proba[0]-0.5)*200:.1f}%")
                



if __name__ == "__main__":
    main()
