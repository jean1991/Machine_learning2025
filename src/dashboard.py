"""
Streamlit Dashboard for Smart Governance Analytics.

This module provides an interactive web interface for policymakers to:
- View and explore governance data
- Make resource allocation predictions
- Visualize trends and insights
- Upload new data for analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import (
    DASHBOARD_CONFIG, 
    SAMPLE_DATA_PATH, 
    VALID_REGIONS, 
    VALID_DISTRICTS,
    MODEL_PATH
)
from etl import DataETL
from model import ResourceAllocationModel, train_and_evaluate_model

# Configure page
st.set_page_config(
    page_title=DASHBOARD_CONFIG["title"],
    page_icon=DASHBOARD_CONFIG["page_icon"],
    layout=DASHBOARD_CONFIG["layout"]
)


@st.cache_data
def load_data():
    """Load and cache the sample data."""
    try:
        return pd.read_csv(SAMPLE_DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def load_model():
    """Load and cache the trained model."""
    try:
        if MODEL_PATH.exists():
            return ResourceAllocationModel.load_model()
        else:
            st.warning("No trained model found. Training a new model...")
            with st.spinner("Training model..."):
                model = train_and_evaluate_model()
            st.success("Model trained successfully!")
            return model
    except Exception as e:
        st.error(f"Error loading/training model: {e}")
        return None


def main():
    """Main dashboard interface."""
    
    # Header
    st.title("🏛️ Smart Governance Analytics Dashboard")
    st.markdown("**AI-Powered Resource Allocation for East African Policy Making**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Data Overview", "🔮 Resource Prediction", "📈 Analytics & Insights", "📤 Data Upload"]
    )
    
    # Load data and model
    data = load_data()
    model = load_model()
    
    if data is None:
        st.error("Unable to load data. Please check the data file.")
        return
    
    # Page routing
    if page == "📊 Data Overview":
        show_data_overview(data)
    elif page == "🔮 Resource Prediction":
        show_prediction_interface(model, data)
    elif page == "📈 Analytics & Insights":
        show_analytics(data, model)
    elif page == "📤 Data Upload":
        show_data_upload()


def show_data_overview(data):
    """Display data overview and exploration interface."""
    
    st.header("📊 Data Overview")
    
    # Data summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Districts", len(data))
    
    with col2:
        st.metric("Total Population", f"{data['population'].sum():,}")
    
    with col3:
        st.metric("Avg Income per Capita", f"${data['income_per_capita'].mean():.0f}")
    
    with col4:
        st.metric("Total Allocation", f"${data['current_allocation'].sum():,}")
    
    # Data filters
    st.subheader("Filter Data")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_regions = st.multiselect(
            "Select Regions:",
            options=VALID_REGIONS,
            default=VALID_REGIONS
        )
    
    with col2:
        income_range = st.slider(
            "Income per Capita Range:",
            min_value=int(data['income_per_capita'].min()),
            max_value=int(data['income_per_capita'].max()),
            value=(int(data['income_per_capita'].min()), int(data['income_per_capita'].max()))
        )
    
    # Filter data
    filtered_data = data[
        (data['region'].isin(selected_regions)) &
        (data['income_per_capita'] >= income_range[0]) &
        (data['income_per_capita'] <= income_range[1])
    ]
    
    # Display filtered data
    st.subheader("Filtered Dataset")
    st.dataframe(filtered_data, use_container_width=True)
    
    # Download filtered data
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_governance_data.csv",
        mime="text/csv"
    )


def show_prediction_interface(model, data):
    """Display resource allocation prediction interface."""
    
    st.header("🔮 Resource Allocation Prediction")
    
    if model is None:
        st.error("Model not available. Please check the model loading.")
        return
    
    st.markdown("Use this tool to predict optimal resource allocation based on district characteristics.")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            region = st.selectbox("Region:", VALID_REGIONS)
            district = st.selectbox("District:", VALID_DISTRICTS)
            population = st.number_input(
                "Population:",
                min_value=10000,
                max_value=2000000,
                value=500000,
                step=10000
            )
        
        with col2:
            income_per_capita = st.number_input(
                "Income per Capita ($):",
                min_value=200,
                max_value=1500,
                value=500,
                step=10
            )
            health_index = st.slider(
                "Health Index:",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.01
            )
            education_index = st.slider(
                "Education Index:",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.01
            )
        
        submitted = st.form_submit_button("🔮 Predict Resource Allocation")
    
    if submitted:
        try:
            with st.spinner("Making prediction..."):
                prediction = model.predict_single(
                    region=region,
                    district=district,
                    population=population,
                    income_per_capita=income_per_capita,
                    health_index=health_index,
                    education_index=education_index
                )
            
            # Display prediction
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Allocation",
                    f"${prediction:,.0f}"
                )
            
            with col2:
                # Calculate per capita allocation
                per_capita = prediction / population
                st.metric(
                    "Per Capita Allocation",
                    f"${per_capita:.2f}"
                )
            
            with col3:
                # Compare with similar districts
                similar_data = data[data['region'] == region]
                avg_allocation = similar_data['current_allocation'].mean()
                difference = ((prediction - avg_allocation) / avg_allocation) * 100
                st.metric(
                    "vs Regional Average",
                    f"{difference:+.1f}%"
                )
            
            # Additional insights
            st.subheader("Allocation Insights")
            
            if prediction > avg_allocation:
                st.info(f"💡 This district requires **{difference:.1f}%** more resources than the regional average, likely due to higher population or lower socioeconomic indicators.")
            else:
                st.info(f"💡 This district requires **{abs(difference):.1f}%** fewer resources than the regional average.")
            
            # Recommendations
            st.subheader("Policy Recommendations")
            
            recommendations = []
            
            if health_index < 0.6:
                recommendations.append("🏥 Prioritize healthcare infrastructure and services")
            
            if education_index < 0.6:
                recommendations.append("🎓 Invest in education facilities and teacher training")
            
            if income_per_capita < 400:
                recommendations.append("💼 Focus on economic development and job creation programs")
            
            if population > 1000000:
                recommendations.append("🏙️ Consider urban planning and infrastructure scaling")
            
            if recommendations:
                for rec in recommendations:
                    st.write(f"• {rec}")
            else:
                st.write("• ✅ District shows balanced development across key indicators")
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")


def show_analytics(data, model):
    """Display analytics and insights dashboard."""
    
    st.header("📈 Analytics & Insights")
    
    # Regional analysis
    st.subheader("Regional Overview")
    
    # Regional metrics
    regional_stats = data.groupby('region').agg({
        'population': 'sum',
        'income_per_capita': 'mean',
        'health_index': 'mean',
        'education_index': 'mean',
        'current_allocation': 'sum'
    }).round(2)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Population by region
        fig = px.bar(
            regional_stats.reset_index(),
            x='region',
            y='population',
            title="Population by Region",
            color='population',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Resource allocation by region
        fig = px.bar(
            regional_stats.reset_index(),
            x='region',
            y='current_allocation',
            title="Resource Allocation by Region",
            color='current_allocation',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Calculate correlation matrix
    numeric_cols = ['population', 'income_per_capita', 'health_index', 'education_index', 'current_allocation']
    corr_matrix = data[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Key Indicators",
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots
    st.subheader("Relationships Between Variables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Income vs Health
        fig = px.scatter(
            data,
            x='income_per_capita',
            y='health_index',
            color='region',
            size='population',
            title="Income vs Health Index",
            hover_data=['district']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Education vs Current Allocation
        fig = px.scatter(
            data,
            x='education_index',
            y='current_allocation',
            color='region',
            size='population',
            title="Education vs Current Allocation",
            hover_data=['district']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model insights
    if model and hasattr(model, 'feature_importance') and model.feature_importance is not None:
        st.subheader("Model Feature Importance")
        
        # Feature importance chart
        fig = px.bar(
            model.feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Most Important Features for Resource Allocation",
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(data.describe(), use_container_width=True)


def show_data_upload():
    """Display data upload and processing interface."""
    
    st.header("📤 Data Upload")
    
    st.markdown("""
    Upload your own governance data to expand the analysis. The uploaded file should be a CSV with the following columns:
    - `region`: Region name
    - `district`: District name  
    - `population`: Population count
    - `income_per_capita`: Income per capita in USD
    - `health_index`: Health index (0-1)
    - `education_index`: Education index (0-1)
    - `current_allocation`: Current resource allocation in USD
    - `year`: Year of data
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            new_data = pd.read_csv(uploaded_file)
            
            st.success("File uploaded successfully!")
            
            # Display preview
            st.subheader("Data Preview")
            st.dataframe(new_data.head(), use_container_width=True)
            
            # Validate data
            st.subheader("Data Validation")
            
            required_columns = [
                'region', 'district', 'population', 'income_per_capita',
                'health_index', 'education_index', 'current_allocation', 'year'
            ]
            
            missing_columns = set(required_columns) - set(new_data.columns)
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                st.success("All required columns present!")
                
                # Process data
                if st.button("Process Uploaded Data"):
                    with st.spinner("Processing data..."):
                        try:
                            etl = DataETL()
                            
                            # Save uploaded data temporarily
                            temp_path = "/tmp/uploaded_data.csv"
                            new_data.to_csv(temp_path, index=False)
                            
                            # Process through ETL
                            processed_data = etl.transform(new_data)
                            
                            st.success("Data processed successfully!")
                            
                            # Display processed data
                            st.subheader("Processed Data")
                            st.dataframe(processed_data.head(), use_container_width=True)
                            
                            # Option to retrain model
                            if st.button("Retrain Model with New Data"):
                                with st.spinner("Retraining model..."):
                                    try:
                                        model = train_and_evaluate_model(temp_path)
                                        st.success("Model retrained successfully!")
                                        st.balloons()
                                    except Exception as e:
                                        st.error(f"Error retraining model: {e}")
                        
                        except Exception as e:
                            st.error(f"Error processing data: {e}")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")


if __name__ == "__main__":
    main()