# Smart Governance Analytics - Machine Learning for East African Policy Making

A comprehensive machine learning and big data analytics solution designed to empower smart governance and evidence-based policy making in East Africa. This platform provides tools for resource allocation prediction, data analysis, and policy insights using advanced ML techniques.

## 🌟 Features

- **Predictive Analytics**: Resource allocation prediction based on regional characteristics
- **Interactive Dashboard**: Web-based interface for policymakers using Streamlit
- **Extensible Framework**: Easy to add new models and data sources
- **Comprehensive Analysis**: Full ETL pipeline with data validation and preprocessing
- **Policy Insights**: Actionable recommendations based on ML predictions
- **Jupyter Integration**: Complete notebook for exploratory data analysis

## 🏗️ Project Structure

```
Machine_learning2025/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
├── data/
│   └── sample_data.csv               # Sample governance dataset
├── src/
│   ├── config.py                     # Configuration settings
│   ├── etl.py                        # Data pipeline (Extract, Transform, Load)
│   ├── model.py                      # ML model training and prediction
│   └── dashboard.py                  # Streamlit web dashboard
├── notebooks/
│   └── EDA_and_Model.ipynb          # Exploratory data analysis notebook
└── models/                           # Trained models (auto-created)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jean1991/Machine_learning2025.git
   cd Machine_learning2025
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run src/dashboard.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The dashboard will automatically train a model if none exists

## 📊 Dashboard Features

### 1. Data Overview
- **Dataset exploration**: View and filter governance data
- **Regional analysis**: Compare metrics across regions
- **Statistical summaries**: Key indicators and distributions
- **Data export**: Download filtered datasets

### 2. Resource Prediction
- **Interactive prediction**: Input district characteristics
- **Real-time results**: Instant resource allocation predictions
- **Comparative analysis**: Compare with regional averages
- **Policy recommendations**: Actionable insights based on predictions

### 3. Analytics & Insights
- **Correlation analysis**: Understand relationships between variables
- **Regional comparisons**: Identify development patterns
- **Feature importance**: See which factors drive resource needs
- **Trend visualization**: Interactive charts and graphs

### 4. Data Upload
- **Custom datasets**: Upload your own governance data
- **Automatic processing**: Data validation and transformation
- **Model retraining**: Update models with new data
- **Format validation**: Ensure data quality and consistency

## 🔧 Usage Examples

### Command Line Model Training

```python
from src.model import train_and_evaluate_model

# Train and evaluate the model
model = train_and_evaluate_model()

# Make a prediction
prediction = model.predict_single(
    region="East",
    district="Kampala", 
    population=1500000,
    income_per_capita=850,
    health_index=0.75,
    education_index=0.82
)

print(f"Predicted resource allocation: ${prediction:,.0f}")
```

### ETL Pipeline Usage

```python
from src.etl import DataETL

# Initialize ETL pipeline
etl = DataETL()

# Run complete pipeline
X, y = etl.run_pipeline("path/to/your/data.csv")

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

### Jupyter Notebook Analysis

Launch the exploration notebook:
```bash
jupyter notebook notebooks/EDA_and_Model.ipynb
```

## 📈 Data Format

The system expects CSV files with the following columns:

| Column | Description | Type | Range |
|--------|-------------|------|-------|
| `region` | Geographic region | String | East, North, West, Central |
| `district` | District name | String | - |
| `population` | Population count | Integer | > 0 |
| `income_per_capita` | Income per capita (USD) | Float | > 0 |
| `health_index` | Health development index | Float | 0-1 |
| `education_index` | Education development index | Float | 0-1 |
| `current_allocation` | Current resource allocation (USD) | Integer | > 0 |
| `year` | Year of data | Integer | - |

### Sample Data

The project includes sample data from 25 East African districts with:
- Population ranges from 140,000 to 1.5M
- Income per capita from $280 to $850
- Health and education indices from 0.42 to 0.82
- Resource allocations from $3.5M to $25M

## 🧠 Machine Learning Models

### Current Models

1. **Random Forest Regressor**
   - Primary model for resource allocation prediction
   - Handles non-linear relationships
   - Provides feature importance insights
   - Robust to outliers

2. **Linear Regression**
   - Baseline model for comparison
   - Interpretable coefficients
   - Fast training and prediction

### Model Features

The system uses engineered features including:
- **Population density**: Population per unit area
- **Socioeconomic index**: Weighted combination of health and education
- **Income categories**: Categorical income brackets
- **Regional encodings**: Numerical representation of regions

## 🔄 Extensibility

### Adding New Models

1. **Fraud Detection**
   ```python
   from sklearn.ensemble import IsolationForest
   
   # Add to model.py
   class FraudDetectionModel:
       def __init__(self):
           self.model = IsolationForest(contamination=0.1)
   ```

2. **Sentiment Analysis**
   ```python
   from textblob import TextBlob
   
   # Add social media sentiment analysis
   def analyze_sentiment(text):
       return TextBlob(text).sentiment.polarity
   ```

### Adding New Data Sources

1. **API Integration**
   ```python
   import requests
   
   def fetch_api_data(endpoint):
       response = requests.get(endpoint)
       return response.json()
   ```

2. **Social Media Data**
   ```python
   import tweepy
   
   # Twitter API integration
   def fetch_twitter_data(query):
       # Implementation for Twitter data
       pass
   ```

## 📋 Configuration

Key configuration options in `src/config.py`:

```python
# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,           # Train/test split
    "random_state": 42,         # Reproducibility
    "model_type": "random_forest",
    "n_estimators": 100,        # Number of trees
    "max_depth": 10             # Maximum tree depth
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "title": "Smart Governance Analytics Dashboard",
    "page_icon": "🏛️",
    "layout": "wide"
}
```

## 🧪 Testing and Validation

### Running Tests

```bash
# Test ETL pipeline
python -c "from src.etl import DataETL; etl = DataETL(); X, y = etl.run_pipeline(); print('ETL test passed')"

# Test model training
python -c "from src.model import train_and_evaluate_model; model = train_and_evaluate_model(); print('Model test passed')"

# Test dashboard (manual)
streamlit run src/dashboard.py
```

### Model Validation

The system includes comprehensive model validation:
- **Cross-validation**: K-fold validation for robust performance estimates
- **Metrics tracking**: R², RMSE, MAE for regression performance
- **Residual analysis**: Check for model assumptions
- **Feature importance**: Understand model decision factors

## 🚀 Deployment

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run src/dashboard.py --server.port 8501
```

### Cloud Deployment Options

1. **Streamlit Cloud**
   - Connect GitHub repository
   - Automatic deployment from main branch
   - Free hosting for public repositories

2. **Heroku**
   ```bash
   # Create Procfile
   echo "web: streamlit run src/dashboard.py --server.port \$PORT" > Procfile
   
   # Deploy
   git push heroku main
   ```

3. **Docker**
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   EXPOSE 8501
   CMD ["streamlit", "run", "src/dashboard.py"]
   ```

## 🔮 Future Enhancements

### Planned Features

1. **Advanced Models**
   - Deep learning models (Neural Networks, LSTM)
   - Ensemble methods
   - Time series forecasting
   - Causal inference models

2. **Enhanced Data Sources**
   - Real-time API integrations
   - Satellite imagery analysis
   - Social media sentiment
   - Economic indicators

3. **Advanced Analytics**
   - Multi-objective optimization
   - Scenario simulation
   - Policy impact analysis
   - Automated reporting

4. **User Experience**
   - Mobile-responsive design
   - User authentication
   - Custom dashboards
   - Export capabilities

### Research Applications

- **Academic Research**: Policy effectiveness studies
- **Government**: Evidence-based decision making
- **NGOs**: Resource allocation optimization
- **International Organizations**: Development planning

## 📞 Support and Contributing

### Getting Help

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join community discussions
- **Documentation**: Check this README and inline code documentation

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use docstrings for functions and classes
- Add type hints where possible
- Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **East African Development Data**: Based on regional development indicators
- **Open Source Libraries**: Built with scikit-learn, Streamlit, pandas, and more
- **Community**: Thanks to contributors and users providing feedback

---

**Built with ❤️ for smart governance in East Africa**

For questions or collaboration opportunities, please reach out through GitHub Issues or Discussions.