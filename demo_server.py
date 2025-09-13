#!/usr/bin/env python3
"""
Simple web demo of the Smart Governance Analytics solution.
This demo works without external dependencies for testing purposes.
"""

import csv
import json
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse as urlparse

# Add src to path
sys.path.append('src')
from config import SAMPLE_DATA_PATH

def load_data():
    """Load data using built-in CSV module."""
    with open(SAMPLE_DATA_PATH, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def simple_prediction(region, district, population, income_per_capita, 
                     health_index, education_index):
    """Simple rule-based prediction without ML libraries."""
    # Simple rule-based calculation
    base_allocation = population * 10  # $10 per person base
    
    # Adjust for socioeconomic factors
    socioeconomic_index = (health_index + education_index) / 2
    need_factor = 1.5 - socioeconomic_index  # Higher need for lower indices
    
    # Adjust for income
    if income_per_capita < 400:
        income_factor = 1.3
    elif income_per_capita < 600:
        income_factor = 1.1
    else:
        income_factor = 0.9
    
    # Regional adjustments
    regional_factors = {'North': 1.2, 'East': 1.0, 'West': 1.1, 'Central': 0.9}
    regional_factor = regional_factors.get(region, 1.0)
    
    predicted_allocation = base_allocation * need_factor * income_factor * regional_factor
    return predicted_allocation

class DemoHandler(SimpleHTTPRequestHandler):
    """Custom handler for demo web interface."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.send_dashboard()
        elif self.path == '/api/data':
            self.send_data()
        elif self.path.startswith('/api/predict'):
            self.handle_prediction()
        else:
            self.send_response(404)
            self.end_headers()
    
    def send_dashboard(self):
        """Send the main dashboard HTML."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>🏛️ Smart Governance Analytics Demo</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .form-group { margin: 10px 0; }
        label { display: inline-block; width: 150px; font-weight: bold; }
        input, select { padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 200px; }
        button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #2980b9; }
        .result { background: #e8f5e8; padding: 15px; border-radius: 5px; margin-top: 10px; }
        .data-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .data-table th, .data-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .data-table th { background: #f2f2f2; }
        .metrics { display: flex; justify-content: space-around; margin: 20px 0; }
        .metric { text-align: center; padding: 10px; background: #ecf0f1; border-radius: 5px; }
        .metric h3 { margin: 0; color: #2c3e50; }
        .metric p { margin: 5px 0 0 0; font-size: 18px; font-weight: bold; color: #27ae60; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ Smart Governance Analytics</h1>
            <p>AI-Powered Resource Allocation for East African Policy Making</p>
        </div>
        
        <div class="metrics" id="metrics">
            <div class="metric">
                <h3>Total Districts</h3>
                <p id="total-districts">25</p>
            </div>
            <div class="metric">
                <h3>Total Population</h3>
                <p id="total-population">10.1M</p>
            </div>
            <div class="metric">
                <h3>Avg Income</h3>
                <p id="avg-income">$484</p>
            </div>
            <div class="metric">
                <h3>Total Allocation</h3>
                <p id="total-allocation">$211.2M</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🔮 Resource Allocation Prediction</h2>
            <p>Enter district characteristics to predict optimal resource allocation:</p>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label>Region:</label>
                    <select id="region" required>
                        <option value="">Select Region</option>
                        <option value="East">East</option>
                        <option value="North">North</option>
                        <option value="West">West</option>
                        <option value="Central">Central</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>District:</label>
                    <input type="text" id="district" placeholder="Enter district name" required>
                </div>
                
                <div class="form-group">
                    <label>Population:</label>
                    <input type="number" id="population" placeholder="e.g., 500000" min="10000" max="2000000" required>
                </div>
                
                <div class="form-group">
                    <label>Income per Capita:</label>
                    <input type="number" id="income" placeholder="e.g., 500" min="200" max="1500" step="10" required>
                </div>
                
                <div class="form-group">
                    <label>Health Index (0-1):</label>
                    <input type="number" id="health" placeholder="e.g., 0.6" min="0" max="1" step="0.01" required>
                </div>
                
                <div class="form-group">
                    <label>Education Index (0-1):</label>
                    <input type="number" id="education" placeholder="e.g., 0.6" min="0" max="1" step="0.01" required>
                </div>
                
                <button type="submit">🔮 Predict Resource Allocation</button>
            </form>
            
            <div id="prediction-result" style="display: none;"></div>
        </div>
        
        <div class="section">
            <h2>📊 Sample Data Overview</h2>
            <div id="data-table"></div>
        </div>
        
        <div class="section">
            <h2>ℹ️ About This Demo</h2>
            <p>This is a simplified demo of the Smart Governance Analytics solution. The full solution includes:</p>
            <ul>
                <li><strong>Advanced ML Models:</strong> Random Forest, Linear Regression with feature engineering</li>
                <li><strong>Interactive Dashboard:</strong> Full Streamlit web application with visualizations</li>
                <li><strong>Data Pipeline:</strong> Complete ETL process with validation and preprocessing</li>
                <li><strong>Jupyter Notebooks:</strong> Comprehensive data analysis and model development</li>
                <li><strong>Extensible Framework:</strong> Easy to add new models and data sources</li>
            </ul>
            <p><strong>To run the full solution:</strong></p>
            <ol>
                <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
                <li>Run dashboard: <code>streamlit run src/dashboard.py</code></li>
                <li>Explore notebook: <code>jupyter notebook notebooks/EDA_and_Model.ipynb</code></li>
            </ol>
        </div>
    </div>
    
    <script>
        // Load initial data
        fetch('/api/data')
            .then(response => response.json())
            .then(data => {
                updateMetrics(data);
                displayDataTable(data);
            });
        
        // Handle prediction form
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const params = new URLSearchParams({
                region: document.getElementById('region').value,
                district: document.getElementById('district').value,
                population: document.getElementById('population').value,
                income: document.getElementById('income').value,
                health: document.getElementById('health').value,
                education: document.getElementById('education').value
            });
            
            fetch('/api/predict?' + params)
                .then(response => response.json())
                .then(result => {
                    displayPredictionResult(result);
                });
        });
        
        function updateMetrics(data) {
            const totalPop = data.reduce((sum, row) => sum + parseInt(row.population), 0);
            const avgIncome = data.reduce((sum, row) => sum + parseFloat(row.income_per_capita), 0) / data.length;
            const totalAlloc = data.reduce((sum, row) => sum + parseInt(row.current_allocation), 0);
            
            document.getElementById('total-districts').textContent = data.length;
            document.getElementById('total-population').textContent = (totalPop / 1000000).toFixed(1) + 'M';
            document.getElementById('avg-income').textContent = '$' + Math.round(avgIncome);
            document.getElementById('total-allocation').textContent = '$' + (totalAlloc / 1000000).toFixed(1) + 'M';
        }
        
        function displayDataTable(data) {
            const table = document.createElement('table');
            table.className = 'data-table';
            
            // Header
            const header = table.createTHead();
            const headerRow = header.insertRow();
            const cols = ['District', 'Region', 'Population', 'Income', 'Health', 'Education', 'Allocation'];
            cols.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col;
                headerRow.appendChild(th);
            });
            
            // Data rows
            const tbody = table.createTBody();
            data.slice(0, 10).forEach(row => {  // Show first 10 rows
                const tr = tbody.insertRow();
                tr.insertCell().textContent = row.district;
                tr.insertCell().textContent = row.region;
                tr.insertCell().textContent = parseInt(row.population).toLocaleString();
                tr.insertCell().textContent = '$' + row.income_per_capita;
                tr.insertCell().textContent = row.health_index;
                tr.insertCell().textContent = row.education_index;
                tr.insertCell().textContent = '$' + parseInt(row.current_allocation).toLocaleString();
            });
            
            document.getElementById('data-table').innerHTML = '';
            document.getElementById('data-table').appendChild(table);
        }
        
        function displayPredictionResult(result) {
            const resultDiv = document.getElementById('prediction-result');
            resultDiv.innerHTML = `
                <div class="result">
                    <h3>Prediction Result</h3>
                    <p><strong>Predicted Allocation:</strong> $${result.prediction.toLocaleString()}</p>
                    <p><strong>Per Capita:</strong> $${result.per_capita.toFixed(2)}</p>
                    <p><strong>Calculation Details:</strong></p>
                    <ul>
                        <li>Base allocation: $${result.details.base.toLocaleString()}</li>
                        <li>Need factor: ${result.details.need_factor.toFixed(2)}</li>
                        <li>Income factor: ${result.details.income_factor.toFixed(2)}</li>
                        <li>Regional factor: ${result.details.regional_factor.toFixed(2)}</li>
                    </ul>
                </div>
            `;
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def send_data(self):
        """Send sample data as JSON."""
        data = load_data()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def handle_prediction(self):
        """Handle prediction API requests."""
        parsed_url = urlparse.urlparse(self.path)
        params = urlparse.parse_qs(parsed_url.query)
        
        try:
            region = params['region'][0]
            district = params['district'][0]
            population = int(params['population'][0])
            income = float(params['income'][0])
            health = float(params['health'][0])
            education = float(params['education'][0])
            
            prediction = simple_prediction(region, district, population, income, health, education)
            
            # Calculate details
            base_allocation = population * 10
            socioeconomic_index = (health + education) / 2
            need_factor = 1.5 - socioeconomic_index
            
            if income < 400:
                income_factor = 1.3
            elif income < 600:
                income_factor = 1.1
            else:
                income_factor = 0.9
            
            regional_factors = {'North': 1.2, 'East': 1.0, 'West': 1.1, 'Central': 0.9}
            regional_factor = regional_factors.get(region, 1.0)
            
            result = {
                'prediction': prediction,
                'per_capita': prediction / population,
                'details': {
                    'base': base_allocation,
                    'need_factor': need_factor,
                    'income_factor': income_factor,
                    'regional_factor': regional_factor
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode())

def main():
    """Run the demo server."""
    port = 8000
    print(f"🏛️ Smart Governance Analytics Demo")
    print(f"Starting server on http://localhost:{port}")
    print(f"Open your browser and navigate to: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    server = HTTPServer(('localhost', port), DemoHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()

if __name__ == "__main__":
    main()