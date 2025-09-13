#!/usr/bin/env python3
"""
Basic test script to demonstrate the ML governance solution structure
without requiring external dependencies.
"""

import csv
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append('src')
from config import SAMPLE_DATA_PATH, VALID_REGIONS, VALID_DISTRICTS

def load_data():
    """Load data using built-in CSV module."""
    with open(SAMPLE_DATA_PATH, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def validate_data(data):
    """Basic data validation."""
    print("=== Data Validation ===")
    
    # Check row count
    print(f"Total rows: {len(data)}")
    
    # Check columns
    expected_cols = ['region', 'district', 'population', 'income_per_capita', 
                    'health_index', 'education_index', 'current_allocation', 'year']
    actual_cols = list(data[0].keys())
    missing_cols = set(expected_cols) - set(actual_cols)
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
    else:
        print("✅ All required columns present")
    
    # Check regions
    regions = set(row['region'] for row in data)
    invalid_regions = regions - set(VALID_REGIONS)
    if invalid_regions:
        print(f"❌ Invalid regions: {invalid_regions}")
    else:
        print("✅ All regions valid")
    
    # Check for missing values
    missing_count = 0
    for row in data:
        for col, val in row.items():
            if not val.strip():
                missing_count += 1
    
    if missing_count > 0:
        print(f"⚠️  Found {missing_count} missing values")
    else:
        print("✅ No missing values")

def analyze_data(data):
    """Basic data analysis."""
    print("\n=== Data Analysis ===")
    
    # Regional distribution
    regional_counts = {}
    for row in data:
        region = row['region']
        regional_counts[region] = regional_counts.get(region, 0) + 1
    
    print("Districts by region:")
    for region, count in sorted(regional_counts.items()):
        print(f"  {region}: {count} districts")
    
    # Population statistics
    populations = [int(row['population']) for row in data]
    print(f"\nPopulation statistics:")
    print(f"  Min: {min(populations):,}")
    print(f"  Max: {max(populations):,}")
    print(f"  Avg: {sum(populations) // len(populations):,}")
    
    # Income statistics
    incomes = [float(row['income_per_capita']) for row in data]
    print(f"\nIncome per capita statistics:")
    print(f"  Min: ${min(incomes):.0f}")
    print(f"  Max: ${max(incomes):.0f}")
    print(f"  Avg: ${sum(incomes) / len(incomes):.0f}")
    
    # Allocation statistics
    allocations = [int(row['current_allocation']) for row in data]
    print(f"\nResource allocation statistics:")
    print(f"  Min: ${min(allocations):,}")
    print(f"  Max: ${max(allocations):,}")
    print(f"  Total: ${sum(allocations):,}")

def simple_prediction(region, district, population, income_per_capita, 
                     health_index, education_index):
    """Simple rule-based prediction without ML libraries."""
    print(f"\n=== Simple Prediction ===")
    print(f"Input: {district}, {region}")
    print(f"  Population: {population:,}")
    print(f"  Income per capita: ${income_per_capita:.0f}")
    print(f"  Health index: {health_index:.2f}")
    print(f"  Education index: {education_index:.2f}")
    
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
    
    print(f"\nPredicted allocation: ${predicted_allocation:,.0f}")
    print(f"  Base (${population:,} × $10): ${base_allocation:,}")
    print(f"  Need factor: {need_factor:.2f}")
    print(f"  Income factor: {income_factor:.2f}")
    print(f"  Regional factor: {regional_factor:.2f}")
    
    return predicted_allocation

def main():
    """Main test function."""
    print("🏛️ Smart Governance Analytics - Basic Test")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Validate data
    validate_data(data)
    
    # Analyze data
    analyze_data(data)
    
    # Example prediction
    prediction = simple_prediction(
        region="East",
        district="Kampala",
        population=1500000,
        income_per_capita=850,
        health_index=0.75,
        education_index=0.82
    )
    
    # Compare with actual
    kampala_actual = None
    for row in data:
        if row['district'] == 'Kampala':
            kampala_actual = int(row['current_allocation'])
            break
    
    if kampala_actual:
        difference = prediction - kampala_actual
        percentage = (difference / kampala_actual) * 100
        print(f"\nComparison with actual:")
        print(f"  Actual allocation: ${kampala_actual:,}")
        print(f"  Difference: ${difference:,.0f} ({percentage:+.1f}%)")
    
    print("\n✅ Basic test completed successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run dashboard: streamlit run src/dashboard.py")
    print("3. Explore notebook: jupyter notebook notebooks/EDA_and_Model.ipynb")

if __name__ == "__main__":
    main()