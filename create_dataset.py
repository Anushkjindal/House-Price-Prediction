"""
Script: create_dataset.py
Description: Generates a realistic, clearly-labelled demo dataset of Indian house prices.
This demo dataset covers major Indian cities (Mumbai, Delhi, Bangalore, Jaipur, Hyderabad, Pune)
with realistic square footage, BHK count, bathrooms, age, and pricing in Lakhs.
"""

import os
import random
import csv

# Set random seed for reproducibility
random.seed(42)

# City & Location mapping with baseline price per sqft (in INR)
# Mumbai: ~16,000 - 28,000 / sqft
# Delhi: ~8,000 - 16,000 / sqft
# Bangalore: ~6,500 - 13,000 / sqft
# Pune: ~5,500 - 10,000 / sqft
# Hyderabad: ~5,000 - 9,500 / sqft
# Jaipur: ~3,200 - 6,000 / sqft

city_locations = {
    "Mumbai": {
        "Bandra West": 26000,
        "Andheri West": 18000,
        "Powai": 17000,
        "Borivali": 14000,
        "Thane": 11000
    },
    "Delhi": {
        "Connaught Place": 22000,
        "Vasant Kunj": 15000,
        "Dwarka": 9500,
        "Saket": 14000,
        "Rohini": 8500
    },
    "Bangalore": {
        "Indiranagar": 14000,
        "Koramangala": 13000,
        "HSR Layout": 11000,
        "Whitefield": 8500,
        "Electronic City": 6000
    },
    "Hyderabad": {
        "Banjara Hills": 13000,
        "Hitec City": 9500,
        "Gachibowli": 9000,
        "Madhapur": 8500,
        "Kukatpally": 6000
    },
    "Pune": {
        "Kothrud": 10500,
        "Viman Nagar": 9800,
        "Baner": 9000,
        "Wakad": 7500,
        "Hinjewadi": 6800
    },
    "Jaipur": {
        "C-Scheme": 7500,
        "Malviya Nagar": 5800,
        "Vaishali Nagar": 5200,
        "Mansarovar": 4200,
        "Jagatpura": 3800
    }
}

os.makedirs("data", exist_ok=True)
csv_filepath = os.path.join("data", "house_prices.csv")

rows = []
# Header
header = ["City", "Location", "Area_sqft", "BHK", "Bathrooms", "Property_Age", "Price_in_Lakhs"]

# Generate 500 samples across the cities and localities
for _ in range(550):
    city = random.choice(list(city_locations.keys()))
    loc = random.choice(list(city_locations[city].keys()))
    base_rate_per_sqft = city_locations[city][loc]
    
    # BHK selection
    bhk = random.choices([1, 2, 3, 4, 5], weights=[0.15, 0.40, 0.30, 0.12, 0.03])[0]
    
    # Area based on BHK
    if bhk == 1:
        area = random.randint(450, 700)
        bathrooms = 1
    elif bhk == 2:
        area = random.randint(750, 1250)
        bathrooms = random.choice([1, 2, 2, 2])
    elif bhk == 3:
        area = random.randint(1200, 2000)
        bathrooms = random.choice([2, 3, 3, 4])
    elif bhk == 4:
        area = random.randint(1900, 3200)
        bathrooms = random.choice([3, 4, 4, 5])
    else: # 5 BHK
        area = random.randint(2800, 4800)
        bathrooms = random.choice([4, 5, 5])
        
    property_age = random.randint(0, 25) # in years
    
    # Calculate price:
    # 1. Base price from area and locality rate
    # 2. Bathroom bonus (+ 1.5 Lakhs per extra bathroom beyond 1)
    # 3. Age depreciation (- 0.6% per year of age)
    # 4. Realistic market noise (+- 5%)
    base_price = (area * base_rate_per_sqft) / 100000.0 # in Lakhs
    bathroom_adjustment = (bathrooms - 1) * 1.5
    age_factor = max(0.75, 1.0 - (property_age * 0.008))
    noise = random.uniform(0.96, 1.04)
    
    price_lakhs = round((base_price * age_factor + bathroom_adjustment) * noise, 2)
    
    rows.append([city, loc, area, bhk, bathrooms, property_age, price_lakhs])

with open(csv_filepath, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Dataset generated successfully at: {csv_filepath} with {len(rows)} records.")
