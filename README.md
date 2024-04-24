# Dairy Diary

## Description
Optimise trade allocation using linear programming.

## Features
- Fetch and display current FX spot rates from Yahoo Finance
- Edit freight rates, mappings, forward curves with user inputs
- Load data from user-uploaded Excel file and feeds it into the linear programming problem in PuLP, taking into account:
    1. Purchase, sale prices
    2. Purchase, sale quantities
    3. Delivery dates
    4. Freight rates
    5. Incoterm
    6. FX Rates/Historical FX Hedge Rates
    7. Quantities
    8. Approved supplier-purchaser flows
- Objective, decision variables, and constraints are created automatically
- Visualises the trade flows using Sankey diagrams
- Calculates potential profit
- Download trade allocations data in CSV

## Tech stack
- Python
- Streamlit
- Pandas
- Yfinance
- JSON
- Base64

## Installation 
1. Clone the repo 
2. Set up a Python virtual environment:
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Run the Streamlit app:
    ```
    streamlit run app.py
    ```
