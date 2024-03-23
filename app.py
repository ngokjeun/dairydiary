import pandas as pd
import streamlit as st
import yfinance as yf
from optimistic_smp import OptimisticSMP
import base64
from datetime import datetime, timedelta
import json

# Function to fetch spot rates from Yahoo Finance


def get_spot_rates():
    # Get spot rates for major currencies against USD
    major_currencies = ['EURUSD=X', 'CNYUSD=X', 'AUDUSD=X']
    spot_rates = {}
    start_date = datetime.today() - timedelta(days=365)  # Fetch data for the last year
    for currency_pair in major_currencies:
        data = yf.download(currency_pair, start=start_date,
                           end=datetime.today())
        if not data.empty:
            spot_rate = data['Close'][-1]
            currency = currency_pair.replace('USD=X', '')
            spot_rates[currency] = spot_rate
    return spot_rates


def display_and_edit_freight_rates(moneymaker):
    st.sidebar.header("Edit Freight Rates")
    default_freight_rates = moneymaker.freights_lookup.to_dict()

    # Display a text area for users to input JSON data
    freight_rates_json = st.sidebar.text_area("Edit Freight Rates (JSON)", json.dumps(
        default_freight_rates, indent=4), height=200)

    # Parse the JSON input
    try:
        freight_rates_dict = json.loads(freight_rates_json)
        moneymaker.freights_lookup = pd.DataFrame(freight_rates_dict)
    except json.JSONDecodeError:
        st.sidebar.error(
            "Invalid JSON format. Please provide valid JSON data.")
        return

def main():
    st.set_page_config(layout="wide", page_icon='🍼')
    st.markdown(
        "<h1 style='text-align: center; color: white;'>Dairy Diary 🐮</h1>",
        unsafe_allow_html=True,
    )

    hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # st.title('Dairy Diary 🐮')
    # File uploader allows user to upload their own Excel data
    uploaded_file = st.file_uploader("Choose a data file", type=['xlsm'])

    if uploaded_file is not None:
        # Assuming the uploaded file is saved temporarily
        with st.spinner('Initiating...'):
            moneymaker = OptimisticSMP(uploaded_file)
            moneymaker.load_data()
        file_uploaded = True
    else:
        file_uploaded = False

    # Fetch spot rates from Yahoo Finance
    spot_rates = get_spot_rates()

    # if st.sidebar.checkbox('Show FX Rates', value=True):    
    st.sidebar.header("Set FX Rates: Spot Default")
    fx_rates={}
    fx_rates['USD'] = 1.0
    currencies = ['EUR', 'CNY', 'AUD']
    for currency in currencies:
        default_rate = spot_rates.get(currency, 1.0)
        fx_rates[currency] = st.sidebar.number_input(
            f"{currency}", value=default_rate, format="%.5f")
    if uploaded_file is not None:
        display_and_edit_freight_rates(moneymaker)

    if st.button('Run Optimisation', disabled=not file_uploaded):
        with st.spinner('Breastfeeding...'):
            if 'fx_rates' in locals():
                # Set FX rates based on user input
                moneymaker.set_fx_rates(fx_rates)

            moneymaker.prepare_data()
            moneymaker.setup_optimization()
            # Get DataFrame of allocations and lists of unfulfilled and overfulfilled sales orders
            allocations_df, unfulfilled_sales, overfulfilled_sales = moneymaker.get_allocations_df()

            st.success('Optimisation completed successfully!')
            # Visualize the allocations
            st.subheader('Trade Allocations')
            allocations_table = allocations_df.reset_index(drop=True)
            allocations_table['PurchaseID'] = allocations_table['PurchaseID'].astype(int).astype(str)
            allocations_table['SaleID'] = allocations_table['SaleID'].astype(
                int).astype(str)
            st.dataframe(allocations_table)

            # Display underfulfilled sales orders
            if unfulfilled_sales:
                st.subheader('Unfulfilled Sales Orders')
                st.dataframe(pd.DataFrame(unfulfilled_sales))

            # Display overfulfilled sales orders
            if overfulfilled_sales:
                st.subheader('Overfulfilled Sales Orders')
                st.dataframe(pd.DataFrame(overfulfilled_sales))

            # Download link for allocations
            csv = allocations_df.to_csv(index=False)
            # Encode CSV string to base64
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="trade_allocations.csv">Download Trade Allocations CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.header(f'Max Possible Profit (indicated prices): $ {moneymaker.total_profit:,.2f}')



if __name__ == "__main__":
    main()
