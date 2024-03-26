import pandas as pd
import streamlit as st
import yfinance as yf
from optimistic import OptimisticSMP
import base64
from datetime import datetime, timedelta
import json
import numpy as np

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
    freight_rates_json = st.sidebar.text_area(
        "Edit Freight Rates (JSON)", json.dumps(moneymaker.freight_rates, indent=4), height=300)

    try:
        freight_rates_dict = json.loads(freight_rates_json)
        moneymaker.freight_rates = freight_rates_dict
        # st.write("Freight rates updated successfully.")
    except json.JSONDecodeError:
        st.sidebar.error(
            "Invalid JSON format. Please provide valid JSON data.")


def display_and_edit_country_region_mappings(moneymaker):
    st.sidebar.header("Edit Country-Region Mappings for FWD Curves")
    default_mappings = {
        "Indonesia": "Asia",
        "Saudi Arabia": "Asia",
        "Philippines": "Asia",
        "China": "Asia",
        "Australia": "Oceania",
        "Asia": "Asia",
        "North America": "US",
        "Western Europe": 'W-EU',
        "Middle east": "Asia",
    }

    mappings_json = st.sidebar.text_area("Edit Country-Region Mappings (JSON)", json.dumps(
        default_mappings, indent=4), height=200)
    try:
        mappings_dict = json.loads(mappings_json)
        moneymaker.country_region_mappings = mappings_dict
        st.sidebar.success("Mappings updated successfully.")
    except json.JSONDecodeError as e:
        st.sidebar.error(
            f"Invalid JSON format: {e.msg} at line {e.lineno}, column {e.colno}. Please provide valid JSON data.")


def display_and_edit_forward_curves(moneymaker):
    st.sidebar.header("Edit Forward Curves for SMP MH")
    st.sidebar.caption(
        "Note: Asia curves absent; copied from Oceania for interim.")
    forward_curves_for_editing = moneymaker.forward_curves_filtered.copy()

    # 2. Format 'Period' column for display (%b-%Y)
    forward_curves_for_editing['Period'] = pd.to_datetime(
        forward_curves_for_editing['Period']
    ).dt.strftime('%b-%Y')

    forward_curves_for_editing = st.sidebar.data_editor(
        forward_curves_for_editing)
    moneymaker.forward_curves_filtered = forward_curves_for_editing
    # st.sidebar.header("Edit Forward Curves for SMP MH")

    # edited_curves_json = st.sidebar.text_area("Edit Forward Curves (JSON)", moneymaker.default_curves_json, height=200)
    # try:

    #     edited_curves_dict = json.loads(edited_curves_json)
    #     edited_forward_curves = pd.read_json(json.dumps(edited_curves_dict), orient='split')
        
    #     moneymaker.forward_curves_filtered = edited_forward_curves
    #     st.sidebar.success("Forward curves updated successfully.")
    # except json.JSONDecodeError as e:
    #     st.sidebar.error(f"Invalid JSON format: {e.msg} at line {e.lineno}, column {e.colno}. Please provide valid JSON data.")
    # except Exception as e:
    #     st.sidebar.error(f"An error occurred: {str(e)}")


def main():
    optimised = False

    st.set_page_config(layout="wide", page_title="Dairy Diary", page_icon='sticker.png')
    st.sidebar.caption(
        "Note: When editing JSON, curly quotes will cause an error. Use straight quotes.")

    st.markdown(
        "<h1 style='text-align: center; color: white;'>Dairy Diary 🐮</h1>",
        unsafe_allow_html=True,
    )
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'optimised' not in st.session_state:
        st.session_state.optimised = False

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
        with st.spinner('Initialising...'):
            moneymaker = OptimisticSMP(uploaded_file)
            # display_forward_curves(moneymaker)
        file_uploaded = True
    else:
        file_uploaded = False

    # Fetch spot rates from Yahoo Finance
    spot_rates = get_spot_rates()

    # if st.sidebar.checkbox('Show FX Rates', value=True):    
    st.sidebar.header("Set FX Rates")
    st.sidebar.caption("Note: Default rates are current spot rates from Yahoo Finance")
    fx_rates={}
    fx_rates['USD'] = 1.0
    currencies = ['EUR', 'CNY', 'AUD']
    for currency in currencies:
        default_rate = spot_rates.get(currency, 1.0)
        fx_rates[currency] = st.sidebar.number_input(
            f"{currency}USD", value=default_rate, format="%.5f")
 
    if st.sidebar.checkbox('Use FX Pivot Transacted Rates', value=True) and file_uploaded:
        st.sidebar.caption('Note: Rates available in the data file will be used, remaining use spot')
        moneymaker.past_fx = True
        # pass
    if st.sidebar.checkbox('Use Forward Curves for Purchase, Sales Prices', value=False):
        st.sidebar.caption('Note: Historical prices will be used for old trades, future prices will be MTM')
        moneymaker.MTM_prices = True


    if uploaded_file is not None:
        display_and_edit_freight_rates(moneymaker)
        display_and_edit_forward_curves(moneymaker)
        display_and_edit_country_region_mappings(moneymaker)
        # moneymaker.display_editable_forward_curves()

    if st.button('Run Optimisation', disabled=not file_uploaded):
        with st.spinner('Optimising...'):
            if 'fx_rates' in locals():
                # Set FX rates based on user input
                moneymaker.fx_rates = fx_rates

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
            col1, col2, col3 = st.columns(3)
            with col1:
                moneymaker.plot_sankey('PurchasePlaceOfDelivery', 'SalePlaceOfDelivery', 'Flow from Purchase to Sale Places')
            with col2:
                moneymaker.plot_sankey('PurchaseAlphaName', 'SaleAlphaName', 'Flow between Purchase and Sale Companies')
            with col3:
                moneymaker.plot_sankey('PurchaseDate', 'SaleDate', 'Flow between Purchase and Sale Dates')
            # moneymaker.plot_allocation_trends()


            # Download link for allocations
            csv = allocations_df.to_csv(index=False)
            # Encode CSV string to base64
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="trade_allocations.csv">Download Trade Allocations CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            if moneymaker.MTM_prices:
                st.header(f'Max Possible Profit (MTM prices): $ {moneymaker.total_profit:,.2f}')
            else:
                st.header(f'Max Possible Profit (indicated prices): $ {moneymaker.total_profit:,.2f}')
            st.markdown('---')
            # Display underfulfilled sales orders
            if unfulfilled_sales:
                st.subheader('Unfulfilled Sales Orders')
                st.dataframe(pd.DataFrame(unfulfilled_sales))

            # st.write(moneymaker.forward_curves_filtered)

            # # Display overfulfilled sales orders
            # if overfulfilled_sales:
            #     st.subheader('Overfulfilled Sales Orders')
            #     st.dataframe(pd.DataFrame(overfulfilled_sales))



if __name__ == "__main__":
    main()
