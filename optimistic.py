import pandas as pd
import numpy as np
import pulp as pl
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

class OptimisticSMP:
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data_initial()
        self.MTM_prices = False
        self.forward_curves = None
        # self.forward_curves_filtered = None
        self.country_region_mappings = {}
        self.past_fx = False
        if 'forward_curves_editor' not in st.session_state:
            st.session_state['forward_curves_editor'] = {}
        self.freight_rates = {
        "Oceania": {
            "Philippines": 80,
            "China": 90,
            "Singapore": 85,
            "Bahrain": 150,
            "Saudi Arabia": 175,
            "Malaysia": 100,
            "Indonesia": 100,
            "Hong Kong": 90,
            "Thailand": 85,
            "Australia": 40,
            "WACC": 0.09,
            "Storage costs per month": "30 USD",
            "Inland freight": "40 USD"
        },
        "North America": {
            "Philippines": 130.0,
            "China": 120.0,
            "Singapore": 110.0,
            "Bahrain": 250.0,
            "Saudi Arabia": 230.0,
            "Malaysia": 150.0,
            "Indonesia": 140.0,
            "Hong Kong": 145.0,
            "Thailand": 160.0,
            "Australia": 140.0,
            "WACC": np.nan,
            "Storage costs per month": np.nan,
            "Inland freight": np.nan
        },
        "W-EU": {
            "Philippines": 60.0,
            "China": 40.0,
            "Singapore": 75.0,
            "Bahrain": 100.0,
            "Saudi Arabia": 120.0,
            "Malaysia": 70.0,
            "Indonesia": 75.0,
            "Hong Kong": 50.0,
            "Thailand": 70.0,
            "Australia": 80.0,
            "WACC": np.nan,
            "Storage costs per month": np.nan,
            "Inland freight": np.nan
        },
        "Asia": {
            "Philippines": 75,
            "China": 70,
            "Singapore": 80,
            "Bahrain": 150,
            "Saudi Arabia": 175,
            "Malaysia": 90,
            "Indonesia": 95,
            "Hong Kong": 85,
            "Thailand": 80,
            "Australia": 100,
            "WACC": 0,
            "Storage costs per month": "25 USD",
            "Inland freight": "35 USD"
        }
    }

        self.unique_pairs = set()

    def load_data_initial(self):
        """Load initial datasets from Excel files."""
        self.purchases_data = pd.read_excel(
            self.data_path, sheet_name='Physical Purchases')
        self.sales_data = pd.read_excel(
            self.data_path, sheet_name='Physical Sales')
        self.sales_data['PlaceOfDelivery'] = self.sales_data['PlaceOfDelivery'].replace('All countries', 'Saudi Arabia') #hard coded for now
        self.inventory_data = pd.read_excel(
            self.data_path, sheet_name='Physical Inventory')
        self.forex_data = pd.read_excel(
            self.data_path, sheet_name='Physical&Forex Pivot', header=2).dropna()
        self.forward_curves = pd.read_excel(
            self.data_path, sheet_name='Forward Curves', header=2)
        # st.write(self.forward_curves)
        self.approved_suppliers = pd.read_excel(
            self.data_path, sheet_name='Approved Supplier List', header=1)
        self.freights = pd.read_excel(
            self.data_path, sheet_name='Freight Rates - Wacc - Storage ', header=1)
        self.freights_lookup = self.freights.set_index(
            'PlaceOfDelivery')
        self.prepare_curves()
        self.add_freights_to_rates()

    def prepare_curves(self):
        """Prepare and filter forward curves data."""
        def _calc_future_date(period_code):
            months_ahead = int(period_code[1:])
            now = datetime.now()
            future_date = (now + pd.DateOffset(months=months_ahead)).date()
            return future_date

        forward_curves_filtered = self.forward_curves[self.forward_curves['Product'] == 'SMP MH']
        forward_curves_filtered['Input Adjustment'] = forward_curves_filtered['Price'] + \
            forward_curves_filtered['Carry']
        forward_curves_filtered['Period'] = forward_curves_filtered['Period'].apply(_calc_future_date)
        # if self.forward_curves_filtered.empty:
        self.forward_curves_filtered = forward_curves_filtered[['Period', 'Price', 'Carry', 'Input Adjustment', 'Market']]
        oceania_curves = self.forward_curves_filtered[self.forward_curves_filtered['Market'] == 'Oceania'].copy()
        oceania_curves['Market'] = 'Asia'
        self.forward_curves_filtered = pd.concat([self.forward_curves_filtered, oceania_curves], ignore_index=True)



    def set_curves(self, forward_curves):
        self.forward_curves = forward_curves
        self.forward_curves_filtered = self.forward_curves
    # def add_freights_to_rates(self):
    #     """Add custom freight rates to the freights DataFrame."""
    #     new_rows = pd.DataFrame({'PlaceOfDelivery': [
    #                             'Australia'], 'Oceania': 40, 'North America': 140, 'W-EU': 75})
    #     self.freights = pd.concat([self.freights, new_rows], axis=0)
    def add_freights_to_rates(self):
        pass 

    def prepare_inventory_and_sales(self):
        """Prepare inventory and sales data for optimization."""
        self.prepare_inventory()
        self.prepare_sales()
        self.prepare_optimization_formats()

    def prepare_inventory(self):
        p = self.purchases_data[['ExternalNr', 'Deliverylinenr', 'EntityDescription', 'AlphaName',
                                'Region', 'DeliveryTermCode', '2ndItemNr', 'ItemName',
                                 'Day of DateDeliverySchemeThru', 'CurrencyCode', 'Price',
                                 'Quantity (MT)']]
        p['PlaceOfDelivery'] = p['Region']
        self.inventory_data['Region'] = self.inventory_data['Origin']
        self.inventory_data['DeliveryTermCode'] = 'Inventory'
        self.inventory_data['Day of DateDeliverySchemeThru'] = datetime(
            2023, 1, 1)  # assume all inventory is available from 2023
        self.inventory_data['CurrencyCode'] = self.inventory_data['CompanyCurrencyCode']
        self.inventory_data['AlphaName'] = self.inventory_data['AlphaName'].str.strip(
        )
        self.inventory_data.loc[self.inventory_data['AlphaName']
                                == 'Northern Pastures', 'Region'] = 'Oceania'
        self.inventory_data.loc[self.inventory_data['AlphaName']
                                == 'Meadows Inc', 'Region'] = 'Asia'
        self.inventory_data['Region'].replace(
            ['New Zealand', 'Australia'], 'Oceania', inplace=True)
        self.inventory_data['PlaceOfDelivery'] = self.inventory_data['Region']
        self.i = self.inventory_data[p.columns]
        all_stocks = pd.concat([self.i, p], axis=0)
        # st.write(all_stocks)
        self.grouped_stocks = self.process_data(all_stocks)

    def prepare_sales(self):
        self.sales_data['AlphaName'] = self.sales_data['Company Name'].str.strip()
        all_sales = self.sales_data[self.i.columns].loc[self.sales_data['ItemName']
                                                        == 'Skimmed Milk Powder']
        
        # assume -2.9 in demand quantity is typo, should be 2.9 else it is a purchase under supply
        all_sales.replace(-2.9, 2.9, inplace=True)
        self.grouped_sales = self.process_data(all_sales)

    def money_changer_row(self, row):
        row_externalNr_as_int = int(row['ExternalNr'])

        if self.past_fx:
            self.forex_data['ExternalNr'] = self.forex_data['ExternalNr'].astype(
                int)

        if row_externalNr_as_int in self.forex_data['ExternalNr'].values and self.past_fx == True:
            rate = float(
                self.forex_data.loc[self.forex_data['ExternalNr'] == row_externalNr_as_int, 'Ave Rate'].iloc[0])
            row['Price'] = row['Price'] * rate
            return row
        else:
            row['Price'] = row['Price'] * \
                self.fx_rates.get(row['CurrencyCode'], 1)  # dictionary rate
            return row
        
    def find_closest_forward_curve(self, period_date, market):
        # Filter the curves for the specified market first
        # st.write(self.forward_curves_filtered)
        market_curves = self.forward_curves_filtered[self.forward_curves_filtered['Market'] == market].copy()
        # st.write(market_curves)
        # Then proceed with finding the closest curve as before
        # print(market_curves)
        if not market_curves.empty:
            market_curves['Period'] = pd.to_datetime(market_curves['Period'], format='%b-%Y')
            market_curves['Date Difference'] = market_curves['Period'].apply(
                lambda x: abs((x - period_date).days))
            closest_curve = market_curves.sort_values(
                by='Date Difference').iloc[0]
            return closest_curve['Input Adjustment']
        else:
            return None
        
    def display_edit_fwd_curves(self):
        st.sidebar.header("Edit Forward Curves for SMP MH")
        st.sidebar.caption(
            "Note: Asia curves absent; copied from Oceania for interim.")
        forward_curves_for_editing = self.forward_curves_filtered.copy()

        # 2. Format 'Period' column for display (%b-%Y)
        forward_curves_for_editing['Period'] = pd.to_datetime(
            forward_curves_for_editing['Period']
        ).dt.strftime('%b-%Y')

        forward_curves_for_editing = st.sidebar.data_editor(
            forward_curves_for_editing)
        self.forward_curves_filtered = forward_curves_for_editing

    def process_data(self, data_frame, item_name='Skimmed Milk Powder'):
        filtered_data = data_frame[data_frame['ItemName'] == item_name].copy()
        filtered_data['AlphaName'] = filtered_data['AlphaName'].str.strip()
        filtered_data['Date'] = pd.to_datetime(
            filtered_data['Day of DateDeliverySchemeThru'], format='%Y-%m-%d')

        filtered_data['ExternalNr'] = filtered_data['ExternalNr'].astype(int)
        filtered_data = filtered_data.apply(self.money_changer_row, axis=1)
        filtered_data['CurrencyCode'] = 'USD'
        filtered_data['Price'] = filtered_data['Price'].replace(0, self.forward_curves_filtered['Input Adjustment'][0])
        filtered_data['pq'] = filtered_data['Price'] * \
            filtered_data['Quantity (MT)']

        final_grouped = filtered_data.groupby(['ExternalNr', 'AlphaName', 'PlaceOfDelivery', 'DeliveryTermCode', 'ItemName', 'Date', 'CurrencyCode', 'Region']).agg(
            TotalQuantity=('Quantity (MT)', 'sum'),
            WeightedPriceSum=('pq', 'sum')
        ).reset_index()
        final_grouped['WeightedAveragePrice'] = final_grouped['WeightedPriceSum'] / \
            final_grouped['TotalQuantity']
        final_grouped.drop(columns=['WeightedPriceSum'], inplace=True)
        current_date = datetime.now().date()
        if self.MTM_prices:
            final_grouped['Region'] = final_grouped['Region'].apply(
                lambda x: self.country_region_mappings.get(x, x))

            final_grouped['AdjustedPrice'] = final_grouped.apply(
                lambda row: row['WeightedAveragePrice'] if row['Date'].date(
                ) <= current_date else self.find_closest_forward_curve(row['Date'], row['Region']),
                axis=1
            )
        return final_grouped

    def rename_city_region(self):
            # st.write(self.sales_df['PlaceOfDelivery'])

            # self.sales_df['PlaceOfDelivery'] = pd.merge(self.sales_df, self.sales_data[['ExternalNr', 'PlaceOfDelivery']], how='left', on='ExternalNr')['PlaceOfDelivery']
            self.sales_df.replace('Dalian', 'China', inplace=True)
            self.sales_df.replace('Melbourne', 'Australia', inplace=True)
            self.stocks_df.replace('Western Europe', 'W-EU', inplace=True)
            # self.stocks_df.replace('Asia', 'Oceania', inplace=True)
            # import streamlit as st
            # st.write('\n\n', self.stocks_df.Region.unique(), self.sales_df.Region.unique())
    def map_to_closest_freight_location(self, location):
        mapping = {
            "All countries": "Saudi Arabia",
            "Vietnam": "China",
            "Port Klang": "Malaysia",
            "Tianjin Xingang": "China",
            "Lat Krabang": "Thailand",
            "Jakarta": "Indonesia",
            "Sydney": "Australia",
            "Taiwan": "China",
            "Xingang": "China",
            "Republic of Korea": "China",
            "Senegal": "Saudi Arabia",
            "Lebanon": "Saudi Arabia",
            "Yemen": "Saudi Arabia",
            "Algeria": "Saudi Arabia",
            "Nigeria": "Saudi Arabia",
            "Australia": "Indonesia",
        }
        return mapping.get(location, location)

    def _get_freight(self, origin, destination):
        if origin == destination:
            return 40  # Inland freight cost
        # Simplified to work directly with the JSON structure
        origin_mapped = self.map_to_closest_freight_location(origin)
        destination_mapped = self.map_to_closest_freight_location(destination)
        try:
            rate = self.freight_rates[origin_mapped][destination_mapped]
            return rate
        except KeyError:
            # print(
                # f"Missing Freight Rate: Origin - {origin_mapped}, Destination - {destination_mapped}")
            self.unique_pairs.add((origin_mapped, destination_mapped))
            return 0


    def _create_freight_rates_dict(self): # add cost of carrying inventory
        self.freight_rates_dict = {}  
        print(self.sales_df.head(), self.stocks_df.head())
        for _, sale_row in self.sales_df.iterrows():
            sale_term = sale_row['DeliveryTermCode']
            sale_destination = sale_row['PlaceOfDelivery']

            for _, stock_row in self.stocks_df.iterrows():
                purchase_term = stock_row['DeliveryTermCode']
                stock_origin = stock_row['PlaceOfDelivery']

                # Combine purchase and sale external numbers for unique key
                pair_key = (stock_row['ExternalNr'], sale_row['ExternalNr'])
                if purchase_term == 'Inventory':
                    self.freight_rates_dict[pair_key] = 0
                    continue

                # Calculate purchase leg freight cost (origin based on purchase term)
                purchase_origin = stock_origin if purchase_term in ('FOB', 'EXW') else sale_destination
                purchase_freight_cost = self._get_freight(purchase_origin, sale_destination)

                # Calculate sale leg freight cost (origin depends on sale term)
                sale_origin = stock_origin if sale_term in ('CFR', 'CIF', 'FCA') else sale_destination
                sale_freight_cost = self._get_freight(sale_origin, sale_destination)

                total_freight_cost = purchase_freight_cost + sale_freight_cost 

                self.freight_rates_dict[pair_key] = total_freight_cost  
        # if self.unique_pairs:
        #     st.write(set(self.unique_pairs)) 
         # st.write(self.sales_df['PlaceOfDelivery'])
        # st.write({str(k): v for k, v in self.freight_rates_dict.items()})
        # self.sales_df.drop(['PlaceOfDelivery'], axis=1, inplace=True)

    def _init_approved_flow(self):
        approved_flows = self.approved_suppliers[['Company Name', 'Dairy Plus ', 'Meadows Inc', 'Advantage Plus',
                                                  "Milk R'Us    ", 'Cows Inc            ', 'Northern Pastures',]]
        # strip column names
        approved_flows.columns = approved_flows.columns.str.strip()
        self.approved_flow_dict = {row['Company Name']: row.drop(
            'Company Name').to_dict() for _, row in approved_flows.iterrows()}
        
    def _get_approved_sellers(self, buyer):
        return [seller for seller, approved in self.approved_flow_dict[buyer].items() if approved == 'P']

    def prepare_optimization_formats(self):
        self.stocks_df = self.grouped_stocks.copy()
        self.sales_df = self.grouped_sales.copy()
        self.stocks_df['Date'] = pd.to_datetime(self.stocks_df['Date'])
        self.sales_df['Date'] = pd.to_datetime(self.sales_df['Date'])

        self.stocks_df['Quantity (MT)'] = pd.to_numeric(
            self.stocks_df['TotalQuantity'])
        self.sales_df['Quantity (MT)'] = pd.to_numeric(
            self.sales_df['TotalQuantity'])

        self.stocks_df['Price'] = pd.to_numeric(
            self.stocks_df['WeightedAveragePrice'])
        self.sales_df['Price'] = pd.to_numeric(
            self.sales_df['WeightedAveragePrice'])
        if self.MTM_prices:
            self.stocks_df['Price'] = pd.to_numeric(
                self.stocks_df['AdjustedPrice'])
            self.sales_df['Price'] = pd.to_numeric(
                self.sales_df['AdjustedPrice'])

        self.rename_city_region()
        self._create_freight_rates_dict()

        self.purchase_prices_dict = self.stocks_df.set_index('ExternalNr')[
            'Price'].to_dict()
        self.sale_prices_dict = self.sales_df.set_index('ExternalNr')[
            'Price'].to_dict()
        self.max_qty_p_dict = self.stocks_df.set_index(
            'ExternalNr')['Quantity (MT)'].to_dict()
        self.demand_dict = self.sales_df.set_index(
            'ExternalNr')['Quantity (MT)'].to_dict()
        self._init_approved_flow()

    def prepare_data(self):
        # self.load_data_initial()
        self.prepare_inventory_and_sales()
        self.purchases = list(self.purchase_prices_dict.keys())

        self.sales = list(self.sale_prices_dict.keys())
        self.S = self.sale_prices_dict
        # st.write(self.S)
        self.P = self.purchase_prices_dict
        # st.write(self.P)
        self.C = self.freight_rates_dict
        self.Q = self.max_qty_p_dict
        self.D = self.demand_dict

        self.purchase_to_seller = pd.Series(
            self.stocks_df.AlphaName.values, index=self.stocks_df.ExternalNr).to_dict()

        self.sale_to_buyer = pd.Series(self.sales_df.AlphaName.values,
                                       index=self.sales_df.ExternalNr).to_dict()

    def setup_optimization(self):
        # unmet_demand_penalty = 100000
        # print(len(self.purchases), len(self.sales))
        prob = pl.LpProblem("OptimisticSMP", pl.LpMaximize)
        # Decision Variables, considering only approved seller-buyer pairs
        X = {}
        for i in self.purchases:
            purchase_date = self.stocks_df.loc[self.stocks_df['ExternalNr']
                                               == i, 'Date'].values[0]
            seller = self.purchase_to_seller[i]
            for j in self.sales:
                sale_date = self.sales_df.loc[self.sales_df['ExternalNr']
                                              == j, 'Date'].values[0]
                buyer = self.sale_to_buyer[j]
                if seller in self._get_approved_sellers(buyer) and purchase_date <= sale_date:
                    X[(i, j)] = pl.LpVariable(f'X_{i}_{j}', lowBound=0)

        # Objective Function
        prob += pl.lpSum([(self.S[j] - self.P[i] - self.C[(i, j)]) * X[(i, j)]
                          for (i, j) in X.keys()])
        # Supply Constraints: Each inventory can be allocated fully but not exceeded.
        for i in self.purchases:
            prob += pl.lpSum([X[(i, j)] for j in self.sales if (i, j)
                             in X]) <= self.Q[i], f"Supply_{i}"
        # Demand Constraints: Each demand must be exactly met.
        for j in self.sales:
            prob += pl.lpSum([X[(i, j)] for i in self.purchases if (i, j)
                            in X]) == self.D[j], f"Demand_{j}"

        prob.solve()
        # Printing the solution
        if pl.LpStatus[prob.status] == 'Optimal':
            print("Optimal Solution Found:")
            for var in X.values():
                if var.varValue > 0:
                    # print seller and buyer names by extracting from sales_df or stocks_df)
                    purchase_id_str, sale_id_str = var.name.split('_')[1:]
                    # Convert IDs to float for dictionary lookup
                    purchase_id = float(purchase_id_str)
                    sale_id = float(sale_id_str)
                    # Lookup seller and buyer names using the IDs
                    seller_name = self.purchase_to_seller.get(
                        purchase_id, "Unknown Seller")
                    buyer_name = self.sale_to_buyer.get(
                        sale_id, "Unknown Buyer")
                    print(f"from {buyer_name} to {seller_name}")
                    print(f"{var.name} = {var.varValue}")
        else:
            print("Optimal solution not found. Status:",
                  pl.LpStatus[prob.status])
        # Check if the problem was solved successfully
        if prob.status == pl.LpStatusOptimal:
            total_profit = pl.value(prob.objective)
            print(f"Total Profit: ${total_profit:,.2f}")
        else:
            print("The problem does not have an optimal solution.")

        self.total_profit = total_profit
        self.X = X

    def adjust_overfulfilled_sales(self, allocations_df):
            # First, re-calculate the total allocated quantity for each sale ID to check current over-fulfillments
        grouped_allocations = allocations_df.groupby('SaleID').agg({
            'AllocatedQuantity': 'sum',
            'SaleQuantity': 'first'  # Assuming the same SaleQuantity for each group
        }).reset_index()

        # Identify overfulfilled sales
        overfulfilled = grouped_allocations[grouped_allocations['AllocatedQuantity']
                                            > grouped_allocations['SaleQuantity']]

        # Adjust the allocated quantities
        for index, row in overfulfilled.iterrows():
            sale_id = row['SaleID']
            total_allocated = row['AllocatedQuantity']
            total_demand = row['SaleQuantity']
            excess = total_allocated - total_demand

            # Find rows in the original df to adjust
            sale_allocations = allocations_df[allocations_df['SaleID'] == sale_id]

            # Proportional reduction
            sale_allocations['AllocatedQuantity'] -= (
                sale_allocations['AllocatedQuantity'] / total_allocated) * excess

            # Ensure that we do not drop below zero due to floating point arithmetic issues
            sale_allocations['AllocatedQuantity'] = sale_allocations['AllocatedQuantity'].clip(
                lower=0)

            # Update the main dataframe
            allocations_df.loc[allocations_df['SaleID'] == sale_id,
                            'AllocatedQuantity'] = sale_allocations['AllocatedQuantity']

        # Re-check if any overfulfillment still exists
        grouped_allocations_after = allocations_df.groupby('SaleID').agg({
            'AllocatedQuantity': 'sum',
            'SaleQuantity': 'first'
        }).reset_index()
        overfulfilled_after = grouped_allocations_after[grouped_allocations_after[
            'AllocatedQuantity'] > grouped_allocations_after['SaleQuantity'] + 0.01]

        # if not overfulfilled_after.empty:
        #     print("Adjustment incomplete, further checks needed.")
        # else:
        #     print("All overfulfillments adjusted.")

        # Round numerical columns to 2 decimal places
        allocations_df = allocations_df.round(2)

        return allocations_df


    def get_allocations_df(self):
        self.stocks_df['PlaceOfDelivery'] = self.stocks_df['PlaceOfDelivery'].apply(
            lambda x: self.country_region_mappings.get(x, x))
        # st.write(self.stocks_df, self.sales_df)
        # st.write(self.stocks_df['Region'].unique())
        # st.write(self.stocks_df[self.stocks_df['Region'] == 'All countries'])
        detailed_allocations = []
        unfulfilled_sales = []
        overfulfilled_sales = []
        # Augment dataframes with unique identifiers for each row
        stocks_augmented = self.stocks_df.reset_index().rename(
            columns={'index': 'StockUniqueID'})
        sales_augmented = self.sales_df.reset_index().rename(
            columns={'index': 'SaleUniqueID'})

        # Iterate through each decision variable in self.X
        for (purchase_id, sale_id), variable in self.X.items():
            quantity = variable.varValue  # Get the allocated quantity from the decision variable
            if quantity:  # Process only allocations with a positive quantity
                purchase_details = stocks_augmented.loc[stocks_augmented['ExternalNr']
                                                        == purchase_id].iloc[0]
                sale_details = sales_augmented.loc[sales_augmented['ExternalNr']
                                                   == sale_id].iloc[0]
                # import streamlit as st
                # st.write(purchase_details)
                detailed_allocations.append({
                    'PurchaseID': purchase_id,
                    'PurchaseAlphaName': purchase_details['AlphaName'],
                    'PurchaseDate': purchase_details['Date'].strftime('%Y-%m-%d'),
                    'PurchasePlaceOfDelivery': purchase_details.get('Region', 'Unknown'),
                    'PurchaseQuantity': purchase_details['TotalQuantity'],
                    'PurchaseWeightedAveragePrice': f"{purchase_details['Price']:,.2f}",

                    'SaleID': sale_id,
                    'SaleAlphaName': sale_details['AlphaName'],
                    'SaleDate': sale_details['Date'].strftime('%Y-%m-%d'),
                    'SalePlaceOfDelivery': sale_details.get('PlaceOfDelivery', 'Unknown'),
                    'SaleQuantity': sale_details['TotalQuantity'],
                    'SaleWeightedAveragePrice': f"{sale_details['Price']:,.2f}",

                    'AllocatedQuantity': quantity
                })

        allocations_df = pd.DataFrame(detailed_allocations)
        # Adjust overfulfilled sales
        allocations_df = self.adjust_overfulfilled_sales(allocations_df)
        self.allocations_df = allocations_df
        grouped_allocations = allocations_df.groupby(
            'SaleID')['AllocatedQuantity'].sum()
        for sale_id, total_allocated_quantity in grouped_allocations.items():
            total_demand = allocations_df.loc[allocations_df['SaleID']
                                              == sale_id, 'SaleQuantity'].iloc[0]
            if total_allocated_quantity < total_demand -0.01:
                unfulfilled_sales.append({
                    'SaleID': sale_id,
                    'SaleAlphaName': allocations_df.loc[allocations_df['SaleID'] == sale_id, 'SaleAlphaName'].iloc[0],
                    'TotalDemand': total_demand,
                    'AllocatedQuantity': total_allocated_quantity
                })
            elif total_allocated_quantity > total_demand + 0.01:
                overfulfilled_sales.append({
                    'SaleID': sale_id,
                    'SaleAlphaName': allocations_df.loc[allocations_df['SaleID'] == sale_id, 'SaleAlphaName'].iloc[0],
                    'TotalDemand': total_demand,
                    'AllocatedQuantity': total_allocated_quantity
                })

        # Print information about unfulfilled and overfulfilled sales orders
        if unfulfilled_sales:
            print("Unfulfilled Sales Orders:")
            for order in unfulfilled_sales:
                print(f"Sale ID: {order['SaleID']}, AlphaName: {order['SaleAlphaName']}, "
                      f"Total Demand: {order['TotalDemand']}, Allocated Quantity: {order['AllocatedQuantity']}")
        if overfulfilled_sales:
            print("Overfulfilled Sales Orders:")
            for order in overfulfilled_sales:
                print(f"Sale ID: {order['SaleID']}, AlphaName: {order['SaleAlphaName']}, "
                      f"Total Demand: {order['TotalDemand']}, Allocated Quantity: {order['AllocatedQuantity']}")

        return allocations_df, unfulfilled_sales, overfulfilled_sales

    def plot_sankey(self, source_column, target_column, title='Sankey Diagram'):
        df = self.allocations_df.copy()  # Avoid modifying original DataFrame

        # Sort by purchase date for both sides
        # Sort by purchase date, then source
        df = df.sort_values(by=['PurchaseDate', source_column])

        # Collect unique entities
        source_entities = df[source_column].unique().tolist()
        target_entities = df[target_column].unique().tolist()

        # Create prefixed labels with sorting applied within each category
        source_labels = [f"{ent} Purchase" for ent in sorted(source_entities)]
        target_labels = [f"{ent} Sale" for ent in sorted(target_entities)]

        # Combine and map labels to indices
        labels = source_labels + target_labels
        entity_to_index = {name: idx for idx, name in enumerate(labels)}

        # Generate sources, targets, and values based on sorted DataFrame
        sources = df[source_column].map(
            lambda x: entity_to_index[f"{x} Purchase"]).tolist()
        targets = df[target_column].map(
            lambda x: entity_to_index[f"{x} Sale"]).tolist()
        values = df['AllocatedQuantity'].tolist()

        # Calculate total allocated quantity for percentage calculations
        total_quantity = sum(values)
        # Include percentages in hover text by modifying the value list
        value_texts = [
            f"{v} MT ({(v / total_quantity * 100):.2f}%)" for v in values]

        # Create and display Sankey diagram with adjustments
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                label=value_texts  # Label is shown in the hover by default
            ))])

        fig.update_layout(title_text=title, font_size=12, width=400)
        st.plotly_chart(fig)
        