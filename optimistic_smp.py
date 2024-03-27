import pandas as pd
import pulp as pl
from datetime import datetime


class OptimisticSMP:
    def __init__(self, data_path):
        self.data_path = data_path
        self.purchases_data = None
        self.sales_data = None
        self.inventory_data = None
        self.forward_curves = None
        self.approved_suppliers = None
        self.freights = None
        self.prob = None
        self.fx_rates = {}


    def load_data(self):
        """
        Load data from the data_path
        """
        self.purchases_data = pd.read_excel(
            self.data_path, sheet_name='Physical Purchases')
        self.sales_data = pd.read_excel(
            self.data_path, sheet_name='Physical Sales')
        self.inventory_data = pd.read_excel(
            self.data_path, sheet_name='Physical Inventory')
        self.forward_curves = pd.read_excel(
            self.data_path, sheet_name='Forward Curves')
        self.approved_suppliers = pd.read_excel(
            self.data_path, sheet_name='Approved Supplier List', header=1)
        self.freights = pd.read_excel(
            self.data_path, sheet_name='Freight Rates - Wacc - Storage ', header=1)
        self.freights_lookup = self.freights.copy()
        self.freights_lookup.set_index('PlaceOfDelivery', inplace=True)

    def set_fx_rates(self, fx_rates):
        if isinstance(fx_rates, dict):
            self.fx_rates = fx_rates
        else:
            raise ValueError("FX rates must be provided as a dictionary.")

    def add_freights(self):
        new_rows = pd.DataFrame({'PlaceOfDelivery': [
                                'Australia'], 'Oceania': 40, 'North America': 140, 'W-EU': 75})
        self.freights = pd.concat([self.freights, new_rows], axis=0)

    def process_data(self, data_frame, item_name='Skimmed Milk Powder'):
        """
        Processes the given DataFrame to calculate the total and weighted average price
        for the specified item.

        Parameters:
        - data_frame: The DataFrame to process.
        - item_name: The name of the item to filter by. Defaults to 'Skimmed Milk Powder'.

        Returns:
        - A processed DataFrame with calculated total quantities and weighted average prices.
        """
        # Filter by item name
        filtered_data = data_frame.loc[data_frame['ItemName'] == item_name]
        filtered_data['AlphaName'] = filtered_data['AlphaName'].str.strip()
        filtered_data['Date'] = pd.to_datetime(filtered_data['Day of DateDeliverySchemeThru'], format='%Y/%m/%d')
        
        # Calculate the product of price and quantity
        filtered_data['pq'] = filtered_data['Price'] * filtered_data['Quantity (MT)']
        
        # Group by relevant columns and aggregate
        final_grouped = filtered_data.groupby(['ExternalNr', 'AlphaName', 'Region', 'DeliveryTermCode', 'ItemName', 'Date', 'CurrencyCode']).agg(
            TotalQuantity=('Quantity (MT)', 'sum'),
            WeightedPriceSum=('pq', 'sum')  
        ).reset_index()

        # Calculate weighted average price and clean up
        final_grouped['WeightedAveragePrice'] = final_grouped['WeightedPriceSum'] / final_grouped['TotalQuantity']
        final_grouped.drop(columns=['WeightedPriceSum'], inplace=True)

        return final_grouped

    def prepare_inventory(self):
        p = self.purchases_data[['ExternalNr', 'Deliverylinenr', 'EntityDescription', 'AlphaName',
                                'Region', 'DeliveryTermCode', '2ndItemNr', 'ItemName',
                                'Day of DateDeliverySchemeThru', 'CurrencyCode', 'Price',
                                'Quantity (MT)']]
        self.inventory_data['Region'] = self.inventory_data['Origin']
        self.inventory_data['DeliveryTermCode'] = 'Inventory'
        self.inventory_data['Day of DateDeliverySchemeThru'] = datetime(2023, 1, 1) # assume all inventory is available from 2023
        self.inventory_data['CurrencyCode'] = self.inventory_data['CompanyCurrencyCode']
        self.i = self.inventory_data[p.columns]

        all_stocks = pd.concat([self.i, p], axis=0)
        self.grouped_stocks = self.process_data(all_stocks)

    def prepare_sales(self):
        self.sales_data['AlphaName'] = self.sales_data['Company Name'].str.strip()
        all_sales = self.sales_data[self.i.columns].loc[self.sales_data['ItemName']
                                                        == 'Skimmed Milk Powder']
        # assume 0s in prices are intentional - refund for claim, etc.
        # assume -2.9 in demand quantity is typo, should be 2.9 else it is a purchase under supply
        all_sales.replace(-2.9, 2.9, inplace=True)
        self.grouped_sales = self.process_data(all_sales)

    def rename_city_region(self):
        self.sales_df.replace('Dalian', 'China', inplace=True)
        self.sales_df.replace('Melbourne', 'Australia', inplace=True)
        self.stocks_df.replace('Western Europe', 'W-EU', inplace=True)
        self.stocks_df.replace('Asia', 'Oceania', inplace=True)

    def _get_freight(self, origin, destination):
        if destination in self.freights_lookup.index and origin in self.freights_lookup.columns:
            return self.freights_lookup.loc[destination, origin]
        elif destination == origin:
            print(origin, destination)
            return 40
        else:
            print(origin, destination)
            return 0  # Return 0 or some default value if no match is found

    def _create_freight_rates_dict(self):
        self.freight_rates_dict = {}
        for index, sale_row in self.sales_df.iterrows():
            for _, stock_row in self.stocks_df.iterrows():
                origin = stock_row['Region']
                destination = sale_row['Region']
                if origin == destination:

                    print(origin, destination)
                pair_key = (stock_row['ExternalNr'], sale_row['ExternalNr'])
                self.freight_rates_dict[pair_key] = self._get_freight(
                    origin, destination)

    def _convert_fx(self):
        # TODO match fx rates with fx pivot in xlsx by sale ID
        # extra: use date column and currency code to get fx rates from fx_rates
 
        if not self.fx_rates:
            print("FX rates not set. Please set FX rates before conversion.")
            return
        print(self.fx_rates)
        # Apply FX conversion
        self.stocks_df['Price'] = self.stocks_df.apply(
            lambda row: row['Price'] * self.fx_rates.get(row['CurrencyCode'], 1), axis=1)
        self.sales_df['Price'] = self.sales_df.apply(
            lambda row: row['Price'] * self.fx_rates.get(row['CurrencyCode'], 1), axis=1)

        # Standardize currency code after conversion
        self.stocks_df['CurrencyCode'] = 'USD'
        self.sales_df['CurrencyCode'] = 'USD'

    def _init_approved_flow(self):
        approved_flows = self.approved_suppliers[['Company Name', 'Dairy Plus ', 'Meadows Inc', 'Advantage Plus',
                                                  "Milk R'Us    ", 'Cows Inc            ', 'Northern Pastures',]]
        # strip column names
        approved_flows.columns = approved_flows.columns.str.strip()
        self.approved_flow_dict = {row['Company Name']: row.drop(
            'Company Name').to_dict() for index, row in approved_flows.iterrows()}

    def prepare_for_optimization(self):
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
        self._convert_fx()

        self.sales_df['Region'] = pd.merge(self.sales_df, self.sales_data[[
                                           'ExternalNr', 'PlaceOfDelivery']], how='left', on='ExternalNr')['PlaceOfDelivery']
        print(self.sales_df.sort_values(by=['ExternalNr']))
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
        self.add_freights()
        self.prepare_inventory()
        self.prepare_sales()
        self.prepare_for_optimization()
        self.purchases = list(self.purchase_prices_dict.keys())

        self.sales = list(self.sale_prices_dict.keys())
        self.S = self.sale_prices_dict
        self.P = self.purchase_prices_dict
        self.C = self.freight_rates_dict
        self.Q = self.max_qty_p_dict
        self.D = self.demand_dict 

        self.purchase_to_seller = pd.Series(
            self.stocks_df.AlphaName.values, index=self.stocks_df.ExternalNr).to_dict()

        self.sale_to_buyer = pd.Series(self.sales_df.AlphaName.values,
                                       index=self.sales_df.ExternalNr).to_dict()

    def _get_approved_sellers(self, buyer):
        return [seller for seller, approved in self.approved_flow_dict[buyer].items() if approved == 'P']

    def setup_optimization(self):
        # unmet_demand_penalty = 100000
        print(len(self.purchases), len(self.sales))
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
            prob += pl.lpSum([X[(i, j)] for j in self.sales if (i, j) in X]) <= self.Q[i], f"Supply_{i}"
        # Demand Constraints: Each demand must be exactly met.
        for j in self.sales:
            prob += pl.lpSum([X[(i, j)] for i in self.purchases if (i, j) in X]) == self.D[j], f"Demand_{j}"
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

    def adjust_prices_with_forward_curves(self):
        
        period_to_date = {
            'C1': '2024-01', 'C2': '2024-02', 'C3': '2024-03', 'C4': '2024-04',
            'C5': '2024-05', 'C6': '2024-06', 'C7': '2024-07', 'C8': '2024-08',
            'C9': '2024-09', 'C10': '2024-10', 'C11': '2024-11', 'C12': '2024-12',
            'C13': '2025-01', 'C14': '2025-02', 'C15': '2025-03', 'C16': '2025-04',
            'C17': '2025-05', 'C18': '2025-06', 'C19': '2025-07', 'C20': '2025-08',
            'C21': '2025-09', 'C22': '2025-10', 'C23': '2025-11', 'C24': '2025-12'
        }
        for index, row in self.purchases_data.iterrows():
            period_date = period_to_date.get(row['Period'], None)
            if period_date:
                matching_curve = self.future_curves[
                    (self.future_curves['Product'] == row['Product']) &
                    (self.future_curves['Period'] == period_date)
                ]
                if not matching_curve.empty:
                    new_price = matching_curve.iloc[0]['Price']
                    self.purchases_data.at[index, 'Price'] = new_price

    def prepare_data_mtm(self):
        """
        Adjust purchase prices based on MTM forward curves for specific regions.
        """
        if self.forward_curves is not None:
            for index, row in self.purchases_data.iterrows():
            
                region = row['Region']
                date = row['Date']
                
                matching_curve = self.forward_curves[
                    (self.forward_curves['Region'] == region) & 
                    (self.forward_curves['Date'] <= date)
                ].sort_values(by='Date', ascending=False).head(1)
                if not matching_curve.empty:
                    # Assume 'Price' is the column in your forward curves containing the new price
                    new_price = matching_curve.iloc[0]['Price']
                    self.purchases_data.at[index, 'Price'] = new_price
        else:
            print("No forward curves data to adjust MTM prices.")

    
    def get_allocations_df(self):
        detailed_allocations = []
        unfulfilled_sales = []
        overfulfilled_sales = []
        # Augment dataframes with unique identifiers for each row
        stocks_augmented = self.stocks_df.reset_index().rename(columns={'index': 'StockUniqueID'})
        sales_augmented = self.sales_df.reset_index().rename(columns={'index': 'SaleUniqueID'})
        
        # Iterate through each decision variable in self.X
        for (purchase_id, sale_id), variable in self.X.items():
            quantity = variable.varValue  # Get the allocated quantity from the decision variable
            if quantity > 0:  # Process only allocations with a positive quantity
                purchase_details = stocks_augmented.loc[stocks_augmented['ExternalNr'] == purchase_id].iloc[0]
                sale_details = sales_augmented.loc[sales_augmented['ExternalNr'] == sale_id].iloc[0]
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
                    'SalePlaceOfDelivery': sale_details.get('Region', 'Unknown'),
                    'SaleQuantity': sale_details['TotalQuantity'],
                    'SaleWeightedAveragePrice': f"{sale_details['Price']:,.2f}",  
                    
                    'AllocatedQuantity': quantity
                })
        
        allocations_df = pd.DataFrame(detailed_allocations)
        grouped_allocations = allocations_df.groupby(
            'SaleID')['AllocatedQuantity'].sum()
        for sale_id, total_allocated_quantity in grouped_allocations.items():
            total_demand = allocations_df.loc[allocations_df['SaleID']
                                            == sale_id, 'SaleQuantity'].iloc[0]
            if total_allocated_quantity < total_demand:
                unfulfilled_sales.append({
                    'SaleID': sale_id,
                    'SaleAlphaName': allocations_df.loc[allocations_df['SaleID'] == sale_id, 'SaleAlphaName'].iloc[0],
                    'TotalDemand': total_demand,
                    'AllocatedQuantity': total_allocated_quantity
                })
            elif total_allocated_quantity > total_demand + 1:
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
