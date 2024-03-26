import pandas as pd
import pulp as pl
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go


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
        self.country_region_mappings = {}
        self.MTM_prices = False
        self.past_fx = False
        # if 'forward_curves_editor' not in st.session_state:
        #     st.session_state['forward_curves_editor'] = {}


    def load_data(self):
        """
        Load data from the data_path
        """
        self.purchases_data = pd.read_excel(
            self.data_path, sheet_name='Physical Purchases')
        self.sales_data = pd.read_excel(
            self.data_path, sheet_name='Physical Sales')
        st.write(self.sales_data['PlaceOfDelivery'].unique())
        self.inventory_data = pd.read_excel(
            self.data_path, sheet_name='Physical Inventory')
        self.forex_data = pd.read_excel(
            self.data_path, sheet_name='Physical&Forex Pivot', header=2).dropna()
        self.forward_curves = pd.read_excel(
            self.data_path, sheet_name='Forward Curves', header=2)

        # TODO integrate forward prices to create MTM prices for purchases and sales, based on region
        # create dictionary input for forward prices to be adjusted
        self.approved_suppliers = pd.read_excel(
            self.data_path, sheet_name='Approved Supplier List', header=1)
        self.freights = pd.read_excel(
            self.data_path, sheet_name='Freight Rates - Wacc - Storage ', header=1)
        self.freights_lookup = self.freights.copy()
        self.freights_lookup.set_index('PlaceOfDelivery', inplace=True)
        # st.write(self.forex_data)
        self.prepare_curves()

    def prepare_curves(self):
        def _calc_future_date(period_code):
            months_ahead = int(period_code[1:])
            now = datetime.now()
            future_date = (now + pd.DateOffset(months=months_ahead)).date()
            return future_date
        # Filter the forward curves for the relevant product
        # Assume SMP MH is the only product
        forward_curves_filtered = self.forward_curves.loc[self.forward_curves['Product'] == 'SMP MH']
        forward_curves_filtered['Input Adjustment']  = forward_curves_filtered['Price'] + forward_curves_filtered['Carry']
        forward_curves_filtered['Period'] = forward_curves_filtered['Period'].apply(_calc_future_date)
        forward_curves_filtered = forward_curves_filtered[['Period', 'Price', 'Carry', 'Input Adjustment', 'Market']]
        self.forward_curves_filtered = forward_curves_filtered.copy()
        # st.dataframe(self.forward_curves_filtered)
        self.sales_data['Date'] = pd.to_datetime(self.sales_data['Day of DateDeliverySchemeThru'], format='%Y/%m/%d')
        # st.dataframe(self.sales_data)


    def find_closest_forward_curve(self, period_date, market):
        # Filter the curves for the specified market first
        market_curves = self.forward_curves_filtered[self.forward_curves_filtered['Market'] == market].copy()

        # Then proceed with finding the closest curve as before
        if not market_curves.empty:
            market_curves['Period'] = pd.to_datetime(market_curves['Period'])
            market_curves['Date Difference'] = market_curves['Period'].apply(
                lambda x: abs((x - period_date).days))
            closest_curve = market_curves.sort_values(by='Date Difference').iloc[0]
            return closest_curve['Input Adjustment']
        else:
            return None  # Or some default action

    
    def set_fx_rates(self, fx_rates):
        if isinstance(fx_rates, dict):
            self.fx_rates = fx_rates
        else:
            raise ValueError("FX rates must be provided as a dictionary.")

    def add_freights_to_rates(self):
        new_rows = pd.DataFrame({'PlaceOfDelivery': [
                                'Australia'], 'Oceania': 40, 'North America': 140, 'W-EU': 75})
        self.freights = pd.concat([self.freights, new_rows], axis=0)


    
    def money_changer_row(self, row):
        row_externalNr_as_int = int(row['ExternalNr'])

        if self.past_fx:
            self.forex_data['ExternalNr'] = self.forex_data['ExternalNr'].astype(int)

        if row_externalNr_as_int in self.forex_data['ExternalNr'].values and self.past_fx == True:
            # print(row_externalNr_as_int)
            rate = float(self.forex_data.loc[self.forex_data['ExternalNr'] == row_externalNr_as_int, 'Ave Rate'].iloc[0])
            row['Price'] = row['Price'] * rate
            return row
        else:
            row['Price'] = row['Price'] * self.fx_rates.get(row['CurrencyCode'], 1)  # dictionary rate
            return row


    def process_data(self, data_frame, item_name='Skimmed Milk Powder'):
        # Ensure AlphaName and Date columns are correctly formatted
        filtered_data = data_frame[data_frame['ItemName'] == item_name].copy()
        filtered_data['AlphaName'] = filtered_data['AlphaName'].str.strip()
        filtered_data['Date'] = pd.to_datetime(filtered_data['Day of DateDeliverySchemeThru'], format='%Y-%m-%d')
        
        # Convert ExternalNr to int for consistency
        filtered_data['ExternalNr'] = filtered_data['ExternalNr'].astype(int)
        # print(filtered_data['ExternalNr'].values)

        filtered_data = filtered_data.apply(self.money_changer_row, axis=1)
        filtered_data['CurrencyCode'] = 'USD'
        
        filtered_data['pq'] = filtered_data['Price'] * filtered_data['Quantity (MT)']

        final_grouped = filtered_data.groupby(['ExternalNr', 'AlphaName', 'Region', 'DeliveryTermCode', 'ItemName', 'Date', 'CurrencyCode']).agg(
            TotalQuantity=('Quantity (MT)', 'sum'),
            WeightedPriceSum=('pq', 'sum')  
        ).reset_index()

        # Calculate weighted average price
        final_grouped['WeightedAveragePrice'] = final_grouped['WeightedPriceSum'] / final_grouped['TotalQuantity']
        final_grouped.drop(columns=['WeightedPriceSum'], inplace=True)
        
        current_date = datetime.now().date() # Ensure comparison is date to date, not datetime to date
        if self.MTM_prices:
            final_grouped['Region'] = final_grouped['Region'].apply(
                lambda x: self.country_region_mappings.get(x, x))
            
            final_grouped['AdjustedPrice'] = final_grouped.apply(
            lambda row: row['WeightedAveragePrice'] if row['Date'].date() <= current_date else self.find_closest_forward_curve(row['Date'], row['Region']),
            axis=1
        )
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
        self.inventory_data['AlphaName'] = self.inventory_data['AlphaName'].str.strip()
        self.inventory_data.loc[self.inventory_data['AlphaName'] == 'Northern Pastures', 'Region'] = 'Oceania'
        self.inventory_data.loc[self.inventory_data['AlphaName'] == 'Meadows Inc', 'Region'] = 'Asia'
        self.inventory_data['Region'].replace(['New Zealand', 'Australia'], 'Oceania', inplace=True)
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
        self.sales_df['PlaceOfDelivery'] = pd.merge(self.sales_df, self.sales_data[['ExternalNr', 'PlaceOfDelivery']], how='left', on='ExternalNr')['PlaceOfDelivery']
        self.sales_df.replace('Dalian', 'China', inplace=True)
        self.sales_df.replace('Melbourne', 'Australia', inplace=True)
        self.stocks_df.replace('Western Europe', 'W-EU', inplace=True)
        self.stocks_df.replace('Asia', 'Oceania', inplace=True)
        # import streamlit as st
        # st.write('\n\n', self.stocks_df.Region.unique(), self.sales_df.Region.unique())

    def _get_freight(self, origin, destination):
        if destination in self.freights_lookup.index and origin in self.freights_lookup.columns:
            return self.freights_lookup.loc[destination, origin]
        elif destination == origin:
            # print(origin, destination)
            return 40
        else:
            # print(origin, destination)
            return 0  # Return 0 or some default value if no match is found

    def _create_freight_rates_dict(self):
        self.freight_rates_dict = {}
        for _, sale_row in self.sales_df.iterrows():
            for _, stock_row in self.stocks_df.iterrows():
                origin = stock_row['Region']
                destination = sale_row['PlaceOfDelivery']
                pair_key = (stock_row['ExternalNr'], sale_row['ExternalNr'])
                self.freight_rates_dict[pair_key] = self._get_freight(
                    origin, destination)
        st.write(self.sales_df['PlaceOfDelivery'])
        st.write({str(k): v for k, v in self.freight_rates_dict.items()})
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
        if self.MTM_prices:
            self.stocks_df['Price'] = pd.to_numeric(
                self.stocks_df['AdjustedPrice'])
            self.sales_df['Price'] = pd.to_numeric(
                self.sales_df['AdjustedPrice'])
        # self._convert_fx()

        # print(self.sales_df.sort_values(by=['ExternalNr']))
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
        self.add_freights_to_rates()
        self.prepare_inventory()
        self.prepare_sales()
        self.prepare_for_optimization()
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


    def get_allocations_df(self):
        self.stocks_df['Region'] = self.stocks_df['Region'].apply(lambda x: self.country_region_mappings.get(x, x))
        # st.write(self.stocks_df, self.sales_df)
        # st.write(self.stocks_df['Region'].unique())
        # st.write(self.stocks_df[self.stocks_df['Region'] == 'All countries'])
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
                    'SalePlaceOfDelivery': sale_details.get('PlaceOfDelivery', 'Unknown'),
                    'SaleQuantity': sale_details['TotalQuantity'],
                    'SaleWeightedAveragePrice': f"{sale_details['Price']:,.2f}",  
                    
                    'AllocatedQuantity': quantity
                })
        
        allocations_df = pd.DataFrame(detailed_allocations)
        self.allocations_df = allocations_df
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
    
    def plot_sankey(self, source_column, target_column, title='Sankey Diagram'):
        df = self.allocations_df

        # Collect unique entities for both source and target
        source_entities = df[source_column].unique().tolist()
        target_entities = df[target_column].unique().tolist()

        # Ensure unique labels across sources and targets by prefixing
        source_labels = [f"{ent} Purchase" for ent in source_entities]
        target_labels = [f"{ent} Sale" for ent in target_entities]

        # Combine into a single list of labels for the diagram
        labels = source_labels + target_labels

        # Create a mapping from entity names to their indices in the labels list
        entity_to_index = {name: idx for idx, name in enumerate(labels)}

        # Generate sources and targets based on indices
        sources = df[source_column].map(lambda x: entity_to_index[f"{x} Purchase"]).tolist()
        targets = df[target_column].map(lambda x: entity_to_index[f"{x} Sale"]).tolist()

        # Values for the flows
        values = df['AllocatedQuantity'].tolist()

        # Create and display the Sankey diagram
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
            ))])

        fig.update_layout(title_text=title, font_size=10)
        st.plotly_chart(fig)

