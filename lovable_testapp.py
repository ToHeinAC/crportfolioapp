import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
import json
import numpy as np
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Crypto Portfolio Tracker",
    page_icon="💰",
    layout="wide"
)

# Define category colors
category_colors = {
    'Layer 1': '#FF9500',
    'Utility': '#34C759',
    'Oracle': '#5856D6',
    'DeFi': '#007AFF',
    'Gaming': '#FF2D55',
    'Meme': '#AF52DE',
    'Metaverse': '#5AC8FA',
    'KI': '#FFCC00',
    'RWA': '#FF3B30',
    'Digital Gold': '#FFD700'
}

# Mapping of crypto symbols to CoinGecko IDs
crypto_mapping = {
    'SOL': 'solana',
    'ATOM': 'cosmos',
    'LINK': 'chainlink',
    'ONT': 'ontology',
    'AAVE': 'aave',
    'ICP': 'internet-computer',
    'RAY': 'raydium',
    'VOXEL': 'voxies',
    'BOME': 'book-of-meme',
    'VANRY': 'vanry',
    'AGLD': 'adventure-gold',
    'SUPER': 'superfarm',
    'PHB': 'phoenix-global',
    'PRIME': 'echelon-prime',
    'TIA': 'celestia',
    'INJ': 'injective-protocol',
    'MDT': 'measurable-data-token',
    'MPL': 'maple',
    'AKT': 'akash-network',
    'SUI': 'sui',
    'APT': 'aptos',
    'JTO': 'jito-governance',
    'TURBO': 'turbos-finance',
    'FET': 'fetch-ai',
    'ONDO': 'ondo-finance',
    'BTC': 'bitcoin'
}

# Header
st.title("Cryptocurrency Portfolio Tracker")
st.write("Upload your portfolio data and analyze your crypto investments")

# Create a sample dataframe for download example
sample_df = pd.DataFrame({
    'Name': ['BTC', 'ETH', 'SOL'],
    'Anzahl': [0.1, 1.5, 10],
    'Kaufpreis $': [45000, 3000, 100],
    'Kategorie': ['Digital Gold', 'Layer 1', 'Layer 1']
})

# Sidebar with instructions and sample file download
with st.sidebar:
    st.header("Instructions")
    st.write("1. Upload an Excel file with your crypto portfolio data")
    st.write("2. The file should have columns: 'Name', 'Anzahl', 'Kaufpreis $', and 'Kategorie'")
    st.write("3. Click 'Process Portfolio' to analyze your investments")
    
    # Create the sample file for download
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        sample_df.to_excel(writer, sheet_name='Portfolio', index=False)
    buffer.seek(0)
    
    st.download_button(
        label="Download Sample Excel Template",
        data=buffer,
        file_name="crypto_portfolio_template.xlsx",
        mime="application/vnd.ms-excel"
    )
    
    st.markdown("---")
    st.write("Made with ❤️ using Streamlit and Plotly")

# File uploader
uploaded_file = st.file_uploader("Upload your portfolio Excel file", type=["xlsx"])

# Initialize session state for portfolio data
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
    st.session_state.processed_portfolio = None
    st.session_state.selected_asset = None

# Process the uploaded file
if uploaded_file is not None:
    try:
        portfolio_data = pd.read_excel(uploaded_file)
        required_columns = ['Name', 'Anzahl', 'Kaufpreis $', 'Kategorie']
        
        # Check if all required columns exist
        if not all(col in portfolio_data.columns for col in required_columns):
            st.error(f"Missing required columns. Please ensure your file has these columns: {', '.join(required_columns)}")
        else:
            # Convert data types
            portfolio_data['Anzahl'] = pd.to_numeric(portfolio_data['Anzahl'], errors='coerce')
            portfolio_data['Kaufpreis $'] = pd.to_numeric(portfolio_data['Kaufpreis $'], errors='coerce')
            
            # Check for nulls after conversion
            if portfolio_data['Anzahl'].isnull().any() or portfolio_data['Kaufpreis $'].isnull().any():
                st.warning("Some values couldn't be converted to numbers. Please check your data.")
            
            # Store in session state
            st.session_state.portfolio_data = portfolio_data
            st.success(f"Successfully loaded {len(portfolio_data)} assets from your portfolio file!")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Process portfolio button
if st.button("Process Portfolio") and st.session_state.portfolio_data is not None:
    with st.spinner("Fetching current prices and processing your portfolio..."):
        try:
            # Get list of crypto IDs to fetch
            portfolio_data = st.session_state.portfolio_data
            crypto_ids = [crypto_mapping.get(name, '') for name in portfolio_data['Name']]
            valid_ids = [id for id in crypto_ids if id]
            
            if not valid_ids:
                st.error("Could not map any of your crypto symbols to CoinGecko IDs")
            else:
                # Fetch current prices from CoinGecko API
                ids_str = ','.join(valid_ids)
                response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={ids_str}&vs_currencies=usd")
                if response.status_code != 200:
                    st.error(f"Error fetching prices: {response.status_code}")
                    st.session_state.processed_portfolio = None
                else:
                    prices = response.json()
                    
                    # Process the portfolio with current prices
                    processed_data = []
                    for _, row in portfolio_data.iterrows():
                        name = row['Name']
                        coingecko_id = crypto_mapping.get(name, '')
                        amount = row['Anzahl']
                        purchase_price = row['Kaufpreis $']
                        category = row['Kategorie']
                        
                        current_price = 0
                        if coingecko_id and coingecko_id in prices:
                            current_price = prices[coingecko_id]['usd']
                        
                        initial_investment = amount * purchase_price
                        current_value = amount * current_price
                        profit_loss = current_value - initial_investment
                        profit_loss_percentage = (profit_loss / initial_investment * 100) if initial_investment > 0 else 0
                        
                        processed_data.append({
                            'name': name,
                            'amount': amount,
                            'purchase_price': purchase_price,
                            'category': category,
                            'current_price': current_price,
                            'initial_investment': initial_investment,
                            'current_value': current_value,
                            'profit_loss': profit_loss,
                            'profit_loss_percentage': profit_loss_percentage
                        })
                    
                    # Convert to DataFrame and store in session state
                    st.session_state.processed_portfolio = pd.DataFrame(processed_data)
                    
                    # Set first asset as selected by default
                    if len(processed_data) > 0 and st.session_state.selected_asset is None:
                        st.session_state.selected_asset = processed_data[0]['name']
                    
                    st.success("Portfolio processed successfully!")
        except Exception as e:
            st.error(f"Error processing portfolio: {str(e)}")
            st.session_state.processed_portfolio = None

# Display portfolio analysis if processed
if st.session_state.processed_portfolio is not None:
    portfolio_df = st.session_state.processed_portfolio
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Portfolio Analysis", "Individual Assets", "Categories"])
    
    with tab1:  # Overview Tab
        # Summary metrics
        total_investment = portfolio_df['initial_investment'].sum()
        total_current_value = portfolio_df['current_value'].sum()
        total_profit_loss = total_current_value - total_investment
        total_profit_loss_percentage = (total_profit_loss / total_investment * 100) if total_investment > 0 else 0
        
        # Create layout for metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Investment", f"${total_investment:,.2f}")
        with col2:
            st.metric("Current Value", f"${total_current_value:,.2f}")
        with col3:
            st.metric("Profit/Loss", f"${total_profit_loss:,.2f}", f"{total_profit_loss_percentage:.2f}%")
        with col4:
            st.metric("Number of Assets", f"{len(portfolio_df)}")
        
        # Portfolio allocation charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Allocation by Asset")
            asset_allocation = portfolio_df[['name', 'current_value']].sort_values('current_value', ascending=False)
            fig = px.pie(
                asset_allocation,
                values='current_value',
                names='name',
                title='Current Value Distribution',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Portfolio Allocation by Category")
            category_allocation = portfolio_df.groupby('category')['current_value'].sum().reset_index()
            fig = px.pie(
                category_allocation,
                values='current_value',
                names='category',
                title='Category Distribution',
                hole=0.4,
                color='category',
                color_discrete_map={cat: color for cat, color in category_colors.items() if cat in category_allocation['category'].values}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio details table
        st.subheader("Portfolio Details")
        
        # Format the table with styles
        def color_profit_loss(val):
            color = 'green' if val >= 0 else 'red'
            return f'color: {color}'
        
        # Create a styled dataframe for display
        display_df = portfolio_df.copy()
        display_df['profit_loss_color'] = display_df['profit_loss'].apply(lambda x: 'green' if x >= 0 else 'red')
        display_df.sort_values('current_value', ascending=False, inplace=True)
        
        # Format for display
        formatted_df = pd.DataFrame({
            'Symbol': display_df['name'],
            'Category': display_df['category'],
            'Amount': display_df['amount'].map('{:,.4f}'.format),
            'Purchase Price': display_df['purchase_price'].map('${:,.4f}'.format),
            'Current Price': display_df['current_price'].map('${:,.4f}'.format),
            'Initial Investment': display_df['initial_investment'].map('${:,.2f}'.format),
            'Current Value': display_df['current_value'].map('${:,.2f}'.format),
            'Profit/Loss': display_df['profit_loss'].map('${:,.2f}'.format),
            'P/L %': display_df['profit_loss_percentage'].map('{:,.2f}%'.format)
        })
        
        # Apply styling
        st.dataframe(
            formatted_df.style.apply(lambda x: ['color: green' if val >= 0 else 'color: red' 
                                       for val in display_df['profit_loss']], 
                           axis=0, subset=['Profit/Loss'])
                     .apply(lambda x: ['color: green' if val >= 0 else 'color: red' 
                                       for val in display_df['profit_loss_percentage']], 
                           axis=0, subset=['P/L %'])
        )
    
    with tab2:  # Portfolio Analysis Tab
        # Top and worst performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Performers")
            top_performers = portfolio_df.sort_values('profit_loss_percentage', ascending=False).head(5)
            
            for _, asset in top_performers.iterrows():
                with st.container():
                    cols = st.columns([3, 2])
                    with cols[0]:
                        st.write(f"**{asset['name']}** ({asset['category']})")
                    with cols[1]:
                        st.write(f"**+{asset['profit_loss_percentage']:.2f}%**")
                        st.write(f"${asset['current_value']:.2f}")
                st.divider()
        
        with col2:
            st.subheader("Worst Performers")
            worst_performers = portfolio_df.sort_values('profit_loss_percentage').head(5)
            
            for _, asset in worst_performers.iterrows():
                with st.container():
                    cols = st.columns([3, 2])
                    with cols[0]:
                        st.write(f"**{asset['name']}** ({asset['category']})")
                    with cols[1]:
                        st.write(f"**{asset['profit_loss_percentage']:.2f}%**")
                        st.write(f"${asset['current_value']:.2f}")
                st.divider()
        
        # Largest holdings
        st.subheader("Largest Holdings by Value")
        largest_holdings = portfolio_df.sort_values('current_value', ascending=False).head(10)
        fig = px.bar(
            largest_holdings,
            x='name',
            y='current_value',
            title='Top 10 Holdings by Current Value',
            labels={'name': 'Asset', 'current_value': 'Current Value ($)'},
            color='name'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Initial vs Current Value
        st.subheader("Initial Investment vs Current Value")
        comparison_df = portfolio_df.sort_values('current_value', ascending=False).head(10)
        comparison_df = comparison_df.melt(
            id_vars=['name'],
            value_vars=['initial_investment', 'current_value'],
            var_name='type',
            value_name='value'
        )
        fig = px.bar(
            comparison_df,
            x='name',
            y='value',
            color='type',
            barmode='group',
            title='Initial Investment vs Current Value (Top 10 Assets)',
            labels={'name': 'Asset', 'value': 'Value ($)', 'type': 'Type'},
            color_discrete_map={'initial_investment': '#0088FE', 'current_value': '#00C49F'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:  # Individual Assets Tab
        # Asset selector
        assets = portfolio_df['name'].tolist()
        selected_asset = st.selectbox("Select an asset to analyze", assets, index=assets.index(st.session_state.selected_asset) if st.session_state.selected_asset in assets else 0)
        st.session_state.selected_asset = selected_asset
        
        # Get the selected asset data
        asset_data = portfolio_df[portfolio_df['name'] == selected_asset].iloc[0]
        
        # Display asset metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Current Price", 
                f"${asset_data['current_price']:,.4f}",
                f"{((asset_data['current_price'] / asset_data['purchase_price']) - 1) * 100:.2f}% from purchase"
            )
        with col2:
            st.metric(
                "Holdings", 
                f"{asset_data['amount']:,.4f} {asset_data['name']}",
                f"Worth ${asset_data['current_value']:,.2f}"
            )
        with col3:
            st.metric(
                "Profit/Loss", 
                f"${asset_data['profit_loss']:,.2f}",
                f"{asset_data['profit_loss_percentage']:.2f}%"
            )
        
        # Try to fetch historical price data
        try:
            st.subheader("Price History (30 Days)")
            
            coingecko_id = crypto_mapping.get(selected_asset, '')
            if coingecko_id:
                hist_response = requests.get(
                    f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart?vs_currency=usd&days=30"
                )
                
                if hist_response.status_code == 200:
                    hist_data = hist_response.json()
                    price_data = [(datetime.fromtimestamp(timestamp/1000), price) 
                                  for timestamp, price in hist_data['prices']]
                    
                    df_hist = pd.DataFrame(price_data, columns=['date', 'price'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_hist['date'], 
                        y=df_hist['price'],
                        mode='lines',
                        name='Price (USD)'
                    ))
                    
                    # Add a horizontal line for purchase price
                    fig.add_shape(
                        type="line",
                        x0=df_hist['date'].min(),
                        x1=df_hist['date'].max(),
                        y0=asset_data['purchase_price'],
                        y1=asset_data['purchase_price'],
                        line=dict(color="red", width=2, dash="dash"),
                    )
                    
                    # Add annotation for purchase price
                    fig.add_annotation(
                        x=df_hist['date'].min(),
                        y=asset_data['purchase_price'],
                        text="Purchase Price",
                        showarrow=False,
                        yshift=10,
                        bgcolor="rgba(255, 255, 255, 0.8)"
                    )
                    
                    fig.update_layout(
                        title=f"{selected_asset} Price History (30 Days)",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Could not fetch historical data: {hist_response.status_code}")
            else:
                st.warning(f"No CoinGecko ID mapping found for {selected_asset}")
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
        
        # Investment details and performance analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Investment Details")
            details = [
                {"label": "Purchase Price", "value": f"${asset_data['purchase_price']:,.4f}"},
                {"label": "Amount", "value": f"{asset_data['amount']:,.4f}"},
                {"label": "Initial Investment", "value": f"${asset_data['initial_investment']:,.2f}"},
                {"label": "Category", "value": asset_data['category']}
            ]
            
            for detail in details:
                col_a, col_b = st.columns([1, 1])
                col_a.write(detail["label"])
                col_b.write(detail["value"])
                st.divider()
        
        with col2:
            st.subheader("Performance Analysis")
            performance = [
                {"label": "Current Value", "value": f"${asset_data['current_value']:,.2f}"},
                {"label": "Profit/Loss", "value": f"${asset_data['profit_loss']:,.2f}", "color": "green" if asset_data['profit_loss'] >= 0 else "red"},
                {"label": "Profit/Loss %", "value": f"{asset_data['profit_loss_percentage']:,.2f}%", "color": "green" if asset_data['profit_loss_percentage'] >= 0 else "red"},
                {"label": "Price Change %", "value": f"{((asset_data['current_price'] / asset_data['purchase_price']) - 1) * 100:,.2f}%", "color": "green" if asset_data['current_price'] >= asset_data['purchase_price'] else "red"}
            ]
            
            for perf in performance:
                col_a, col_b = st.columns([1, 1])
                col_a.write(perf["label"])
                if "color" in perf:
                    col_b.markdown(f"<span style='color:{perf['color']}'>{perf['value']}</span>", unsafe_allow_html=True)
                else:
                    col_b.write(perf["value"])
                st.divider()
        
        # Price Scenarios
        st.subheader("Price Scenarios")
        
        scenarios = [
            {"name": "-50%", "price_factor": 0.5},
            {"name": "-25%", "price_factor": 0.75},
            {"name": "Current", "price_factor": 1.0},
            {"name": "+25%", "price_factor": 1.25},
            {"name": "+50%", "price_factor": 1.5},
            {"name": "+100%", "price_factor": 2.0},
            {"name": "+200%", "price_factor": 3.0}
        ]
        
        scenario_data = []
        for scenario in scenarios:
            price = asset_data['current_price'] * scenario['price_factor']
            value = price * asset_data['amount']
            profit_loss = value - asset_data['initial_investment']
            profit_loss_percentage = (profit_loss / asset_data['initial_investment'] * 100) if asset_data['initial_investment'] > 0 else 0
            
            scenario_data.append({
                "name": scenario['name'],
                "price": price,
                "value": value,
                "profit_loss": profit_loss,
                "profit_loss_percentage": profit_loss_percentage
            })
        
        scenario_df = pd.DataFrame(scenario_data)
        
        # Plot scenarios
        fig = px.bar(
            scenario_df,
            x='name',
            y='profit_loss',
            title='Profit/Loss in Different Price Scenarios',
            labels={'name': 'Scenario', 'profit_loss': 'Profit/Loss ($)'},
            color='profit_loss',
            color_continuous_scale=['red', 'green'],
            color_continuous_midpoint=0
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario table
        st.subheader("Scenario Details")
        scenario_display = pd.DataFrame({
            'Scenario': scenario_df['name'],
            'Price': scenario_df['price'].map('${:,.4f}'.format),
            'Value': scenario_df['value'].map('${:,.2f}'.format),
            'Profit/Loss': scenario_df['profit_loss'].map('${:,.2f}'.format),
            'P/L %': scenario_df['profit_loss_percentage'].map('{:,.2f}%'.format)
        })
        
        # Apply styling to profit/loss columns
        st.dataframe(
            scenario_display.style.apply(lambda x: ['color: green' if val >= 0 else 'color: red' 
                                       for val in scenario_df['profit_loss']], 
                           axis=0, subset=['Profit/Loss'])
                     .apply(lambda x: ['color: green' if val >= 0 else 'color: red' 
                                       for val in scenario_df['profit_loss_percentage']], 
                           axis=0, subset=['P/L %'])
        )
    
    with tab4:  # Categories Tab
        # Get category summary
        category_summary = portfolio_df.groupby('category').agg({
            'initial_investment': 'sum',
            'current_value': 'sum'
        }).reset_index()
        
        category_summary['profit_loss'] = category_summary['current_value'] - category_summary['initial_investment']
        category_summary['profit_loss_percentage'] = (category_summary['profit_loss'] / category_summary['initial_investment'] * 100)
        
        # Category performance chart
        st.subheader("Category Performance")
        
        fig = px.bar(
            category_summary,
            x='category',
            y=['initial_investment', 'current_value'],
            title='Initial Investment vs Current Value by Category',
            barmode='group',
            labels={
                'category': 'Category',
                'value': 'Value ($)',
                'variable': 'Type'
            },
            color_discrete_map={
                'initial_investment': '#0088FE',
                'current_value': '#00C49F'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Profit/Loss by category
        st.subheader("Profit/Loss by Category")
        
        fig = px.bar(
            category_summary,
            x='category',
            y='profit_loss',
            title='Profit/Loss by Category',
            labels={
                'category': 'Category',
                'profit_loss': 'Profit/Loss ($)'
            },
            color='profit_loss',
            color_continuous_scale=['red', 'green'],
            color_continuous_midpoint=0
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual category details
        st.subheader("Category Details")
        
        for _, category_row in category_summary.iterrows():
            category_name = category_row['category']
            category_assets = portfolio_df[portfolio_df['category'] == category_name]
            
            # Create an expander for each category
            with st.expander(f"{category_name} - ${category_row['current_value']:,.2f} ({category_row['profit_loss_percentage']:+.2f}%)"):
                # Display category metrics
                profit_loss_color = "green" if category_row['profit_loss'] >= 0 else "red"
                st.markdown(f"**Profit/Loss:** <span style='color:{profit_loss_color}'>${category_row['profit_loss']:,.2f} ({category_row['profit_loss_percentage']:+.2f}%)</span>", unsafe_allow_html=True)
                
                # Display assets in this category
                st.subheader("Assets in this category")
                
                # Create columns for asset cards
                cols = st.columns(3)
                
                # Sort assets by current value
                category_assets = category_assets.sort_values('current_value', ascending=False)
                
                # Display asset cards
                for i, (_, asset) in enumerate(category_assets.iterrows()):
                    col_index = i % 3
                    with cols[col_index]:
                        with st.container(border=True):
                            st.markdown(f"**{asset['name']}**")
                            st.write(f"${asset['current_value']:,.2f}")
                            st.write(f"{asset['amount']:,.4f} @ ${asset['current_price']:,.4f}")
                            profit_loss_color = "green" if asset['profit_loss_percentage'] >= 0 else "red"
                            st.markdown(f"<span style='color:{profit_loss_color}'>{asset['profit_loss_percentage']:+.2f}%</span>", unsafe_allow_html=True)
