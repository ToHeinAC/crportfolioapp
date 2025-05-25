import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sqlalchemy import create_engine
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import yfinance as yf
import appdirs as ad
import requests
from pycoingecko import CoinGeckoAPI
#from pandas_datareader import data as pdr
#yf.pdr_override() # <== that's all it takes :-)

binance=False
if binance:
    from binance.client import Client
    client = Client()

ad.user_cache_dir = lambda *args: "/tmp"
#assets = pd.read_excel('crassets.xlsx',index_col=0)

#start app
st.title('Crypto Dashboard :rocket:')

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Portfolio","Asset Cats Analysis","OHCL Single Asset"]
    )

@st.cache_data(show_spinner=False, ttl = 3600)
def load_assets(file):
    assets = pd.read_excel(file,index_col=0)
    return assets
    
with st.expander("Portfolio Data File"):
    upload = st.file_uploader('Upload the portfolio assets .xlsx file!')
    if upload is None:
        st.info("Upload a assets .xlsx file", icon = 'ℹ️')
        st.stop()

    if upload is not None:
        st.success('File uploaded successfully!')
        
    assets = load_assets(upload)    
    
coins = [str(assets.index[i]) for i in range(len(assets.index))]
pairs = [coins[i] + 'USDT' for i in range(len(coins))]
pairs2 = [coins[i] + '-USD' for i in range(len(coins))]

def replace_entry_by_name(lst, name_to_replace, new_entry):
    for i, entry in enumerate(lst):
        if entry == name_to_replace:
            lst[i] = new_entry
    return lst
    
pairs2 = replace_entry_by_name(pairs2, 'SUPER-USD', 'SUPER8290-USD')
pairs2 = replace_entry_by_name(pairs2, 'PRIME-USD', 'PRIME23711-USD')
pairs2 = replace_entry_by_name(pairs2, 'TIA-USD', 'TIA22861-USD')
pairs2 = replace_entry_by_name(pairs2, 'FORT-USD', 'FORT20622-USD')
pairs2 = replace_entry_by_name(pairs2, 'JUP-USD', 'JUP29210-USD')
pairs2 = replace_entry_by_name(pairs2, 'SUI-USD', 'SUI20947-USD')
pairs2 = replace_entry_by_name(pairs2, 'APT-USD', 'APT21794-USD')
pairs2 = replace_entry_by_name(pairs2, 'BANANA-USD', 'BANANA28066-USD')

assets['Invest $']=assets['Anzahl']*assets['Kaufpreis $']

@st.cache_data(show_spinner=False, ttl = 3600)
def getdata(pair, lookback='365'):
    #print(pair)
    frame = pd.DataFrame(client.get_historical_klines(pair, '1d', lookback + ' days ago UTC'))
    frame = frame.iloc[:,:]
    frame.columns = ['Time','Open','High','Low','Close','Volume', \
                     'CloseTime','QuoteAssetVolume','Trades','TakerBaseAssetVolume','takerQuoteAssetVolume','Ignored']
    frame[['Open','High','Low','Close','Volume', \
           'QuoteAssetVolume','Trades','TakerBaseAssetVolume','takerQuoteAssetVolume']] \
    = frame[['Open','High','Low','Close','Volume', \
           'QuoteAssetVolume','Trades','TakerBaseAssetVolume','takerQuoteAssetVolume']].astype(float)
    frame.Time = pd.to_datetime(frame.Time, unit='ms')
    if pair=='TVKUSDT':
        pair2 = 'VANRYUSDT'
        frame2 = pd.DataFrame(client.get_historical_klines(pair2, '1d', lookback + ' days ago UTC'))
        frame2 = frame2.iloc[:,:]
        frame2.columns = ['Time','Open','High','Low','Close','Volume', \
                         'CloseTime','QuoteAssetVolume','Trades','TakerBaseAssetVolume','takerQuoteAssetVolume','Ignored']
        frame2[['Open','High','Low','Close','Volume', \
               'QuoteAssetVolume','Trades','TakerBaseAssetVolume','takerQuoteAssetVolume']] \
        = frame2[['Open','High','Low','Close','Volume', \
               'QuoteAssetVolume','Trades','TakerBaseAssetVolume','takerQuoteAssetVolume']].astype(float)
        frame2.Time = pd.to_datetime(frame2.Time, unit='ms')
        frameges = pd.concat([frame,frame2])
        return frameges
    else:
        return frame

#sql
sql = False
if sql:
    engine = create_engine('sqlite:///CryproPrices1d_all.db')    
    def sql_importer(data,table_in_sql,engine):
        #data must have index as date
        max_date = pd.read_sql(f'SELECT MAX(Time) FROM "{table_in_sql}"',engine).values[0][0]
        new_rows = data[data.index > max_date]
        new_rows.to_sql(table_in_sql,engine,if_exists='append')
        #print(f'{len(new_rows)} new rows imported to {table_in_sql}')
        
    fetch=0

    if fetch:
        for item in pairs:
            #print(coin)
            getdata(coin, '3650').to_sql(item,engine,index=False,if_exists='replace')
    else:
        for item in pairs:
            #print(coin)
            sql_importer(getdata(item, '3650').set_index('Time'),item,engine)

    conn = st.connection('CryproPrices1d_all', type='sql')
    btc_prices = conn.query(f'SELECT * from {pairs[0]}')
    st.dataframe(btc_prices)

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
    'VANRY': 'vanar-chain',
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
    'JTO': 'jito',
    'TURBO': 'turbos-finance',
    'FET': 'fetch-ai',
    'ONDO': 'ondo-finance',
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'USDT': 'tether',
    'USDC': 'usd-coin',
    'BNB': 'binancecoin',
    'XRP': 'ripple',
    'ADA': 'cardano',
    'DOGE': 'dogecoin',
    'MATIC': 'matic-network',
    'DOT': 'polkadot',
    'AVAX': 'avalanche-2',
    'SHIB': 'shiba-inu',
    'LTC': 'litecoin',
    'UNI': 'uniswap',
    'XLM': 'stellar',
    'XMR': 'monero',
    'BCH': 'bitcoin-cash',
    'ALGO': 'algorand',
    'NEAR': 'near',
    'RENDER': 'render-token'
}

#app
@st.cache_data(show_spinner=False, ttl = 3600)
def get_data2(pairs, period='max'):
    data = []
    progress_text = "Running...(fetch the data)"
    my_bar = st.progress(0.0, text=progress_text)
    len_pairs = len(pairs)
    percent_complete = 0.0
    my_bar.progress(percent_complete, text=progress_text)
    
    # Initialize CoinGecko API client
    cg = CoinGeckoAPI()
    
    # Debug information
    debug_info = []
    
    for item in pairs:
        # Debug information for this item
        item_debug = {"symbol": item, "yahoo_success": False, "coingecko_success": False}
        
        # Try Yahoo Finance first
        try:
            # Add a small delay between requests to avoid rate limiting
            import time
            import random
            time.sleep(random.uniform(1, 3))  #  delay between requests
            
            # Implement retry logic with exponential backoff for rate limiting
            max_retries = 3
            retry_delay = 2  # Start with 2 seconds
            
            for retry in range(max_retries):
                try:
                    if item == 'RENDER-USD': 
                        fetch = yf.download(item, period='5d', progress=False)  # Disable progress to reduce output noise
                        if len(fetch) <= 1:
                            fetch = extend_dataframe_with_same_dates(fetch)
                    else:
                        fetch = yf.download(item, period=period, progress=False)  # Disable progress to reduce output noise
                    break  # If successful, break out of retry loop
                except Exception as retry_e:
                    if 'Rate limit' in str(retry_e) and retry < max_retries - 1:
                        # Add jitter to avoid synchronized retries
                        jitter = random.uniform(0, 1)
                        wait_time = retry_delay * (2 ** retry) + jitter
                        st.warning(f"Rate limit hit for {item}, retrying in {wait_time:.2f} seconds (attempt {retry+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        # Re-raise if it's not a rate limit error or we've exhausted retries
                        raise
            
            # Check if Yahoo Finance returned valid data
            if len(fetch) > 1 and not fetch.empty and not fetch['Close'].isnull().all().item():
                item_debug["yahoo_success"] = True
                item_debug["data_source"] = "Yahoo Finance"
                item_debug["data_points"] = len(fetch)
                debug_info.append(item_debug)
                data.append(fetch)
                st.success(f"Successfully fetched data for {item} from Yahoo Finance")
                
                # Update progress bar
                percent_complete += 1.0/len_pairs
                if percent_complete > 1.0:
                    percent_complete = 1.0            
                my_bar.progress(percent_complete, text=progress_text+f'...{round(percent_complete*100,1)}%')
                continue  # Skip to next item if Yahoo Finance data is valid
        except Exception as e:
            st.warning(f"Error fetching data from Yahoo Finance for {item}: {str(e)}")
            item_debug["yahoo_error"] = str(e)
        
        # If Yahoo Finance failed, try CoinGecko
        st.info(f"Yahoo Finance data unavailable for {item}, trying CoinGecko...")
        
        # Extract symbol from the pair (remove -USD suffix)
        symbol = item.split('-')[0]
        
        # Clean the symbol by removing any numbers
        # This ensures that symbols like "USDC2" are treated as "USDC" for CoinGecko lookup
        clean_symbol = ''.join([char for char in symbol if not char.isdigit()])
        
        # Get CoinGecko ID for the symbol
        coin_id = crypto_mapping.get(clean_symbol)
        
        if coin_id:
            st.info(f"Using cleaned symbol '{clean_symbol}' for CoinGecko lookup (original: '{symbol}')")
            item_debug["coingecko_symbol"] = clean_symbol
            item_debug["coingecko_id"] = coin_id
        elif not coin_id:
            # Try the original symbol as fallback
            coin_id = crypto_mapping.get(symbol)
            if coin_id:
                st.info(f"Using original symbol '{symbol}' for CoinGecko lookup")
                item_debug["coingecko_symbol"] = symbol
                item_debug["coingecko_id"] = coin_id
        
        if coin_id:
            try:
                # Always use 365 days for CoinGecko API to ensure we don't exceed limits
                # This is to prevent potential API rate limiting issues and to ensure consistent behavior
                # The 'max' period can sometimes cause timeouts or errors with the CoinGecko API
                days = '365'
                
                # Get market data from CoinGecko
                market_data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
                
                # Extract price data
                prices = market_data['prices']
                
                # Create DataFrame
                price_df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='ms')
                price_df.set_index('timestamp', inplace=True)
                
                # Rename columns to match Yahoo Finance format
                price_df.rename(columns={'price': 'Close'}, inplace=True)
                
                # Add other required columns (approximations)
                price_df['Open'] = price_df['Close'].shift(1)
                price_df['High'] = price_df['Close'] * 1.005  # Approximate
                price_df['Low'] = price_df['Close'] * 0.995   # Approximate
                price_df['Adj Close'] = price_df['Close']
                price_df['Volume'] = 0  # No volume data from this endpoint
                
                # Fill NaN values in the first row
                if pd.isna(price_df['Open'].iloc[0]):
                    # Use loc instead of iloc on a slice to avoid SettingWithCopyWarning
                    price_df.loc[price_df.index[0], 'Open'] = price_df['Close'].iloc[0]
                
                # Check if CoinGecko returned valid data
                if len(price_df) > 1 and not price_df.empty and not price_df['Close'].isnull().all().item():
                    fetch = price_df
                    item_debug["coingecko_success"] = True
                    item_debug["data_source"] = "CoinGecko"
                    item_debug["data_points"] = len(fetch)
                    debug_info.append(item_debug)
                    st.success(f"Successfully fetched data for {item} from CoinGecko using symbol {item_debug.get('coingecko_symbol', symbol)}")
                else:
                    st.error(f"CoinGecko returned empty or invalid data for {item}")
                    item_debug["coingecko_error"] = "Empty or invalid data"
                    # Create a dummy dataframe with the current price
                    fetch = create_dummy_dataframe_with_price(item, coin_id, cg)
            except Exception as e:
                st.error(f"Error fetching data from CoinGecko for {item}: {str(e)}")
                item_debug["coingecko_error"] = str(e)
                # Create a dummy dataframe with the current price
                fetch = create_dummy_dataframe_with_price(item, coin_id, cg)
        else:
            st.warning(f"No CoinGecko mapping found for {symbol}")
            item_debug["coingecko_error"] = "No mapping found"
            # Create a dummy dataframe with default values
            fetch = create_dummy_dataframe()
        
        # Add the data to the list
        data.append(fetch)
        debug_info.append(item_debug)
        
        # Update progress bar
        percent_complete += 1.0/len_pairs
        if percent_complete > 1.0:
            percent_complete = 1.0            
        my_bar.progress(percent_complete, text=progress_text+f'...{round(percent_complete*100,1)}%')
    
    # Display debug information
    st.expander("Data Retrieval Debug Info").write(debug_info)
    
    my_bar.progress(1.0, text="Finished...(fetch the data)...100%")
    return data

def create_dummy_dataframe():
    """Create a dummy dataframe with default values for when both data sources fail"""
    # Create a date range for the last 30 days
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a dataframe with default values
    dummy_df = pd.DataFrame(index=date_range)
    dummy_df['Open'] = 0
    dummy_df['High'] = 0
    dummy_df['Low'] = 0
    dummy_df['Close'] = 0
    dummy_df['Adj Close'] = 0
    dummy_df['Volume'] = 0
    
    return dummy_df

def create_dummy_dataframe_with_price(item, coin_id, cg):
    """Create a dummy dataframe with the current price when historical data retrieval fails"""
    try:
        # Try to get the current price from CoinGecko
        price_data = cg.get_price(ids=coin_id, vs_currencies='usd')
        current_price = price_data.get(coin_id, {}).get('usd', 0)
    except Exception:
        current_price = 0
    
    # Create a date range for the last 30 days
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a dataframe with the current price
    dummy_df = pd.DataFrame(index=date_range)
    dummy_df['Open'] = current_price
    dummy_df['High'] = current_price * 1.005
    dummy_df['Low'] = current_price * 0.995
    dummy_df['Close'] = current_price
    dummy_df['Adj Close'] = current_price
    dummy_df['Volume'] = 0
    
    st.info(f"Using current price ({current_price}) for {item} as fallback")
    
    return dummy_df

def extend_dataframe_with_same_dates(df):
    last_day = df.index[-1]  # Last date
    data = df.iloc[-1].values  # Get the last row values
    # Create a new DataFrame with the same values and index for the last 31 days
    new_dates = [last_day - timedelta(days=i) for i in range(31)]
    extended_df = pd.DataFrame([data]*31, columns=df.columns, index=new_dates)
    return extended_df.sort_index()

@st.cache_data(show_spinner=False, ttl = 3600)
def get_data(pairs,start,lb):
    data=[]
    progress_text = "Running...(fetch the data)"
    my_bar = st.progress(0.0, text=progress_text)
    len_pairs=len(pairs)
    percent_complete = 0.0
    my_bar.progress(percent_complete, text=progress_text)
    for item in pairs:
        data.append(getdata(item, lookback=lb).set_index('Time'))#binance
        percent_complete+=1.0/len_pairs 
        my_bar.progress(percent_complete, text=progress_text+f'...{round(percent_complete*100,1)}%')
    my_bar.progress(1.0, text="Finnished...(fetch the data)...100%")
    return data

def plot_investment(actual_value, invest_value, plot_width, plot_height):
    gain_loss = actual_value - invest_value

    colors = ['orange','blue']
    bar_names = ['Actual $', 'Invested $']
    values = [actual_value, invest_value]
    bar_colors = ['green' if gain_loss >= 0 else 'red']

    trace = []
    trace.append(go.Bar(
        y=['Gain/Loss $'],
        x=[gain_loss],
        orientation='h',
        name='Gain/Loss $',
        marker=dict(color=bar_colors),
        hoverinfo='x',
    ))
    
    for i in range(len(bar_names)):
        trace.append(go.Bar(
            y=[bar_names[i]],
            x=[values[i]],
            orientation='h',
            name=bar_names[i],
            marker=dict(color=colors[i]),
            hoverinfo='x',
        ))

    annotations = []
    for i in range(len(bar_names)):
        annotations.append(dict(x=values[i], y=bar_names[i], text=str(values[i]),
                                xanchor='left', 
                                font=dict(color='black'),
                                showarrow=False))

    annotations.append(dict(x=gain_loss, y='Gain/Loss $', text=str(gain_loss),
                            xanchor='left', 
                            font=dict(color='black'),
                            showarrow=False))
    layout = go.Layout(
        title='Status quo',
        barmode='stack',
        width=plot_width,
        height=plot_height,
        annotations=annotations,
    )

    fig = go.Figure(data=trace, layout=layout)
    st.plotly_chart(fig ,use_container_width=True)

def plot_port_assetcat(df, sortcol, highest=10, category=None):
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    if category is not None:
        try:
            df = df[df['Kategorie'] == category]
        except:
            print(f'{category} is not a viable category!')
    
    # Check if required columns exist, if not create them
    # First check and create 'Wert $' if needed
    if 'Wert $' not in df.columns and 'Anzahl' in df.columns and 'Last $' in df.columns:
        df['Wert $'] = df['Anzahl'] * df['Last $']
        st.info("Column 'Wert $' was calculated as 'Anzahl' * 'Last $'")
    
    # Then check and create 'Gain/Loss $' if needed
    if 'Gain/Loss $' not in df.columns and 'Wert $' in df.columns and 'Invest $' in df.columns:
        df['Gain/Loss $'] = df['Wert $'] - df['Invest $']
        st.info("Column 'Gain/Loss $' was calculated as 'Wert $' - 'Invest $'")
    
    # Now check if the sortcol exists, if not create it
    if sortcol not in df.columns:
        st.warning(f"Column '{sortcol}' not found and couldn't be calculated. Using placeholder values.")
        df[sortcol] = 1  # Use placeholder values
    
    # Ensure all values in the sortcol are numeric
    df[sortcol] = pd.to_numeric(df[sortcol], errors='coerce').fillna(0)
    
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Sort the dataframe
    if isinstance(highest, int):
        try:
            df_sorted = df_copy.sort_values(sortcol, ascending=True).tail(highest)
        except Exception as e:
            st.error(f"Error sorting dataframe: {str(e)}")
            # Fallback: just take the last n rows without sorting
            df_sorted = df_copy.tail(highest)
    else: #all
        try:
            df_sorted = df_copy.sort_values(sortcol, ascending=True)
        except Exception as e:
            st.error(f"Error sorting dataframe: {str(e)}")
            # Fallback: just use the dataframe as is
            df_sorted = df_copy
    
    # Ensure all numeric columns are properly converted to numeric
    for col in ['Gain/Loss $', 'Invest $', 'Wert $']:
        if col in df_sorted.columns:
            df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce').fillna(0)
    
    fig = go.Figure()

    # Add trace for Gain/Loss
    fig.add_trace(go.Bar(
        y=df_sorted.index,
        x=df_sorted['Gain/Loss $'],
        orientation='h',
        name='Gain/Loss $',
        text=round(df_sorted['Gain/Loss $']),
        textposition='inside',
        marker=dict(color='orange')
    ))

    # Add trace for Total Investment
    fig.add_trace(go.Bar(
        y=df_sorted.index,
        x=df_sorted['Invest $'],
        orientation='h',
        name='Invest $',
        text=round(df_sorted['Invest $']),
        textposition='inside',
        marker=dict(color='blue')
    ))
    
    # Add trace for Current Value
    fig.add_trace(go.Bar(
        y=df_sorted.index,
        x=df_sorted['Wert $'],
        orientation='h',
        name='Wert $',
        text=round(df_sorted['Wert $']),
        textposition='inside',
        marker=dict(color='green')
    ))
    
    # Update layout
    try:
        min_value = min(0, df_sorted[['Invest $', 'Wert $', 'Gain/Loss $']].min().min() * 1.1)
        max_value = df_sorted[['Invest $', 'Wert $', 'Gain/Loss $']].max().max() * 1.1
    except Exception as e:
        st.warning(f"Error calculating min/max values: {str(e)}. Using default values.")
        min_value = 0
        max_value = 1000  # Default max value
    
    fig.update_layout(
        barmode='group',
        title='Portfolio Investment Analysis',
        xaxis=dict(title='Wert $',range=[min_value, max_value]),
        yaxis=dict(title='Asset'),
        legend=dict(
            x=0.6,
            y=1.10,
            orientation='h'
        )
    )

    # Show plot
    if len(df_sorted)<11:
        fig.update_layout(height=500)
        st.plotly_chart(fig ,use_container_width=True,height=500)
    elif len(df_sorted)<21:
        fig.update_layout(height=30*len(df_sorted))  # Use actual length instead of highest which could be "all"
        st.plotly_chart(fig ,use_container_width=True,height=30*len(df_sorted))
    else: #all
        fig.update_layout(height=1000)
        st.plotly_chart(fig ,use_container_width=True,height=1000)
    
def get_close_price_for_period(dataframe, period):
    end_date = dataframe.index[-1]  # Get the latest date in the dataframe
    start_date = end_date - pd.Timedelta(days=period)  # Calculate the start date based on the period

    # Filter the dataframe for the specified period and return the Close prices
    period_data = dataframe.loc[(dataframe.index >= start_date) & (dataframe.index <= end_date)]
    
    return period_data['Close']
    
def aggregate_specific_column(dataframes_list, column_name, ticker_list):
    aggregated_column = pd.DataFrame()  # Create an empty dataframe to aggregate the column data
    asset_num=0
    for dataframe in dataframes_list:
        if column_name in dataframe.columns:
            # Check if dataframe[column_name] is already a DataFrame or a Series
            if isinstance(dataframe[column_name], pd.Series):
                column_data = dataframe[column_name].to_frame()  # Convert Series to DataFrame
            else:
                # If it's already a DataFrame, just select the column
                column_data = dataframe[[column_name]]
                
            column_data.columns = [f'{ticker_list[asset_num]}']  # Rename the column uniquely
            aggregated_column = pd.concat([aggregated_column, column_data], axis=1)  # Concatenate the column to the new dataframe
        asset_num+=1
    return aggregated_column
    
def multiply_row_elements_and_sum(df, row_index, multiplier_list):
    return (df.iloc[row_index]*multiplier_list).sum()
    
def calc_portfoliovalues(aggregated_prices,lookback,multiplier):
    pval=[]
    lasts=lookback
    leng=len(aggregated_prices)
    for i in range(lasts):
        pval.append([aggregated_prices.iloc[leng-lasts+i].name,multiply_row_elements_and_sum(aggregated_prices, leng-lasts+i, multiplier)])

    return pd.Series([x[1] for x in pval], index=[x[0] for x in pval], name = 'Close')
    
def plot_sparkline(data):
    fig = px.line(data, x=data.index, y='Close')#, title='Close Price Sparkline')
    first_price = data.iloc[0]['Close']
    last_price = data.iloc[-1]['Close']
    if last_price > first_price:
        fig.update_traces(line_color='green')
    else:
        fig.update_traces(line_color='red')
    fig.update_xaxes(visible=False,showticklabels=False)  # Hide x-axis tick labels for a cleaner sparkline
    fig.update_yaxes(visible=False,showticklabels=False)  # Hide y-axis tick labels for a cleaner sparkline
    fig.update_layout(height=400)#, margin=dict(l=0, r=0, b=0, t=0))  # Adjust layout for sparkline
    # Add annotations for the first and last prices - safely convert to int
    try:
        first_price_int = int(first_price)
    except (TypeError, ValueError):
        first_price_int = int(float(first_price)) if isinstance(first_price, str) and first_price.replace('.', '', 1).isdigit() else 0
    
    try:
        last_price_int = int(last_price)
    except (TypeError, ValueError):
        last_price_int = int(float(last_price)) if isinstance(last_price, str) and last_price.replace('.', '', 1).isdigit() else 0
    
    fig.add_annotation(x=data.index[0], y=first_price, text=f'{first_price_int} $', showarrow=True, arrowhead=1)
    fig.add_annotation(x=data.index[-1], y=last_price, text=f'{last_price_int} $', showarrow=True, arrowhead=1)        
    #fig.update_layout(xaxis_title=f'Portfolio last {len(data)} days', yaxis_title='')
    st.plotly_chart(fig ,use_container_width=True)  

# Function to format values with K (thousands), M (millions), B (billions) suffixes
def sizeof_number(number, currency=None):
    """
    Format values per thousands: K-thousands, M-millions, B-billions.
    
    Parameters:
    -----------
    number: The number you want to format
    currency: The prefix that is displayed if provided (€, $, £...)
    
    Returns:
    --------
    Formatted string with appropriate suffix (K, M, B)
    """
    try:
        # Convert to float if it's not already a number
        number = float(number)
    except (ValueError, TypeError):
        return '0' if currency is None else f'{currency}0'
        
    # Define suffixes and thresholds
    suffixes = ['', 'K', 'M', 'B', 'T']
    suffix_idx = 0
    
    # Determine appropriate suffix
    while number >= 1000 and suffix_idx < len(suffixes) - 1:
        suffix_idx += 1
        number /= 1000.0
    
    # Format with 1 decimal place if not a whole number, otherwise as an integer
    if number == int(number):
        formatted = f'{int(number)}'
    else:
        formatted = f'{number:.1f}'
    
    # Add suffix
    formatted += suffixes[suffix_idx]
    
    # Add currency prefix if provided
    if currency is not None:
        formatted = f'{currency}{formatted}'
        
    return formatted

def plot_grouped_bar_chart_with_calculation(dataframe, category_column, quantity_column, price_column):
    # Calculate total value for each stock in each category
    agg_data = dataframe.groupby([dataframe.index, category_column]).apply(lambda x: (x[quantity_column] * x[price_column]).sum()).reset_index(name='Total Value')

    # Pivot the data to prepare for stacked bar chart
    pivot_data = agg_data.pivot(index='Name', columns=category_column, values='Total Value').fillna(0)

    # Calculate total value for each category and sort in descending order
    category_total = pivot_data.sum().sort_values(ascending=False).index

    # Sort pivot data columns based on category total value order
    pivot_data = pivot_data[category_total]

    # Get stock names as the index
    stock_names = pivot_data.index

    # Create a stacked bar chart
    fig = go.Figure()

    for stock in pivot_data.index:
        fig.add_trace(go.Bar(x=pivot_data.columns, y=pivot_data.loc[stock], name=str(stock),
                             text=pivot_data.loc[stock].apply(lambda val: f'{stock} '+sizeof_number(val, currency='$')),
                             hoverinfo='text', showlegend=False,marker=dict(color='blue'),textposition='inside'))

    # Update layout and formatting
    fig.update_layout(title='Total Value by Category - Stacked',
                      xaxis_title='Category', yaxis_title='Total Value $',
                      barmode='stack', width=800, height=400)
    st.plotly_chart(fig ,use_container_width=True)
    
def create_custom_treemap(df, category_column, value_column, gainloss_column):
    # Make a copy of the dataframe to avoid modifying the original
    df_ = df.copy().reset_index()
    
    # Check if the required columns exist, if not create them
    if value_column not in df_.columns:
        # If 'Wert $' doesn't exist but 'Anzahl' and 'Last $' do, calculate it
        if 'Anzahl' in df_.columns and 'Last $' in df_.columns:
            df_[value_column] = df_['Anzahl'] * df_['Last $']
        else:
            st.warning(f"Column '{value_column}' not found and couldn't be calculated. Using placeholder values.")
            df_[value_column] = 1  # Use placeholder values
    
    if gainloss_column not in df_.columns:
        # If 'Gain/Loss $' doesn't exist but 'Wert $' and 'Invest $' do, calculate it
        if value_column in df_.columns and 'Invest $' in df_.columns:
            df_[gainloss_column] = df_[value_column] - df_['Invest $']
        else:
            st.warning(f"Column '{gainloss_column}' not found and couldn't be calculated. Using placeholder values.")
            df_[gainloss_column] = 0  # Use placeholder values
    
    # Create the treemap visualization
    fig = px.treemap(df_, path=[px.Constant("Portfolio"), category_column, 'Name'], values=value_column,
                     color=gainloss_column, hover_data=['Name'],
                     color_continuous_scale='RdYlGn',
                     color_continuous_midpoint=0)
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig ,use_container_width=True)
    
def calculate_asset_value_and_plot(prices_df, portfolio_df, start_date, end_date):
    # Filter data based on input date range
    prices_subset = prices_df.loc[start_date:end_date]
    
    # Multiply prices by number of stocks in the portfolio
    asset_value = prices_subset.multiply(portfolio_df['Anzahl'], axis=1)
    #print(asset_value)
    
    # Merge with the portfolio dataframe to get the category
    asset_value_with_category = asset_value.T.join(portfolio_df['Kategorie'])
    
    # Group by category and sum the values
    aggregated_values = asset_value_with_category.groupby('Kategorie').sum()
    
    # Normalize each category's current value by the first value in the category
    normalized_values = aggregated_values.T / aggregated_values.iloc[:,0]
    
    #color mapping
    cm = {unit: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, unit in enumerate(normalized_values.columns.values)}
    
    # Plotting
    fig = px.line(normalized_values, x=normalized_values.index, y=normalized_values.columns,
                  title='Time Evolution of Asset Categories', color_discrete_map=cm)
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Asset Value Gain/Loss Multiplier')
    st.plotly_chart(fig ,use_container_width=True)
    
#app
if selected == 'Portfolio':
    st.header("Portfolio Overview :eyeglasses:")
    
    # We'll calculate derived columns after lastprices is defined
    # (moved this code to after the lastprices definition)
    
    # Handle potential non-numeric values or NaN values in the 'Invest $' column
    try:
        invest = int(assets['Invest $'].sum())
    except (TypeError, ValueError):
        # Convert to numeric first, coercing errors to NaN, then sum and convert to int
        invest = int(pd.to_numeric(assets['Invest $'], errors='coerce').fillna(0).sum())
    
fstart=pd.to_datetime('2023-01-01').date()
fend=pd.to_datetime('today').date()

today = pd.to_datetime('today').date()
lb = (today-fstart).days
#st.write(f'Lookback time is {lb} days.')
#st.write(dropdown)

if binance:
    data = get_data(pairs,fstart,str(lb))
else:
    data = get_data2(pairs2)

#if st.button("Clear All Cache"):
#    # Clear values from *all* all in-memory and on-disk data caches:
#    # i.e. clear values from both square and cube
#    st.cache_data.clear()
#st.dataframe(data[0])
#st.text(len(data[0]))
lastprices = []
lastweek = []
lastmonth = []
failed_symbols = []

for i, symbol in zip(data, pairs2):
    try:
        if not i.empty and 'Close' in i.columns:
            # Extract scalar values from Series objects using the recommended approach
            lastprices.append(float(i['Close'].iloc[-1].iloc[0] if isinstance(i['Close'].iloc[-1], pd.Series) else i['Close'].iloc[-1]))
            lastweek.append(float(i['Close'].iloc[-7].iloc[0] if isinstance(i['Close'].iloc[-7], pd.Series) else i['Close'].iloc[-7]))
            
            # Handle the conditional expression for lastmonth properly
            last_month_value = i['Close'].iloc[-30] if len(i) >= 30 else i['Close'].iloc[0]
            lastmonth.append(float(last_month_value.iloc[0] if isinstance(last_month_value, pd.Series) else last_month_value))
        else:
            # If data is empty or missing Close column, try to fetch it again
            retry_data = yf.download(symbol, period='7d')
            if not retry_data.empty and 'Close' in retry_data.columns:
                # Extract scalar values from Series objects using the recommended approach
                lastprices.append(float(retry_data['Close'].iloc[-1].iloc[0] if isinstance(retry_data['Close'].iloc[-1], pd.Series) else retry_data['Close'].iloc[-1]))
                
                # Handle the conditional expression for lastweek properly
                last_week_retry = retry_data['Close'].iloc[-7] if len(retry_data) >= 7 else retry_data['Close'].iloc[0]
                lastweek.append(float(last_week_retry.iloc[0] if isinstance(last_week_retry, pd.Series) else last_week_retry))
                
                # Handle the conditional expression for lastmonth properly
                last_month_retry = retry_data['Close'].iloc[-30] if len(retry_data) >= 30 else retry_data['Close'].iloc[0]
                lastmonth.append(float(last_month_retry.iloc[0] if isinstance(last_month_retry, pd.Series) else last_month_retry))
            else:
                # If retry fails, use the last known price from assets DataFrame if available
                if 'Last $' in assets.columns:
                    last_known_price = assets.loc[symbol.split('-')[0], 'Last $']
                    lastprices.append(last_known_price)
                    lastweek.append(last_known_price)
                    lastmonth.append(last_known_price)
                else:
                    # If no last known price, use purchase price as fallback
                    purchase_price = assets.loc[symbol.split('-')[0], 'Kaufpreis $']
                    lastprices.append(purchase_price)
                    lastweek.append(purchase_price)
                    lastmonth.append(purchase_price)
                failed_symbols.append(symbol)
    except Exception as e:
        st.warning(f"Failed to get price for {symbol}: {str(e)}")
        # Use purchase price as fallback
        purchase_price = assets.loc[symbol.split('-')[0], 'Kaufpreis $']
        lastprices.append(purchase_price)
        lastweek.append(purchase_price)
        lastmonth.append(purchase_price)
        failed_symbols.append(symbol)

if failed_symbols:
    st.warning(f"Could not fetch current prices for: {', '.join(failed_symbols)}. Using fallback prices.")

# Only proceed with the assignment if we have the correct number of prices
if len(lastprices) == len(assets):
    assets['Last $'] = lastprices
    assets['LastWeek $'] = lastweek
    assets['LastMonth $'] = lastmonth
    
    # Debug information
    st.expander("Debug Price Information").write({
        "Last Prices": lastprices,
        "Number of Prices": len(lastprices),
        "Price Types": [type(p).__name__ for p in lastprices]
    })
else:
    st.error(f"Price data length mismatch. Expected {len(assets)} prices but got {len(lastprices)}.")

if selected == 'Portfolio':
    # Calculate derived columns
    assets['Wert $'] = assets['Anzahl'] * assets['Last $']
    assets['Wert Lastweek $'] = assets['Anzahl'] * assets['LastWeek $']
    assets['Wert Lastmonth $'] = assets['Anzahl'] * assets['LastMonth $']
    assets['Gain/Loss $'] = assets['Wert $'] - assets['Invest $']
    
    # Debug information for calculation
    st.expander("Debug Calculation Information").write({
        "Anzahl Sample": assets['Anzahl'].head().tolist(),
        "Last $ Sample": assets['Last $'].head().tolist(),
        "Wert $ Sample": assets['Wert $'].head().tolist(),
        "Wert $ Sum": assets['Wert $'].sum(),
        "Wert $ Types": [type(w).__name__ for w in assets['Wert $'].head().tolist()]
    })

    # Now use the calculated columns
    # Ensure all values are properly converted to numeric before summing
    assets['Wert $'] = pd.to_numeric(assets['Wert $'], errors='coerce').fillna(0)
    assets['Wert Lastweek $'] = pd.to_numeric(assets['Wert Lastweek $'], errors='coerce').fillna(0)
    assets['Wert Lastmonth $'] = pd.to_numeric(assets['Wert Lastmonth $'], errors='coerce').fillna(0)
    assets['Gain/Loss $'] = pd.to_numeric(assets['Gain/Loss $'], errors='coerce').fillna(0)
    
    # Calculate the total portfolio value
    wert = int(assets['Wert $'].sum())

    # Handle potential non-numeric values or NaN values in the 'Invest $' column
    try:
        invest = int(assets['Invest $'].sum())
    except (TypeError, ValueError):
        # Convert to numeric first, coercing errors to NaN, then sum and convert to int
        invest = int(pd.to_numeric(assets['Invest $'], errors='coerce').fillna(0).sum())

    plot_investment(wert, invest, 800, 250)
    
    with st.expander("Protfolio Investment Analysis"):
        col1, col2 = st.columns([4,1])
        # Store the initial value of widgets in session state
        with col1:
            assetsnum = st.radio(
                "Set number of assets ",
                [10, 20, "all"],
                index=0,
                horizontal=True 
            )
            # Convert assetsnum to the appropriate type before passing to the function
            assets_to_show = assetsnum if isinstance(assetsnum, int) or assetsnum == "all" else int(assetsnum)
            plot_port_assetcat(assets, 'Wert $', assets_to_show, category=None)

        toggle = col2.toggle('Monthly')
        if toggle:
            dt=30
        else:
            dt=7
        #total = assets['Wert $'].sum()
        aggregated_prices = aggregate_specific_column(data, 'Close', assets.index)
        #st.dataframe(aggregated_prices)    
        portvalues = calc_portfoliovalues(aggregated_prices,dt,assets['Anzahl'].values)
        total = portvalues.tail(1).sum()
        delta=total-portvalues.iloc[-dt].sum()
        col2.metric('Portfolio Total $',value=str(round(total)),delta=str(round(delta)))
        
        #st.write(assets.index)
        with col2:
            plot_sparkline(portvalues.to_frame())
        

if selected == 'Asset Cats Analysis':
    st.header("Asset Cats Overview :eyeglasses:")
        
    plot_grouped_bar_chart_with_calculation(assets, 'Kategorie', 'Anzahl', 'Last $')
    
    with st.expander("Asset Map"):
        create_custom_treemap(assets, 'Kategorie', 'Wert $', 'Gain/Loss $')
    
    with st.expander("Asset Cats Comparison"):
        col3, col4 = st.columns(2)
        start = col3.date_input('Start', pd.to_datetime('2023-01-01'))
        end = col4.date_input('End', pd.to_datetime('today'))
        aggregated_prices = aggregate_specific_column(data, 'Close', assets.index)    
        calculate_asset_value_and_plot(aggregated_prices, assets, start, end)
    
    with st.expander("Single Asset Cat Analysis"):
        col1, col2 = st.columns([4,1])
        # Store the initial value of widgets in session state
        with col1:
            assetcat = st.selectbox(
                "Select assets category ",
                assets['Kategorie'].unique()
            )
            plot_port_assetcat(assets, 'Wert $', "all", assetcat)

        toggle = col2.toggle('Monthly')
        ac = assets[assets['Kategorie'] == assetcat]
        selected = list(ac.index)
        if toggle:
            dt=30
        else:
            dt=7
            
        portvalues = calc_portfoliovalues(aggregated_prices[selected],dt,list(assets.T[selected].T['Anzahl'].values))
        total = portvalues.tail(1).sum()
        delta=total-portvalues.iloc[-dt].sum()
        col2.metric(f'{assetcat} Total $',value=str(round(total)),delta=str(round(delta)))
        
        #st.write(assets.index)
        with col2:
            plot_sparkline(portvalues.to_frame())
    
    
if selected == 'OHCL Single Asset':
    st.header("OHCL Chart Single Asset :eyeglasses:")
    col3, col4, col5 = st.columns(3)
    start = col3.date_input('Start', pd.to_datetime('2023-01-01'))
    end = col4.date_input('End', pd.to_datetime('today'))
    dropdown =col5.selectbox('Pick your asset',pairs)
    
    def calculate_rsi(data, column='Close', window=14):
        # Calculate daily price changes
        delta = data[column].diff(1)

        # Gain and loss for each day
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and average loss over the specified window
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        # Calculate Relative Strength (RS) and Relative Strength Index (RSI)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
    
    def generate_ohlc_chartplotly(symbol, data, start=None, end=None):
        # Check if data is a string (ticker symbol) and fetch data if needed
        if isinstance(data, str):
            st.info(f"Fetching data for {data}...")
            try:
                # For crypto tickers ending with USDT, convert to Yahoo Finance format
                if data.endswith('USDT'):
                    # Convert SOLUSDT to SOL-USD format for Yahoo Finance
                    base_currency = data.replace('USDT', '')
                    yahoo_ticker = f"{base_currency}-USD"
                    st.info(f"Converting {data} to Yahoo Finance format: {yahoo_ticker}")
                    # Download data using the correct Yahoo Finance format
                    crypto_data = yf.download(yahoo_ticker, start=start, end=end)
                    
                    # If no data, try CoinGecko API as fallback
                    if crypto_data.empty:
                        st.info(f"No data from Yahoo Finance. Trying CoinGecko API...")
                        try:
                            # Extract the base currency (e.g., 'SOL' from 'SOLUSDT')
                            base_currency = data.replace('USDT', '').lower()
                            
                            # Use CoinGecko API to get historical data
                            cg = CoinGeckoAPI()
                            # Convert dates to UNIX timestamps (milliseconds)
                            from_timestamp = int(pd.to_datetime(start).timestamp())
                            to_timestamp = int(pd.to_datetime(end).timestamp())
                            
                            # Get market chart data
                            coin_data = cg.get_coin_market_chart_range_by_id(
                                id=base_currency, 
                                vs_currency='usd',
                                from_timestamp=from_timestamp,
                                to_timestamp=to_timestamp
                            )
                            
                            # Process the data into a DataFrame
                            prices = coin_data['prices']  # [[timestamp, price], ...]
                            volumes = coin_data['total_volumes']  # [[timestamp, volume], ...]
                            
                            # Create DataFrame
                            df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
                            df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                            
                            # Convert timestamp to datetime
                            df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
                            df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
                            
                            # Set timestamp as index
                            df_prices.set_index('timestamp', inplace=True)
                            df_volumes.set_index('timestamp', inplace=True)
                            
                            # Merge price and volume data
                            crypto_data = pd.DataFrame()
                            crypto_data['Close'] = df_prices['price']
                            crypto_data['Open'] = df_prices['price'].shift(1)  # Use previous close as open
                            crypto_data['High'] = df_prices['price'] * 1.005  # Approximate high as 0.5% above close
                            crypto_data['Low'] = df_prices['price'] * 0.995   # Approximate low as 0.5% below close
                            crypto_data['Volume'] = df_volumes['volume']
                            
                            # Forward fill missing values
                            crypto_data.fillna(method='ffill', inplace=True)
                            
                            if crypto_data.empty:
                                st.error(f"Could not fetch data for {symbol} from CoinGecko either.")
                                return
                        except Exception as cg_error:
                            st.error(f"Error fetching from CoinGecko: {str(cg_error)}")
                            return
                    
                    data = crypto_data
                else:
                    # For non-crypto assets, use yfinance directly
                    data = yf.download(data, start=start, end=end)
                    
                if data.empty:
                    st.error(f"No data available for {symbol}")
                    return
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {str(e)}")
                return
                
        # Calculate moving averages
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()

        # Calculate Bollinger Bands
        data['std'] = data['Close'].rolling(window=20).std()
        data['upper_band'] = data['MA_20'] + (data['std'] * 2)
        data['lower_band'] = data['MA_20'] - (data['std'] * 2)

        # Calculate Exponential Moving Average for volume with a window of 5 periods
        data['V_EMA_5'] = data['Volume'].ewm(span=5, adjust=False).mean()
        
        # Calculate RSI(14)
        data['RSI_14'] = calculate_rsi(data, column='Close', window=14)

        mask = (pd.to_datetime(data.index) > pd.to_datetime(start)) & (pd.to_datetime(data.index) <= pd.to_datetime(end))
        stock_data = data.loc[mask]

        # Create subplots with shared x-axis
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.015, row_heights=[0.6, 0.2, 0.2])

        # Add candlestick trace to the first subplot
        fig.add_trace(go.Candlestick(x=stock_data.index,
                                     open=stock_data['Open'],
                                     high=stock_data['High'],
                                     low=stock_data['Low'],
                                     close=stock_data['Close'],
                                     name='OHLC'), row=1, col=1)

        # Add Bollinger Bands traces to the first subplot
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['upper_band'], mode='lines', name='Upper Band', line=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['lower_band'], mode='lines', name='Lower Band', line=dict(color='red')), row=1, col=1)

        # Add moving averages traces to the first subplot
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA_200'], mode='lines', name='200-day Moving Average', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA_50'], mode='lines', name='50-day Moving Average', line=dict(color='orange')), row=1, col=1)

        # Add Volume subplot to the second subplot
        fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume', marker_color='purple'), row=2, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['V_EMA_5'], mode='lines', name='V_EMA_5', line=dict(color='black')), row=2, col=1)

        # Add RSI subplot to the third subplot
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI_14'], mode='lines', name='RSI(14)', line=dict(color='orange')), row=3, col=1)
        fig.add_shape(type='line', x0=stock_data.index.min(), x1=stock_data.index.max(), y0=30, y1=30, row=3, col=1, line=dict(color='green', width=2), name='RSI 30')
        fig.add_shape(type='line', x0=stock_data.index.min(), x1=stock_data.index.max(), y0=70, y1=70, row=3, col=1, line=dict(color='red', width=2), name='RSI 70')

        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            title=f'{symbol} OHLC Chart',
            #xaxis=dict(title='Date'),
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Volume', anchor='x', side='bottom'),
            yaxis3=dict(title='RSI(14)', anchor='x', side='left'),
            legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=8)),
            height=600  # Adjust the height as needed
        )

        # Calculate performance percentage - safely handle empty data
        if not stock_data.empty:
            try:
                # Use iloc to safely access first and last elements
                first_close = stock_data['Close'].iloc[0]
                last_close = stock_data['Close'].iloc[-1]
                perf_percent = round((last_close / first_close - 1.0) * 100, 1)
                fig.update_layout(title=f'{symbol} OHLC Chart - Performance: {perf_percent}%', xaxis_rangeslider_visible=False)
            except (IndexError, ZeroDivisionError) as e:
                # Handle potential errors
                st.warning(f"Could not calculate performance: {str(e)}")
                fig.update_layout(title=f'{symbol} OHLC Chart', xaxis_rangeslider_visible=False)
        else:
            fig.update_layout(title=f'{symbol} OHLC Chart - No data available', xaxis_rangeslider_visible=False)
        #fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)
        
    # Fix the truth value error by properly comparing DataFrames
    # Find the index of the selected dropdown value in pairs
    try:
        # For crypto tickers ending with USDT, convert to Yahoo Finance format
        if dropdown.endswith('USDT'):
            # Extract the base currency (e.g., 'SOL' from 'SOLUSDT')
            base_currency = dropdown.replace('USDT', '')
            # Create the proper Yahoo Finance ticker format
            yahoo_ticker = f"{base_currency}-USD"
            # Generate the chart with the proper display name and Yahoo ticker format
            generate_ohlc_chartplotly(f"{base_currency}/USD", yahoo_ticker, start=start, end=end)
        else:
            # For regular assets, use the index lookup method
            index_of_selected = pairs.index(dropdown)
            # Use the index to get the corresponding data
            datafiltered = pairs2[index_of_selected]
            # Generate the chart
            generate_ohlc_chartplotly(dropdown, datafiltered, start=start, end=end)
    except ValueError:
        st.error(f"Could not find data for {dropdown}. Available tickers: {', '.join(pairs[:5])}...")
    except Exception as e:
        st.error(f"Error generating chart: {str(e)}")
        # Add debugging information
        st.expander("Debug Information").write({
            "Selected Ticker": dropdown,
            "Ticker Type": type(dropdown).__name__,
            "Sample Available Tickers": pairs[:5] if len(pairs) > 5 else pairs
        })

# The sizeof_number function has been moved above the plot_grouped_bar_chart_with_calculation function

# This is the original sizeof_number function that was in the codebase
# It has been commented out because we've implemented a more comprehensive version above
# def sizeof_number(number, currency=None):
#     """
#     format values per thousands : K-thousands, M-millions, B-billions. 
#     
#     parameters:
#     -----------
#     number is the number you want to format
#     currency is the prefix that is displayed if provided (€, $, £...)
#     
#     """
#     currency=''if currency is None else currency+''
#     for unit in ['','K','M']:
#         if abs(number) < 1000.0:
#             return f"{currency}{number:6.1f}{unit}"
#         number /= 1000.0
#     return f"{currency}{number:6.1f}B"
