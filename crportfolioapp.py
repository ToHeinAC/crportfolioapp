import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sqlalchemy import create_engine
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import yfinance as yf
import appdirs as ad
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

#app
@st.cache_data(show_spinner=False, ttl = 3600)
def get_data(pairs,start,lb):
    data=[]
    progress_text = "Running...(fetch the data)"
    my_bar = st.progress(0.0, text=progress_text)
    len_pairs=len(pairs)
    percent_complete = 0.0
    #latest_iteration = st.empty()
    my_bar.progress(percent_complete, text=progress_text)
    for item in pairs:
        data.append(getdata(item, lookback=lb).set_index('Time'))#binance
        percent_complete+=1.0/len_pairs 
        my_bar.progress(percent_complete, text=progress_text+f'...{round(percent_complete*100,1)}%')
        #latest_iteration.text(f'{round(percent_complete*100,1)}%')
    my_bar.progress(1.0, text="Finnished...(fetch the data)...100%")
    return data

@st.cache_data(show_spinner=False, ttl = 3600)
def get_data2(pairs,period='5y'):
    data=[]
    progress_text = "Running...(fetch the data)"
    my_bar = st.progress(0.0, text=progress_text)
    len_pairs=len(pairs)
    percent_complete = 0.0
    #latest_iteration = st.empty()
    my_bar.progress(percent_complete, text=progress_text)
    for item in pairs:
        #fetch = pdr.get_data_yahoo(item, start=start, end=end)#pandas datareader with yahoo finance
        #st.write(item)
        fetch = yf.download(item, period=period)#yahoo finance
        #st.dataframe(fetch)
        data.append(fetch)
        percent_complete+=1.0/len_pairs
        if percent_complete > 1.0:
            percent_complete=1.0            
        my_bar.progress(percent_complete, text=progress_text+f'...{round(percent_complete*100,1)}%')
        #latest_iteration.text(f'{round(percent_complete*100,1)}%')
    my_bar.progress(1.0, text="Finnished...(fetch the data)...100%")
    return data

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
lastprices=[]
lastweek=[]
lastmonth=[]
for i in data:
    lastprices.append(i['Close'].iloc[-1])
    lastweek.append(i['Close'].iloc[-7])
    lastmonth.append(i['Close'].iloc[-30])
    
assets['Last $']=lastprices
assets['Wert $']=assets['Anzahl']*assets['Last $']
assets['Lastweek $']=lastweek
assets['Lastmonth $']=lastmonth
assets['Wert Lastweek $']=assets['Anzahl']*assets['Lastweek $']
assets['Wert Lastmonth $']=assets['Anzahl']*assets['Lastmonth $']
assets['Gain/Loss $'] = assets['Wert $'] - assets['Invest $']

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
    if category is not None:
        try:
            df = df[df['Kategorie'] == category]
        except:
            print(f'{category} is not a viable category!')
    
    if isinstance(highest, int):
        df_sorted = df.sort_values(sortcol, ascending=True)[-highest:]
    else: #all
        df_sorted = df.sort_values(sortcol, ascending=True)
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
    #if len(df_sorted[['Invest $', 'Wert $', 'Gain/Loss $']])>1:
    #    min_value = min(0, df_sorted[['Invest $', 'Wert $', 'Gain/Loss $']].min().min() * 1.1)
    #    max_value = df_sorted[['Invest $', 'Wert $', 'Gain/Loss $']].max().max() * 1.1
    #else:
    #    min_value = min(0,df_sorted[['Invest $', 'Wert $', 'Gain/Loss $']].min() * 1.1)
    #    max_value = df_sorted[['Invest $', 'Wert $', 'Gain/Loss $']].max() * 1.1
    min_value = min(0, df_sorted[['Invest $', 'Wert $', 'Gain/Loss $']].min().min() * 1.1)
    max_value = df_sorted[['Invest $', 'Wert $', 'Gain/Loss $']].max().max() * 1.1    
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
        fig.update_layout(height=30*highest)
        st.plotly_chart(fig ,use_container_width=True,height=30*highest)
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
            column_data = dataframe[column_name].to_frame()  # Extract the specific column data
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
    # Add annotations for the first and last prices
    fig.add_annotation(x=data.index[0], y=first_price, text=f'{int(first_price)} $', showarrow=True, arrowhead=1)
    fig.add_annotation(x=data.index[-1], y=last_price, text=f'{int(last_price)} $', showarrow=True, arrowhead=1)        
    #fig.update_layout(xaxis_title=f'Portfolio last {len(data)} days', yaxis_title='')
    st.plotly_chart(fig ,use_container_width=True)  

#def plot_grouped_bar_chart_with_calculation(dataframe, category_column, quantity_column, price_column):
#    agg_data = dataframe.groupby(category_column).apply(lambda x: (x[quantity_column] * x[price_column]).sum()).reset_index(name='Total Value')
#    sort_data = agg_data.sort_values('Total Value', ascending=False) 
#        
#    fig = px.bar(sort_data, x='Kategorie', y='Total Value', text='Total Value',
#    title='Total Value by Category', labels={'Kategorie': 'Category', 'Total Value': 'Total Value'})
#
#    fig.update_traces(texttemplate='%{text:.2s}', textposition='inside')
#    fig.update_layout(xaxis_title='Category', yaxis_title='Total  Value $',width=800,height=400)   
#    # Show plot
#    st.plotly_chart(fig ,use_container_width=True)
    
def sizeof_number(number, currency=None):
    """
    format values per thousands : K-thousands, M-millions, B-billions. 
    
    parameters:
    -----------
    number is the number you want to format
    currency is the prefix that is displayed if provided (€, $, £...)
    
    """
    currency=''if currency is None else currency+''
    for unit in ['','K','M']:
        if abs(number) < 1000.0:
            return f"{currency}{number:6.1f}{unit}"
        number /= 1000.0
    return f"{currency}{number:6.1f}B"

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
    df_ = df.reset_index()
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
    wert=int(assets['Wert $'].sum())
    invest=int(assets['Invest $'].sum())
    plot_investment(wert, invest, 800, 250)
    
    with st.expander("Protfolio Investment Analysis"):
        col1, col2 = st.columns([4,1])
        # Store the initial value of widgets in session state
        with col1:
            assetsnum = st.radio(
                "Set number of assets 👇",
                [10, 20, "all"],
                index=0,
                horizontal=True 
            )
            plot_port_assetcat(assets,'Wert $',assetsnum,category=None)

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
                "Select assets category 👇",
                assets['Kategorie'].unique()
            )
            plot_port_assetcat(assets,'Wert $','all',assetcat)

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

        # Calculate performance percentage
        perf_percent = round((stock_data['Close'][-1] / stock_data['Close'][0] - 1.0) * 100, 1)
        fig.update_layout(title=f'{symbol} OHLC Chart - Performance: {perf_percent}%', xaxis_rangeslider_visible=False)
        #fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)
        
    datafiltered=[j for i, j in zip(pairs, data) if i==dropdown][0] 
    generate_ohlc_chartplotly(dropdown,datafiltered,start=start,end=end)
