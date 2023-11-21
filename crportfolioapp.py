import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from binance.client import Client
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go

client = Client()

assets = pd.read_excel('crassets.xlsx',index_col=0)
coins = [str(assets.index[i]) for i in range(len(assets.index))]
pairs = [coins[i] + 'USDT' for i in range(len(coins))]

def getdata(pair, lookback='365'):
    frame = pd.DataFrame(client.get_historical_klines(pair, '1d', lookback + ' days ago UTC'))
    frame = frame.iloc[:,:]
    frame.columns = ['Time','Open','High','Low','Close','Volume', \
                     'CloseTime','QuoteAssetVolume','Trades','TakerBaseAssetVolume','takerQuoteAssetVolume','Ignored']
    frame[['Open','High','Low','Close','Volume', \
           'QuoteAssetVolume','Trades','TakerBaseAssetVolume','takerQuoteAssetVolume']] \
    = frame[['Open','High','Low','Close','Volume', \
           'QuoteAssetVolume','Trades','TakerBaseAssetVolume','takerQuoteAssetVolume']].astype(float)
    frame.Time = pd.to_datetime(frame.Time, unit='ms')
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
st.title('My Crypto Dashboard :rocket:')

col1, col2, col3 = st.columns(3)
dropdown =col1.selectbox('Pick your asset',pairs)
start = col2.date_input('Start', pd.to_datetime('2023-01-01'))
end = col3.date_input('End', pd.to_datetime('today'))

today = pd.to_datetime('today').date()
lb = (today-start).days
st.write(f'Lookback time is {lb} days.')
#st.write(dropdown)

@st.cache_data
def get_data(pairs,start,lb):
    data=[]
    for item in pairs:
        data.append(getdata(item, lookback=lb).set_index('Time'))
    return data
        
data = get_data(pairs,start,str(lb))
#st.dataframe(data[0])

def generate_ohlc_chartplotly(symbol,data,start=start,end=end):
    # Calculate moving averages
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()

    # Calculate Bollinger Bands
    data['std'] = data['Close'].rolling(window=20).std()
    data['upper_band'] = data['MA_20'] + (data['std'] * 2)
    data['lower_band'] = data['MA_20'] - (data['std'] * 2)

    mask = (pd.to_datetime(data.index) > pd.to_datetime(start)) & (pd.to_datetime(data.index) <= pd.to_datetime(end))
    stock_data=data.loc[mask]
    # Create the figure and add candlestick trace
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'],
                                         name='OHLC')])

    # Add Bollinger Bands trace
    fig.add_trace(go.Scatter(x=data.index, y=data['upper_band'], mode='lines', name='Upper Band', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data.index, y=data['lower_band'], mode='lines', name='Lower Band', line=dict(color='red')))

    # Add moving averages traces
    fig.add_trace(go.Scatter(x=stock_data.index, y=data['MA_200'], mode='lines', name='200-day Moving Average', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=stock_data.index, y=data['MA_50'], mode='lines', name='50-day Moving Average', line=dict(color='orange')))

    # Set layout and show the chart
    fig.update_layout(title=f'{symbol} OHLC Chart', xaxis_rangeslider_visible=False)
    fig.update_layout(
    legend=dict(
        x=0,
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=8
            #color="black"
        ),
    ))
    fig.update_layout(autosize=True)
    #fig.update_layout(
    #autosize=True,
    #width=1000,
    #height=600)
    st.plotly_chart(fig, use_container_width=True)
    #fig.show()

datafiltered=[j for i, j in zip(pairs, data) if i==dropdown][0] 
generate_ohlc_chartplotly(dropdown,datafiltered,start=start,end=today)
