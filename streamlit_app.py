import streamlit as st
import lightgbm as lgb
import numpy as np
import pandas as pd
import pandas_ta as ta

import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

import json


import yfinance as yf



# Fama French 49 industries
# see https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html



DATA = pd.read_csv(r'Vars_small.csv')
ff49 = DATA.Industry.drop_duplicates().values.tolist()
c = DATA['Country/Region Code '].drop_duplicates().values.tolist()
# Background
st.sidebar.header("Valuation Inputs")
Ticker     = st.sidebar.text_input(label='Ticker',value="AAPL")

selected_ff49 = st.sidebar.selectbox('Industry',ff49, index=0)
selected_country = st.sidebar.selectbox('Country',c, index=0)
industry = ff49.index(selected_ff49) + 1 # python is zero-indexed, FF49 starts at 1
country = c.index(selected_country) + 1 
rate1yr  = st.sidebar.slider('1 Year Real Treasury Yield - %',  min_value = -5.0, max_value=12.0, step=0.1, value=2.0) / 100






st.title(f'Firm   {Ticker}')
st.header("Price and Volume")

   
 
 
COLOR_BULL = 'rgba(38,166,154,0.9)' # #26a69a
COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350
# Request historic pricing data via finance.yahoo.com API
df = yf.Ticker(Ticker).history(period='4mo')[['Open', 'High', 'Low', 'Close', 'Volume']]

# Some data wrangling to match required format
df = df.reset_index()
df.columns = ['time','open','high','low','close','volume']                  # rename columns
df['time'] = df['time'].dt.strftime('%Y-%m-%d')                             # Date to string
df['color'] = np.where(  df['open'] > df['close'], COLOR_BEAR, COLOR_BULL)  # bull or bear
df.ta.macd(close='close', fast=6, slow=12, signal=5, append=True)           # calculate macd

# export to JSON format
candles = json.loads(df.to_json(orient = "records"))
volume = json.loads(df.rename(columns={"volume": "value",}).to_json(orient = "records"))
macd_fast = json.loads(df.rename(columns={"MACDh_6_12_5": "value"}).to_json(orient = "records"))
macd_slow = json.loads(df.rename(columns={"MACDs_6_12_5": "value"}).to_json(orient = "records"))
df['color'] = np.where(  df['MACD_6_12_5'] > 0, COLOR_BULL, COLOR_BEAR)  # MACD histogram color
macd_hist = json.loads(df.rename(columns={"MACD_6_12_5": "value"}).to_json(orient = "records"))


chartMultipaneOptions = [
    {
        "width": 600,
        "height": 400,
        "layout": {
            "background": {
                "type": "solid",
                "color": 'white'
            },
            "textColor": "black"
        },
        "grid": {
            "vertLines": {
                "color": "rgba(197, 203, 206, 0.5)"
                },
            "horzLines": {
                "color": "rgba(197, 203, 206, 0.5)"
            }
        },
        "crosshair": {
            "mode": 0
        },
        "priceScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)"
        },
        "timeScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)",
            "barSpacing": 15
        },
        "watermark": {
            "visible": True,
            "fontSize": 48,
            "horzAlign": 'center',
            "vertAlign": 'center',
            "color": 'rgba(171, 71, 188, 0.3)',
            "text": f'{Ticker}',
        }
    },
    {
        "width": 600,
        "height": 100,
        "layout": {
            "background": {
                "type": 'solid',
                "color": 'transparent'
            },
            "textColor": 'black',
        },
        "grid": {
            "vertLines": {
                "color": 'rgba(42, 46, 57, 0)',
            },
            "horzLines": {
                "color": 'rgba(42, 46, 57, 0.6)',
            }
        },
        "timeScale": {
            "visible": False,
        },
        "watermark": {
            "visible": True,
            "fontSize": 18,
            "horzAlign": 'left',
            "vertAlign": 'top',
            "color": 'rgba(171, 71, 188, 0.7)',
            "text": 'Volume',
        }
    },
    {
        "width": 600,
        "height": 200,
        "layout": {
            "background": {
                "type": "solid",
                "color": 'white'
            },
            "textColor": "black"
        },
        "timeScale": {
            "visible": False,
        },
        "watermark": {
            "visible": True,
            "fontSize": 18,
            "horzAlign": 'left',
            "vertAlign": 'center',
            "color": 'rgba(171, 71, 188, 0.7)',
            "text": 'MACD',
        }
    }
]

seriesCandlestickChart = [
    {
        "type": 'Candlestick',
        "data": candles,
        "options": {
            "upColor": COLOR_BULL,
            "downColor": COLOR_BEAR,
            "borderVisible": False,
            "wickUpColor": COLOR_BULL,
            "wickDownColor": COLOR_BEAR
        }
    }
]

seriesVolumeChart = [
    {
        "type": 'Histogram',
        "data": volume,
        "options": {
            "priceFormat": {
                "type": 'volume',
            },
            "priceScaleId": "" # set as an overlay setting,
        },
        "priceScale": {
            "scaleMargins": {
                "top": 0,
                "bottom": 0,
            },
            "alignLabels": False
        }
    }
]

seriesMACDchart = [
    {
        "type": 'Line',
        "data": macd_fast,
        "options": {
            "color": 'blue',
            "lineWidth": 2
        }
    },
    {
        "type": 'Line',
        "data": macd_slow,
        "options": {
            "color": 'green',
            "lineWidth": 2
        }
    },
    {
        "type": 'Histogram',
        "data": macd_hist,
        "options": {
            "color": 'red',
            "lineWidth": 1
        }
    }
]


renderLightweightCharts([
    {
        "chart": chartMultipaneOptions[0],
        "series": seriesCandlestickChart
    },
    {
        "chart": chartMultipaneOptions[1],
        "series": seriesVolumeChart
    },
    {
        "chart": chartMultipaneOptions[2],
        "series": seriesMACDchart
    }
], 'multipane')



#st.set_page_config(page_title='', page_icon=":dollar:", layout='centered', initial_sidebar_state='expanded')
st.header('Machine Valuation')
st.write('WARNING: Experimental Research Project Alpha version. Do **NOT** use for investment decisions!')


comp_data = DATA[DATA["Ticker "]==Ticker]

industry = DATA['Industry']
#rate1yr  = st.sidebar.slider('1 Year Real Treasury Yield - %',  min_value = -5.0, max_value=12.0, step=0.1, value=2.0) / 100
axis_v = ["FQ42022","FQ32022","FQ22022","FQ12022",
          "FQ42021","FQ32021","FQ22021","FQ12021",
          "FQ42020","FQ32020","FQ22020","FQ12020",
          "FQ42019","FQ32019","FQ22019","FQ12019",
          "FQ42018","FQ32018","FQ22018","FQ12018",
          "FQ42017","FQ32017","FQ22017","FQ12017"]
# P&L
sale_val     = comp_data[[column for column in comp_data.columns if column.startswith('Total Revenue')]].set_axis(axis_v,axis=1)
ebitda_val   = comp_data[[column for column in comp_data.columns if column.startswith('EBIT (')]].set_axis(axis_v,axis=1)
ib_val       = comp_data[[column for column in comp_data.columns if column.startswith('Net Income (')]].set_axis(axis_v,axis=1)

# Balancesheet
debt_val     = comp_data[[column for column in comp_data.columns if column.startswith('Total Debt')]].set_axis(axis_v,axis=1)
net_debt_val     = comp_data[[column for column in comp_data.columns if column.startswith('Net Debt')]].set_axis(axis_v,axis=1)
book_val     = comp_data[[column for column in comp_data.columns if column.startswith('Common Stock')]].set_axis(axis_v,axis=1)
mcap_val = comp_data[[column for column in comp_data.columns if column.startswith('MCap')]].set_axis(axis_v,axis=1)
fr_fm_val = comp_data[[column for column in comp_data.columns if column.startswith('fr_fm')]].set_axis(axis_v,axis=1)
fins = pd.concat([sale_val.T,ebitda_val.T,ib_val.T],axis=1).set_axis(["Sales","EBIT","Net Income"],axis = 1)/1000
fins = fins.iloc[::-1]
st.area_chart(fins)

# P&L
sale     = st.sidebar.number_input('Sales - $ mn', min_value=0.0, max_value=1000000.0,value=sale_val.values[0,1]/1000, step=10.0)
ebitda   = st.sidebar.number_input('EBIT - $ mn', min_value=0.0, max_value=sale, value= ebitda_val.values[0,1]/1000, step=10.0)
ib       = st.sidebar.number_input('Income After Tax - $ mn', min_value=-100000.0, max_value=1000000.0, value=ib_val.values[0,1]/1000, step=10.0)

# Balancesheet
debt     = st.sidebar.number_input('Total Debt - $ mn', min_value=0.0, max_value=1000000.0, value=debt_val.values[0,1]/1000 , step=10.0)
book     = st.sidebar.number_input('Book Value of Equity - $ mn', min_value=0.0, max_value=1000000.0, value=book_val.values[0,1]/1000, step=10.0)

# Calculated items

rate1yr_mc = rate1yr
ib_eb   = ib/ebitda 
debt_eb = debt/ebitda 
book_eb = book/ebitda 
sale_eb = sale/ebitda 


loaded_model = lgb.Booster(model_file='base_model.txt')

# Index(['book_eb', 'debt_eb', 'ib_eq', 'industry', 'rate1yr_mc', 'sale_eb'], dtype='object')
X_dict = {'book_eb':book_eb, 'debt_eb':debt_eb, 'ib_eb':ib_eb, 'industry':industry, 'rate1yr_mc':rate1yr_mc, 'sale_eb':sale_eb}

# convert to list of values, sorted by feature name (order matters for predict)
X = [X_dict[key] for key in sorted(X_dict.keys())]

pred = loaded_model.predict(data=[X])
multiple = round(np.exp(pred[0]),2)
discountrate = round((1/multiple)*100,2)
value = round(ebitda*multiple,0)

st.header('Estimated EBITDA Valuation Multiple')

st.write(f"EBITDA multiple = ** {multiple} x **")

st.header('Estimated Enterprise Valuation')

st.write(f"= EBITDA x EBITDA multiple = $ {ebitda} mn x {multiple} = **$ {value} mn**")

st.header('Implied EBITDA Discount Rate (Zero Growth)')

st.write(f"= 1 /  EBITDA multiple = 1 / {multiple} = ** {discountrate} % **")

st.header("Variables Used")

X_df = pd.DataFrame(X_dict, index=[0])
st.write(X_df)
