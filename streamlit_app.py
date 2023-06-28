import streamlit as st
import lightgbm as lgb
import numpy as np
import pandas as pd
import pandas_ta as ta
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts
import json
import altair as alt
from streamlit_lottie import st_lottie
  

import yfinance as yf



# Fama French 49 industries
# see https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html



DATA = pd.concat([pd.read_csv(r'Vars_small1.csv'),pd.read_csv(r'Vars_small2.csv'),pd.read_csv(r'Vars_small3.csv')])
ff49 = [
    '1-Agriculture',
    '2-Food Products',
    '3-Candy & Soda',
    '4-Beer & Liquor',
    '5-Tobacco Products',
    '6-Recreation',
    '7-Entertainment',
    '8-Printing and Publishing',
    '9-Consumer Goods',
    '10-Apparel',
    '11-Healthcare',
    '12-Medical Equipment',
    '13-Pharmaceutical Products',
    '14-Chemicals',
    '15-Rubber and Plastic Products',
    '16-Textiles',
    '17-Construction Materials',
    '18-Construction',
    '19-Steel Works Etc',
    '20-Fabricated Products',
    '21-Machinery',
    '22-Electrical Equipment',
    '23-Automobiles and Trucks',
    '24-Aircraft',
    '25-Shipbuilding',
    '26-Defense',
    '27-Precious Metals',
    '28-Non-Metallic and Industrial Metal Mining',
    '29-Coal',
    '30-Petroleum and Natural Gas',
    '31-Utilities',
    '32-Communication',
    '33-Personal Services',
    '34-Business Services',
    '35-Computers',
    '36-Computer Software',
    '37-Electronic Equipment',
    '38-Measuring and Control Equipment',
    '39-Business Supplies',
    '40-Shipping Containers',
    '41-Transportation',
    '42-Wholesale',
    '43-Retail',
    '44-Restaurants',
    '45-Banking',
    '46-Insurance',
    '47-Real Estate',
    '48-Trading',
    '49-Almost Nothing or Missing'
]

c = DATA['Country/Region Code '].drop_duplicates().values.tolist()
# Background
st.sidebar.header("Valuation Inputs")
Ticker     = st.sidebar.text_input(label='Ticker',value="AAPL")

selected_ff49 = st.sidebar.selectbox('Industry',ff49, index=0)
selected_country = st.sidebar.selectbox('Country',c, index=0)


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

axis_v = ["FQ42022","FQ32022","FQ22022","FQ12022",
          "FQ42021","FQ32021","FQ22021","FQ12021",
          "FQ42020","FQ32020","FQ22020","FQ12020",
          "FQ42019","FQ32019","FQ22019","FQ12019",
          "FQ42018","FQ32018","FQ22018","FQ12018",
          "FQ42017","FQ32017","FQ22017","FQ12017"]
st.header("Financials")

if Ticker in DATA["Ticker "].values:
   #EV vs EBIT
   mcap_peer =  DATA[[column for column in DATA.columns if column.startswith('MCap')]].set_axis(axis_v,axis=1)
   mcap_peer = mcap_peer[mcap_peer<=1*10**14]
   ni_peer = DATA[[column for column in DATA.columns if column.startswith('Net Income (')]].set_axis(axis_v,axis=1)
   ni_peer = ni_peer[(ni_peer<10**6)&((ni_peer>-10**5))]
   ev_peer = (mcap_peer - DATA[[column for column in DATA.columns if column.startswith('Net Debt')]].set_axis(axis_v,axis=1))
   ebit_peer = DATA[[column for column in DATA.columns if column.startswith('EBIT (')]].set_axis(axis_v,axis=1)
   ebit_peer = ebit_peer[(ebit_peer<10**6)&((ebit_peer>=-0.5*10*3))]
   sales_peer = DATA[[column for column in DATA.columns if column.startswith('Total Revenue')]].set_axis(axis_v,axis=1)
   sales_peer = sales_peer[(sales_peer<10**7)&((sales_peer>=0))]
   
   peer_df = pd.concat([ev_peer.median(axis=1),sales_peer.median(axis=1),ebit_peer.median(axis=1),mcap_peer.median(axis=1),ni_peer.median(axis=1),DATA["Industry"]]
                       ,axis=1).set_axis(["EV","Sales","EBIT","MCap","Net Income","Industry"],axis=1).dropna()
   
   comp_data = DATA[DATA["Ticker "]==Ticker]
   
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
   ev = mcap_val + net_debt_val
   fins_cf = (pd.concat([sale_val.T,ebitda_val.T,ib_val.T],axis=1).set_axis(["Sales","EBIT","Net Income"],axis = 1)/1000).iloc[::-1]
   fins_bs = (pd.concat([debt_val.T,book_val.T],axis=1).set_axis(["Debt","Equity"],axis = 1)/1000).iloc[::-1] 

   #Rearrange Values
   sale_val     = sale_val.values[0,1]/1000
   ebitda_val   = ebitda_val.values[0,1]/1000
   ib_val       = ib_val.values[0,1]/1000
   
   # Balancesheet
   debt_val     = debt_val.values[0,1]/1000
   book_val     = book_val.values[0,1]/1000
   

   #Plotting the results
   
   data_container = st.container()
   with data_container:
    plot1, plot2 = st.columns(2)
    with plot1:
        st.area_chart(fins_cf, use_container_width=False)
    with plot2:
        st.bar_chart(fins_bs, use_container_width=False)
   
   
   chart1 = alt.Chart(peer_df).mark_circle().encode(x='Sales',y='EV',color='Industry',).interactive()
   chart2 = alt.Chart(peer_df).mark_circle().encode(x='Net Income',y='MCap',color='Industry',).interactive()
   
   tab1, tab2 = st.tabs(["EV/Sales", "P/E"])

   with tab1:
       # Use the Streamlit theme.
       st.altair_chart(chart1, theme="streamlit", use_container_width=True)
   with tab2:
       # Use the native Altair theme.
       st.altair_chart(chart2, theme="streamlit", use_container_width=True)
      
      


else:
  st.write("Ops, the ticker you've chosen is not available at current moment. We are working hard to improve our product and will add your desired company into our dataset")
  url = requests.get("https://assets9.lottiefiles.com/packages/lf20_3kjzsbjv.json")
  # Creating a blank dictionary to store JSON file,
  # as their structure is similar to Python Dictionary
  url_json = url.json()

  
  st_lottie(url_json)
  sale_val     = 10000.0
  ebitda_val   = 5000.0
  ib_val       = 2500.0
  # Balancesheet
  debt_val     = 10000.0
  net_debt_val = 1000.0
  book_val     = 100000.0
  mcap_val     = "Not available at the current moment"
  fr_fm_val    = 1
  

#Categorical
country = c.index(selected_country) + 1 
industry = ff49.index(selected_ff49) + 1 # python is zero-indexed, FF49 starts at 1

#Rate
rate1yr  = st.sidebar.slider('1 Year Real Treasury Yield - %',  min_value = -2.0, max_value=12.0, step=0.1, value=2.0) / 100
# P&L
sale     = st.sidebar.number_input('Sales - $ mn', min_value=0.0, max_value=1000000.0,value=sale_val, step=10.0)
ebitda   = st.sidebar.number_input('EBIT - $ mn', min_value=0.0, max_value=sale, value= ebitda_val, step=10.0)
ib       = st.sidebar.number_input('Income After Tax - $ mn', min_value=-100000.0, max_value=1000000.0, value=ib_val, step=10.0)

# Balancesheet
debt     = st.sidebar.number_input('Total Debt - $ mn', min_value=0.0, max_value=1000000.0, value=debt_val , step=10.0)
book     = st.sidebar.number_input('Book Value of Equity - $ mn', min_value=0.0, max_value=1000000.0, value=book_val, step=10.0)

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
st.header('Machine Valuation')

st.subheader('Estimated EBIT Valuation Multiple')

st.write(f"EBIT multiple = **{multiple} x**")

st.subheader('Estimated Enterprise Valuation')

st.write(f"= EBIT x EBIT multiple =  **{ebitda} mn x {multiple}** =  {value} mn")

st.subheader('Implied EBIT Discount Rate (Zero Growth)')

st.write(f"= 1 /  EBIT multiple = 1 / {multiple} = **{discountrate} %**")

st.subheader("Variables Used")

X_df = pd.DataFrame(X_dict, index=[0])
st.write(X_df)

st.write('WARNING: Experimental Research Project Alpha version. Do **NOT** use for investment decisions!')
