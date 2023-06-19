import streamlit as st
from  PIL import Image
import os 
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from urllib.request import Request, urlopen
#from datetime import timezone #timezone 변경
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openai
import finnhub

# Name in the sidebar
st.set_page_config(page_title = 'Stock Market AI')
###################
def sidebar_bg():
   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url("https://cdn.pixabay.com/photo/2013/09/26/00/21/swan-186551_1280.jpg")
               }}
      </style>
      """,
      unsafe_allow_html=True,
      )
sidebar_bg()
###############################################
###############################################
# NAVIGATION BAR 
#https://discuss.streamlit.io/t/the-navigation-bar-im-trying-to-add-to-my-streamlit-app-is-blurred/24104/3
# First clean up the bar 
st.markdown(
"""
<style>
header[data-testid="stHeader"] {
    background: none;
}
</style>
""",
    unsafe_allow_html=True,
)
# Then put the followings (Data Prof: https://www.youtube.com/watch?v=hoPvOIJvrb8)
# Background color에 붉은빛이 약간 들어간 #ffeded
# https://color-hex.org/color/ffeded
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #F8F8F8;">
  <a class="navbar-brand" href="https://digitalgovlab.com" target="_blank">Digital Governance Lab</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
    </ul>
  </div>
</nav>
<style>
    .navbar-brand{
    color: #89949F !important;
     }
    .nav-link disabled{
    color: #89949F !important;
     }
    .nav-link{
    color: #89949F !important;
     }
</style>
""", unsafe_allow_html=True)
#############################################################
#############################################################
##############################################################
#--- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
#########################################################################
# LOGO
#https://pmbaumgartner.github.io/streamlitopedia/sizing-and-images.html
image = Image.open('sentimentai_logo2.jpg')
st.image(image, caption='')
########################
st.markdown(""" <style> .font2 {
     font-size:30px ; font-family: 'Cooper Black'; color: #000000;} 
     </style> """, unsafe_allow_html=True)
#st.markdown('<p class="font2">InnoCase AI</p>', unsafe_allow_html=True) 
#st.markdown(""" <style> .font3 {
#     font-size:20px ; font-family: 'Cooper Black'; color: #000000;} 
#     </style> """, unsafe_allow_html=True)
url = "https://platform.openai.com/account/api-keys"
st.markdown("""
투자감성 AI는 웹크롤링과 생성형 AI 기술을 결합하여 인터넷 정보에 관한 실시간 감성분석을 수행하는 AI입니다.  \
AI의 감성 이해 능력에 대한 연구목적으로 만들어졌으며, 현재는 미국의 7대 첨단기업 관련 주요 뉴스에 대한 분석만 실시하고 있습니다.
""")
#st.markdown("""
#투자감성 AI는 웹크롤링과 생성형 AI 기술을 결합하여 인터넷 정보에 관한 실시간 감성분석을 수행하는 AI입니다.  \
#AI의 감성 이해 능력에 대한 연구목적으로 만들어졌으며, 현재는 미국의 7대 첨단기업 관련 주요 뉴스에 대한 분석만 실시하고 있습니다.  \
#투자감성 AI를 사용하시려면 OpenAI 유료계정과 API Key를 생성하시기 바랍니다. [API Key 생성하러 가기](%s)
#""" % url)
################################
#https://trading-data-analysis.pro/combining-web-scraping-and-prompt-engineered-large-language-models-as-a-stock-headline-monitor-and-2c375ffde576
#openai.api_key = 'XXXX' # REPLACE WITH THIS YOUR KEY
#finnhub_api_key = 'XXXX' # replace this with your key
finnhub_api_key = st.secrets['finnhub_api_key']

st.markdown("---")
#tickers = pd.read_csv('data/nasdaq-listed.csv')
#indicators = tickers.sort_values("tickers").tickers.unique().tolist()
st.markdown("검색할 종목의 ticker를 선택하세요 (애플, MS, 구글, 아마존, 테슬라, 메타, NVDIA 중 택일)")
ticker = st.selectbox(label="XX", label_visibility="collapsed", 
                      options = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"], index = 0)
############################################
############################################
#st.markdown("종목 선택 후, OpenAI API Key를 입력하고 Enter를 눌러주세요 (API Key는 sk-로 시작합니다)")
#openai.api_key = st.text_input(label=" ", label_visibility="collapsed")
###########################
## LOCAL
#import os
#openai_api_key = os.environ.get('openai_api_key')
## STREAMLIT
openai_api_key = st.secrets['openai_api_key']
openai.api_key = openai_api_key
############################################
############################################
st.markdown("---")
st.markdown("##")

if len(openai.api_key):
    res = Request(f'https://finviz.com/quote.ashx?t={ticker}&p=d', headers={'User-Agent': 'Safari/602.1'})
    webpage = urlopen(res).read()
    html = BeautifulSoup(webpage, "html.parser")

    df = pd.read_html(str(html), attrs = {'class': 'fullview-news-outer'})[0]
    df.columns = ['Date', 'Headline']
    for i,r in df.iterrows():
        if len(r.Date)<10:
            r.Date = df.loc[i-1,'Date'].split(' ')[0] + ' ' + r.Date
    df.Date = pd.to_datetime(df.Date)
    ###############################
    from zoneinfo import ZoneInfo
    # 주말이면 3일, 주중이면 1일
    current = dt.datetime.now(ZoneInfo("America/New_York"))
    start_time = dt.datetime.now()-dt.timedelta(days=3)
    #current = dt.datetime.now()
    #if current.weekday() == 0:
    #    start_time = dt.datetime.now()-dt.timedelta(days=3)
    #else:
    #    start_time = dt.datetime.now()-dt.timedelta(days=1)
    ############################################################
    timestamp = dt.datetime(start_time.year,start_time.month,start_time.day,20,0,0)
    latest = df[df.Date>timestamp].set_index('Date').sort_index(ascending=True)
    ###
    def get_completion_from_messages(messages, model="gpt-3.5-turbo-0613", temperature=0):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, # this is the degree of randomness of the model's output
        )
    #     print(str(response.choices[0].message))
        return response.choices[0].message["content"]

    print(latest)
    print(latest.Headline)
    headline = latest.Headline.iloc[0]

    messages =  [  
    {'role':'system', 'content':'You are an analyst of the stock market. Given the headlines of a stock, you rate the overall sentiment of the stock in an integer score between -10 and 10. -10 means very negative. 0 means neutral, 10 means very positive.'},    
    {'role':'user', 'content':headline} ]

    response = get_completion_from_messages(messages, temperature=0)
    #print(headline)
    #print(response)
    ###############################
    x = latest.index.to_numpy()
    y = np.zeros(len(x))
    headlines = ''
    for i in range(len(latest)):
        headlines = headlines + latest.Headline.iloc[i] + '; '
        messages =  [  
    {'role':'system', 'content':'You are an analyst of the stock market. Given the headlines of a stock delimited by \;, you rate the overall sentiment of the stock in an integer score between -10 and 10. -10 means very negative. 0 means neutral, 10 means very positive. Limit your answer to one number. '},
    {'role':'user', 'content':headlines},
                    ]
        response = get_completion_from_messages(messages, temperature=0)
        if len(str(response))>3:
            score = 0
        else:
            score = int(response)
        y[i] = score

    #print(x)
    print(y)

    #############  
    fig, ax = plt.subplots(figsize=(19, 9)) # Create the plot
    plt.style.use('seaborn-darkgrid') # Set the style
    #ax.plot(x, y, color='steelblue', linewidth=3, linestyle='--') # Plot the data
    ax.plot(x, y, color='midnightblue', linewidth=4) # Plot the data

    # Format the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.tick_params(axis='x', rotation=30, labelsize=20)
    ax.set_ylabel('News Sentiment Score', fontsize=25) # Format the y-axis
    ax.set_title(f'AI Sentiment Analysis for {ticker}', fontsize=26)
    ax.grid(color='lightgray', linestyle='--') # Customize the gridlines
    fig.patch.set_facecolor('whitesmoke') # Customize the background color
    ax.set_ylim(-10, 10) # Set the y-axis limits
    st.pyplot(fig)


