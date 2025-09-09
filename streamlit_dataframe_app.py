
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType,Tool
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

import warnings
warnings.simplefilter('ignore')

df = pd.read_csv('data/CCTV_in_Seoul.csv')
gpt = ChatOpenAI( model='gpt-4o-mini', temperature=0)
agent = create_pandas_dataframe_agent( gpt, df, allow_dangerous_code=True,
                                       verbose=True,handle_parsing_error=True)

st.dataframe( df, use_container_width=True)
st.title('자연어로 데이터 분석하기')

query = st.text_input('데이터분석 명령어를 입력하시요',placeholder="데이터분석")
btn = st.button('실행')
if btn and query:
    with st.spinner("데이터 분석중입니다. ..."):
        try:
            result = agent.run( query)
            if plt.get_fignums(): #차트가생성된경우
                st.pyplot( plt )
            else:
                st.write( result )
                st.success('분석완료')

        except Exception as err:
            st.error(f'분석에러:{err}')

