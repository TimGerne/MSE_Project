import streamlit as st
import pandas as pd
import numpy as np
import time

#Examples inputs################
example_retrieved_docs = pd.DataFrame({"Query":[0,0,0,0,1,1,1,1],
                                       "Rank":[1,2,3,4,1,2,3,4],
                                       "URL":["www.bbc.com","www.n24.de","www.tagesschau.de","www.n-tv.de","www.ard.de","ww.zdf.de","www.youtube.com","www.netflix.com"],
                                       "Relevance":[1,0.5,0.3,0.1,1,0.5,0.3,0.1]})
example_queries = ["Tübingen","Food"]
#################################


def start_crawl():
    st.session_state.crawling = True
    st.session_state.has_crawled = True
    #TODO:add more function that actually starts crawler.py

def stop_crawl():
    st.session_state.crawling = False
    #TODO add pause function for crawler.py

def start_searching():
    with st.spinner(text="Search in Progress"):
        time.sleep(2)
        #TODO: Replace by actual start search call
        st.success("Retrieved Documents")

st.set_page_config(layout="wide")
st.title("Our search engine")
queries= ["one","two"]

if "has_crawled" not in st.session_state:
    st.session_state.has_crawled=False
if "crawling" not in st.session_state:
    st.session_state.crawling=False
if "has_searched" not in st.session_state:
    st.session_state.has_searched=False
if "ndocs" not in st.session_state:
    st.session_state.ndocs=0


left_column, right_column = st.columns(2)

with left_column:
    if st.session_state.crawling:
        st.button('Pause Crawling',on_click=stop_crawl)
    elif st.session_state.has_crawled:
        st.button('Resume Crawling',on_click=start_crawl)
    else:
        st.button('Start Crawling',on_click=start_crawl)

    #on-click activation
    latest_iteration = st.empty()
    crawl_bar = st.progress(0)
    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Percentage of Crawl {i+1}%')
        crawl_bar.progress(i + 1)
        time.sleep(0.01)

    st.markdown(f':green-badge[Crawling completed] :orange-badge[{st.session_state.ndocs} documents]')
    
    st.markdown("Insert your queries here:")
    text_side, upload_side = st.columns([3,1])
    with text_side:
        st.text_area("Your Queries here",help="Insert one or more queries here with line breaks")
    with upload_side:
        st.file_uploader("Upload Query",help="Upload Queryfile")

    #Search    
    st.button('Start Search',disabled=st.session_state.has_crawled,on_click=start_searching)




retrieved_documents=None

with right_column:

    
    queries = example_queries#
    
    st.subheader("Search results")

    query_tabs_name = [f"Query {i}" for i in range(len(queries))]
    query_tabs = st.tabs(query_tabs_name)
    for i,(query,tab) in enumerate(zip(queries,query_tabs)):
        tab.write(f"Query: {query}")
        retrieved_docs = example_retrieved_docs.loc[example_retrieved_docs["Query"]==i]
        #tab.dataframe(retrieved_docs)
        tab.data_editor(
            retrieved_docs,
            column_config={
                "URL": st.column_config.LinkColumn("URL")
            },
            hide_index=True,
        )

    st.download_button('download search results', example_retrieved_docs.to_csv(index=False,header=False,sep='\t'))

#@st.cache_resource for ML resources
#https://docs.streamlit.io/get-started/fundamentals/advanced-concepts #DB connection