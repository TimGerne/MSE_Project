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


def retrieve_queries(query:str,query_file)->list:
    queries = []
    if query:
        queries.append(query)
    if query_file:
        query_file_content = query_file.read().decode("utf-8")
        lines = query_file_content.splitlines()
        queries.append(lines)
    return queries
    

def start_searching(query,queries):
    retrieve_queries(query,queries)
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

    st.subheader("Crawling")

    button_side,time_side = st.columns([5,2])
    with button_side:    
        if st.session_state.crawling:
            st.button('Pause Crawling',on_click=stop_crawl)
        elif st.session_state.has_crawled:
            st.button('Resume Crawling',on_click=start_crawl)
        else:
            st.button('Start Crawling',on_click=start_crawl)
    with time_side:
        d=0

    #on-click activation
    latest_iteration = st.empty()
    crawl_bar = st.progress(0)
    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Percentage of Crawl {i+1}%')
        crawl_bar.progress(i + 1)
        time.sleep(0.01)

    st.markdown(f':green-badge[Crawling completed] :orange-badge[{st.session_state.ndocs} documents]')
    
    
    st.subheader("Search")

    with st.form("search_form"):
        queries_file = st.file_uploader("Upload Queries via file 📁",help="Upload tab separated format") 
        query = st.text_input("or put your Query here 👇",placeholder="What are you looking for?",help="Accept only one lined query")
            
    #Search    
        submitted = st.form_submit_button('Start Search 🔎',disabled=st.session_state.has_crawled,on_click=start_searching)
        if submitted:
            start_searching(query,queries_file)





retrieved_documents=None

with right_column:

    
    queries = example_queries#
    
    st.subheader("Search results")

    query_tabs_name = [f"Results {i}" for i in range(len(queries))]
    query_tabs = st.tabs(query_tabs_name)
    for i,(query,tab) in enumerate(zip(queries,query_tabs)):
        tab.write(f"Query: {query}")
        retrieved_docs = example_retrieved_docs.loc[example_retrieved_docs["Query"]==i].iloc[:,1:]
        #tab.dataframe(retrieved_docs)
        tab.dataframe(
            retrieved_docs,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank",width="small"),
                "URL": st.column_config.LinkColumn("URL",width="mid"),
                "Relevance": st.column_config.NumberColumn("Relevance",width="small")
            },
            hide_index=True,
        )

    st.download_button('download search results', example_retrieved_docs.to_csv(index=False,header=False,sep='\t'),file_name="Results.tsv")

#@st.cache_resource for ML resources
#https://docs.streamlit.io/get-started/fundamentals/advanced-concepts #DB connection