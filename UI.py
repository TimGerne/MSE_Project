import streamlit as st
from streamlit_js_eval import streamlit_js_eval, get_geolocation
import pandas as pd
import numpy as np
import time
import io

#Examples inputs################
example_retrieved_docs = pd.DataFrame({"Query":[0,0,0,0,1,1,1,1],
                                       "Rank":[1,2,3,4,1,2,3,4],
                                       "URL":["www.bbc.com","www.n24.de","www.tagesschau.de","www.n-tv.de","www.ard.de","ww.zdf.de","www.youtube.com","www.netflix.com"],
                                       "Relevance":[1,0.5,0.3,0.1,1,0.5,0.3,0.1]})
example_queries = ["Tübingen","Food"]
#################################

def retrieve_queries(query:str,query_file)->list:
    queries = []
    if query:
        queries.append(query)
    if query_file:
        query_file_content = query_file.read().decode("utf-8")
        lines = query_file_content.splitlines()
        queries.append(lines)
    return queries
    

def start_searching(query,queries)->list:
    if queries or query:
        queries = retrieve_queries(query,queries)
        st.toast("Started Search")
        with st.spinner(text="Search in Progress"):
            time.sleep(2)
            #docs = start_search(queries)
            docs=None
            #TODO: Replace by actual start search call
            st.success("✅ Retrieved Documents")
            return docs
    else:
        st.error("❌ Search failed: no query")
        return None


def get_queries(query:str|None,queries_file:st.runtime.uploaded_file_manager.UploadedFile)->list:
    if queries_file:
        stringio = io.StringIO(queries_file.getvalue().decode("utf-8"))
        queries = stringio.read()
        queries = queries.replace("\r","\n")
        queries = queries.split("\n")
        return queries
    elif query:
        return [query]
    else:
        return None


st.set_page_config(page_title="Search",page_icon="🔎",layout="wide")


queries_file=None
query=None
queries=None

#st.sidebar.success("Select a demo above.")
st.title("Our search engine")
st.session_state.queries = None

if "has_searched" not in st.session_state:
    st.session_state.has_searched=False
if "ndocs" not in st.session_state:
    st.session_state.ndocs=10000 #Look for documents

st.markdown(f':orange-badge[{st.session_state.ndocs} documents to search]')

with st.form("search_form"):
    queries_file = st.file_uploader("Upload Queries via file 📁",help="Upload tab separated format",accept_multiple_files=False) 
    query = st.text_input("or put your Query here 👇",placeholder="What are you looking for?",help="Accept only one lined query")
    
#Search

    submitted = st.form_submit_button('Start Search 🔎')#,on_click=start_searching)
    if submitted:
        queries = get_queries(query,queries_file)
        st.session_state.queries=queries
        docs = start_searching(query,queries_file)
        docs = example_retrieved_docs#TODO: remove
        st.session_state.docs = docs


#queries = example_queries#

left,right = st.columns([6,1])
with left:
    st.markdown("### Search results")

if queries:
    with right:
        n_results_to_show = st.number_input("Number of results",min_value=1,value=100,label_visibility="visible")

    query_tabs_name = [f"Results {i}" for i in range(len(queries))]
    query_tabs = st.tabs(query_tabs_name)
    for i,(query,tab) in enumerate(zip(queries,query_tabs)):
        with tab:
            example_retrieved_docs = st.session_state.docs
            ls,rs = st.columns([7,1])
            ls.markdown(f"##### Query: {query}")
            rs.download_button("Download",example_retrieved_docs[example_retrieved_docs["Query"]==i].to_csv(index=False,header=False,sep='\t'),file_name=f"results_query_{query}.tsc")
            retrieved_docs = example_retrieved_docs.loc[example_retrieved_docs["Query"]==i].iloc[:,1:]
            shown_docs = retrieved_docs.iloc[:n_results_to_show,:]
            #tab.dataframe(retrieved_docs)
            tab.dataframe(
                shown_docs,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank",width="small"),
                    "URL": st.column_config.LinkColumn("URL",width="mid"),
                    "Relevance": st.column_config.NumberColumn("Relevance",width="small")
                },
                hide_index=True,
            )

    st.download_button('Download all results', example_retrieved_docs.to_csv(index=False,header=False,sep='\t'),file_name="Results.tsv")

    #@st.cache_resource for ML resources
    #https://docs.streamlit.io/get-started/fundamentals/advanced-concepts #DB connection
    #st.html("<p>Foo bar <p>")
    #st.metric("Some Value",42,2)
    #st.table() might be helpful for results.
else:
    st.markdown("## No results")

screen_info = streamlit_js_eval(js_expressions="screen.width + ',' + screen.height", key="get_screen_size")

if screen_info:
    width, height = map(int, screen_info.split(","))
    st.success(f"Screen size: {width} × {height}")
