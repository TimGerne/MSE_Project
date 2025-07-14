import streamlit as st
from streamlit_js_eval import streamlit_js_eval, get_geolocation
from streamlit_extras.stylable_container import stylable_container

import pandas as pd
import numpy as np
import time
import io

import streamlit as st
from st_screen_stats import ScreenData, StreamlitNativeWidgetScreen, WindowQuerySize, WindowQueryHelper

MOBILE_THRESHOLD_PX = 76  # e.g., treat widths < 76px as mobile
# Initialize the screen stats component (with a 1-second delay for initial data)
screen_data = ScreenData(setTimeout=300)  
screen_stats = screen_data.st_screen_data()  # Get a dict of screen and window dimensions
window_width = screen_stats["innerWidth"]


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


def load_session(current_session:int):
    st.session_state.current_session = current_session
    st.rerun()


def create_sidebar(history):
    if not history:
        st.write("No history")
    else:
        for i,session in enumerate(history):
            my_style = """
                button {
                    background-color: #262730;
                    color: white;
                    border-radius: 0px;
                }
                button[data-testid="stPopoverButton"] {
                    border: None !important;
                }
                div[data-testid="st] div[data-testid="stHorizontalBlock"] div[data-testid="stTextInput"] input {
                    color: white;
                }
                """

            with stylable_container(
                key=f"style_container{i}",
                css_styles=my_style,
            ):
                with st.container(border=True):
                    if st.session_state.current_session==i:
                        st.badge("active", color="green")
                    left_sb,middle_sb,right_sb = st.columns([2,5,2])
                    session_queries = session[0]
                    session_name = session[2]
                    
                    #name session
                    st.session_state.history[i][2] = middle_sb.text_input("Sessioname_text{i}",session_name,label_visibility="collapsed")
                    session_name = st.session_state.history[i][2]
                    #load session
                    if left_sb.button("→",type="tertiary",key=f"Session_load_{i}",use_container_width=True):
                        load_session(i)
                    #shows all queries
                    popover = right_sb.popover(label='')
                    for j,session_query in enumerate(session_queries):
                        popover.text(session_query)
                    


##Initialization
st.set_page_config(page_title="Search",page_icon="🔎",layout="wide")
if "ndocs" not in st.session_state:
    st.session_state.ndocs=10000 #TODO Look for documents
st.session_state.n_results = 100
if "current_session" not in st.session_state:
    st.session_state.current_session =None
if "history" not in st.session_state:
    st.session_state.history = []


queries_file=None
query=None

with st.sidebar:
    st.header("History",width="stretch",divider="grey")
    create_sidebar(st.session_state.history)
    st.html("<hr>")
    right_sidebar,middle_sidebar,left_sidebar =  st.columns([8,2,8])

if window_width<MOBILE_THRESHOLD_PX:
    #activate mobile view
    st.write(window_width)
else:
    #activate desktop view
    st.title("Our search engine")

    st.markdown(f':orange-badge[{st.session_state.ndocs} documents to search]')

    with st.form("search_form"):
        queries_file = st.file_uploader("Upload Queries via file 📁",help="Upload tab separated format",accept_multiple_files=False) 
        query = st.text_input("or put your Query here 👇",placeholder="What are you looking for?",help="Accept only one lined query")
        
        #Search code
        submitted = st.form_submit_button('Start Search 🔎')#,on_click=start_searching)
        if submitted:
            queries = get_queries(query,queries_file)
            docs = start_searching(query,queries_file)
            docs = example_retrieved_docs #TODO: remove when search implemented
            
            st.session_state.current_session = len(st.session_state.history)
            session_name = f"Session {st.session_state.current_session}"
            st.session_state.history.append([queries,docs,session_name])
            st.rerun()


    st.html("<hr>")
    
    st.markdown("### Search results")

    if len(st.session_state.history)>0: #check if already searched

        current_session = st.session_state.current_session
        current_queries,current_docs,current_session_name = st.session_state.history[current_session]
        
        left,right = st.columns([6,1])
        with left:
            st.markdown(f"## {current_session_name}")
        with right:
            st.session_state.n_results = st.number_input("Number of results",min_value=1,value=100,label_visibility="collapsed",help="Number of results shown")

        query_tabs_name = [f"Results {i}" for i in range(len(current_queries))]
        query_tabs = st.tabs(query_tabs_name)
        for i,(query,tab) in enumerate(zip(current_queries,query_tabs)):
            with tab:
                example_retrieved_docs = current_docs
                ls,rs = st.columns([7,1])
                ls.markdown(f"##### Query: {query}")
                rs.download_button("📥",example_retrieved_docs[example_retrieved_docs["Query"]==i].to_csv(index=False,header=False,sep='\t'),file_name=f"results_query_{query}.tsc")
                retrieved_docs = example_retrieved_docs.loc[example_retrieved_docs["Query"]==i].iloc[:,1:]
                shown_docs = retrieved_docs.iloc[:st.session_state.n_results,:]
                
                #Show results as Dataframe
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

        #@st.cache_resource for ML resources #https://docs.streamlit.io/get-started/fundamentals/advanced-concepts #DB connection #st.metric("Some Value",42,2)
    else:
        st.markdown("## No results")
