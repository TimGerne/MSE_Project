import streamlit as st
from streamlit_js_eval import streamlit_js_eval, get_geolocation
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.bottom_container import bottom
from streamlit_extras.word_importances import format_word_importances
import itertools

import pandas as pd
import numpy as np
import time
import io
import random
import requests
from bs4 import BeautifulSoup
import cohere
from transformers import pipeline
import fasttext


import streamlit as st
from st_screen_stats import ScreenData, StreamlitNativeWidgetScreen, WindowQuerySize, WindowQueryHelper


#Examples inputs################
example_retrieved_docs = pd.DataFrame({"Query":[0,0,0,0,1,1,1,1],
                                       "Rank":[1,2,3,4,1,2,3,4],
                                       "URL":["https://www.bbc.com","https://www.n24.de","https://www.tagesschau.de","https://www.n-tv.de","https://www.ard.de","https://www.zdf.de","https://www.youtube.com","https://www.netflix.com"],
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
    

def start_search(queries:list):
    pass

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
        queries = queries[:-1]
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
                    #if st.session_state.current_session==i:
                    #    st.badge("active", color="green")
                    left_sb,middle_sb,right_sb = st.columns([2,5,2])
                    session_queries = session[0]
                    session_name = session[2]
                    
                    if st.session_state.current_session==i:
                        #st.badge("active", color="green")
                        current_session_icon = ":material/my_location:"
                    else:
                        current_session_icon = None
                    #name session
                    st.session_state.history[i][2] = middle_sb.text_input("Sessioname_text{i}",session_name,icon=current_session_icon,label_visibility="collapsed")#,icon=":material/edit:"
                    session_name = st.session_state.history[i][2]
                    #load session
                    if left_sb.button("→",type="tertiary",key=f"Session_load_{i}",use_container_width=True):
                        load_session(i)
                    #shows all queries
                    popover = right_sb.popover(label='')
                    for j,session_query in enumerate(session_queries):
                        popover.text(session_query)
                    

def predict_text_categories(url,k=3):
    #not used might be interesting for labels
    content = extract_text_from_url(url)
    model = fasttext.load_model("model.bin")
    pred = model.predict(content,k=k)
    return pred


def LLM_website_summary(content):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(content,max_length=150,min_length=50,do_sample=False)
    return summary[0]["summary_text"]


def get_website_summary(url:str):
    content = extract_text_from_url(url)
    return LLM_website_summary(content)


@st.dialog("Website Summary")
def get_summary(url:str):
    st.title(url)
    summary = get_website_summary(url)
    st.write(summary)


def talk_as_stream(response:str):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def LLM_query_refinement(system_prompt,prev_messages,use_cohere=False):
    "normal LLM"
    system_message = {"role":"system","content": system_prompt}
    if use_cohere:
        co = cohere.ClientV2("cohere_KEY")#TODO: geheimnis nacher wieder masken
        #response = co.chat(
        #    model="command-a-03-2025", 
        #    messages=system_message + prev_messages
        #)
        response = co.chat_stream(
            model="command-a-03-2025", 
            messages=system_message + prev_messages
        )

        return response.message.content[0].text
    else:
        return talk_as_stream("Activate Cohere")


def extract_meta_description(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Look for the <meta name="description"> tag
    meta_description = soup.find('meta', attrs={'name': 'description'})
    
    if meta_description:
        # If found, return the content of the meta description
        return meta_description.get('content')
    else:
        # If no meta description is found, return a message
        return "No meta description found."


def LLM_website_recommendation(system_prompt,prev_messages):
    """This is RAG based"""

    if len(st.session_state.history)<1:
        return talk_as_stream("Search first")
    urls = st.session_state.history[st.session_state.current_session][st.session_state.history[st.session_state.current_session]['Query']==0]["URL"].tolist()
    descriptions = [extract_meta_description(url) for url in urls]
    docs = [{"data":{"text": content, "url":url}} for url,content in zip(urls,descriptions)]

    system_message = {"role":"system","content": system_prompt}
    co = cohere.ClientV2("cohere_KEY")
    response = co.chat_stream(
        model="command-a-03-2025",
        messages=system_message +prev_messages,
        documents=docs)
    return response.message.content[0].text


def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for tag in soup(['script', 'style']):
            tag.decompose()

        # Get visible text
        text = soup.get_text(separator='\n')
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = '\n'.join(line for line in lines if line)

        return text

    except Exception as e:
        return f"Error: {e}"


def conversate_with_LLM(mode):
    messages = st.session_state.messages
    general_system_prompt = "You are a knowledgeable and friendly assistant that helps users with technical questions. Respond professionally, with warmth and clarity, avoiding robotic or overly formal language. Never use emojis in your responses. If you do not know the answer to a question, say so honestly, do not invent facts."
    system_prompt_query_refinement = "Your primary task is to analyze user queries and identify any arbitrary, vague, or potentially misleading terms that could confuse search engines or cause unrelated websites to rank highly. Once you highlight these terms, wait for the user's clarification. After that, suggest improved versions of the query that reduce ambiguity and improve search precision—based on the user's intent."
    system_prompt_website_recommendation = "Your primary task is to help the user find the most suitable website from a curated list of 100 options. Based on the user's query, identify and recommend the best-matching website from this set. If the query is ambiguous or missing key details, politely ask follow-up questions to clarify the user's intent. Once clarified, explain why a specific website fits the user's needs and offer one or more relevant alternatives if appropriate."
    system_prompt = general_system_prompt
    if mode == "Query refinement":
        system_prompt += system_prompt_query_refinement
        return talk_as_stream(f"dddddddddihey {len(st.session_state.messages)}")
        return LLM_query_refinement(system_prompt = system_prompt, prev_messages=messages)
    if mode == "Query recomendation":
        system_prompt += system_prompt_website_recommendation
        return talk_as_stream(f"hihey {len(st.session_state.messages)}")
        return LLM_website_recommendation(system_prompt = system_prompt, prev_messages=messages)


def begin_conversation(mode:str):
    first_message_query_refinement="Please insert the query that you want to refine."
    first_message_website_recommendation = "Give me more context what you are looking for."
    n_messages = len(st.session_state.messages)
    if n_messages ==0:
        if mode == "Query refinement":
            return talk_as_stream(first_message_query_refinement)
        if mode == "Website recommendation":
            return talk_as_stream(first_message_website_recommendation)
    else:
        return conversate_with_LLM(mode)


def compute_all_subsets_of_query(query:list):
    n = len(query)
    subsets = []
    for r in range(n + 1):
        subsets.extend(itertools.combinations(query, r))
    return subsets[1:]


def compute_shapley_values(query:list,original_urls:str)->list:
    subsets = compute_all_subsets_of_query(query)
    computable_subset = subsets
    if len(subsets)>30: #limit to a sensible size
        computable_subset = random.sample(subsets,30)
    new_queries = [" ".join(query_tuple) for query_tuple in computable_subset]
    shap_docs = start_search(new_queries)
    
    fraction_same_docs = []
    for i in range(len(computable_subset)):
        new_urls = shap_docs[shap_docs["Query"]==i]["URL"]
        cut = set(original_urls) & set(new_urls)
        fraction_same_docs.append(len(cut)/100)
    
    in_subset = np.zeros((len(query),len(fraction_same_docs)),dtype=bool)
    fraction = np.zeros((len(query),len(fraction_same_docs)),dtype=bool)
    for i,element_query in range(query):
        for j,new_query in enumerate(computable_subset):
            if element_query in new_query:
                in_subset[i][j]=True
                fraction[i][j] = fraction_same_docs[j]
    amount_in_subset = in_subset.sum(axis=1)
    amount_in_subset[amount_in_subset==0] = 1
    shap_vals = fraction.sum(axis=1)/amount_in_subset
    return tuple(shap_vals.tolist())


#logic for popover button with agent
def assistant():
    with st.popover("",icon=":material/smart_toy:",use_container_width=True):
        left_side,right_side = st.columns([5,1])
        if right_side.button("",key="delete_chat",icon=":material/delete:",type="tertiary"):
            st.session_state.messages=[]
            st.rerun()
        left_side.markdown("##### What can i help you with?")
        mode = st.pills("Mode",["Query refinement","Website recommendation"],default=st.session_state.chatbot_mode,label_visibility="collapsed")
        if mode != st.session_state.chatbot_mode:
            st.session_state.messages=[]
        st.session_state.chatbot_mode = mode

        with st.container(height=150,border=False):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            if len(st.session_state.messages) % 2 ==0:
                with st.chat_message("assistant"):
                    response = st.write_stream(begin_conversation(st.session_state.chatbot_mode))
                st.session_state.messages.append({"role": "assistant", "content": response})

        prompt = st.chat_input("Get assistance")

        if prompt:
            st.session_state.messages.append({"role":"user","content": prompt})
            st.rerun()


##Initialization
st.set_page_config(page_title="Search",page_icon="🔎",layout="wide")
if "ndocs" not in st.session_state:
    st.session_state.ndocs=10000 #Look for documents
st.session_state.n_results = 100
if "current_session" not in st.session_state:
    st.session_state.current_session =None
if "history" not in st.session_state:
    st.session_state.history = []
if "highlight_clicked" not in st.session_state:
    st.session_state.highlight_clicked = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chatbot_mode" not in st.session_state:
    st.session_state.chatbot_mode = "Query refinement"


queries_file=None
query=None

with st.sidebar:
    with st.container():
        left_sb, right_sb = st.columns(2)

        left_sb.container(height=20,border=False)
        left_sb.markdown("## History",width="stretch")#,divider="grey")
        #left_sb.button("# History",type="tertiary",use_container_width=True)
    
        right_sb.container(height=28,border=False)
        if right_sb.button("❤️",type="tertiary",use_container_width=True):
            for i in range(15):
                st.toast("We love you ❤️")
    st.html("<hr>")
    create_sidebar(st.session_state.history)
    st.html("<hr>")
    
    
    left_sb,right_sb = st.columns(2)
    
    left_sb.link_button("","https://github.com/TimGerne/MSE_Project",icon=":material/school:",type="secondary",use_container_width=True)
    with right_sb:
        assistant()


st.title("Our search engine")

st.markdown(f':orange-badge[{st.session_state.ndocs} documents to search]')

with st.form("search_form"):
    queries_file = st.file_uploader("Upload Queries via file 📁",help="Upload tab separated format",accept_multiple_files=False) 
    query = st.text_input("or put your Query here 👇",placeholder="What are you looking for?",help="Accept only one lined query")
    
    #Search code
    submitted = st.form_submit_button('Start Search 🔎')#,on_click=start_searching)
    if submitted:
        queries = get_queries(query,queries_file)
        #st.session_state.queries=queries
        docs = start_searching(query,queries_file)
        docs = example_retrieved_docs #TODO: remove when search implemented
        #st.session_state.pd_docs = docs
        st.session_state.current_session = len(st.session_state.history)
        session_name = f"{st.session_state.current_session +1}. Session"
        st.session_state.history.append([queries,docs,session_name])
        st.rerun()


st.html("<hr>")

left,right = st.columns([6,1])
with left:
    st.markdown("### Search results")
with right:
    popover = st.popover("",icon=":material/settings:")
    st.session_state.n_results = popover.number_input("number of results",min_value=1,value=20,label_visibility="visible",help="Number of results shown")
    left_side,right_side = popover.columns([2,1])
    #st.session_state.show_tags = left_side.checkbox("Generate Tags",help="Crawls the websites to generate small texts")
    if right_side.button(label="",icon=":material/refresh:",key="Rerun_button",type="tertiary"):
        st.rerun()


if len(st.session_state.history)>0: #check if already searched

    current_session = st.session_state.current_session
    current_queries,current_docs,current_session_name = st.session_state.history[current_session]
    
    left_side,right_side = st.columns([5,1])
    left_side.markdown(f"## {current_session_name}")
    
    
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
            
            left_side,middle_side,right_side = tab.columns([1,1,1])
            for j,element in shown_docs.iterrows():
                rank = element["Rank"]
                url = element["URL"]
                rel = element["Relevance"]
                left_side.markdown(f"{rank}. [{url}]({url})")
                if right_side.button("Summary",key=f"summary_button{j}",icon=":material/summarize:"):
                    get_summary(url)
                
                
                #with right_side.expander("Show page summary") as exp:
                #    if exp:
                #        st.write("so cool")

            #Show results as Dataframe
            #tab.dataframe(
            #    shown_docs,
            #    column_config={
            #        "Rank": st.column_config.NumberColumn("Rank",width="small"),
            #        "URL": st.column_config.LinkColumn("URL",width="mid"),
            #        "Relevance": st.column_config.NumberColumn("Relevance",width="small")
            #    },
            #    hide_index=True,
            #)
        left_side,right_side = st.columns(2)
        if right_side.button("Compute Query importance",icon=":material/memory:"):
            st.write(st.session_state.history)
            docs = st.session_state.history[st.session_state.current_session][1]
            query = st.session_state.history[st.session_state.current_session][0][0]
            ori_urls = docs[docs["Query"]==0]["URL"]
            shap_val = compute_shapley_values(query.split(),original_urls=ori_urls)
            html = format_word_importances(
                words=query.split(),
                importances=shap_val)
            st.write(html, unsafe_allow_html=True)



    left_side.download_button('Download all results', example_retrieved_docs.to_csv(index=False,header=False,sep='\t'),icon=":material/ad:",file_name="Results.tsv")

    #@st.cache_resource for ML resources #https://docs.streamlit.io/get-started/fundamentals/advanced-concepts #DB connection #st.metric("Some Value",42,2)
#streamlit extras grid for layout #stateful_chats for conversation with assistant #pdf viewer #rowlayout
else:
    st.markdown("## No results")
    

#with bottom():
