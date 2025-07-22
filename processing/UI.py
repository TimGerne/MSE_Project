import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.bottom_container import bottom
from streamlit_extras.word_importances import format_word_importances
import itertools
from pathlib import Path

#from processing.query_expansion import expand_query_with_prf, expand_query_with_synonyms, expand_query_with_filtered_synonyms, expand_query_with_glove
from models import DenseRetrievalModel,BM25RetrievalModel,HybridReciprocalRankFusionModel,HybridAlphaModel,load_faiss_and_mapping

import pandas as pd
import numpy as np
import time
import io
import random
import requests
from bs4 import BeautifulSoup
from yake import KeywordExtractor
import cohere
from transformers import pipeline
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import json

def retrieve_queries(query:str,query_file)->list:
    queries = []
    if query:
        queries.append(query)
    if query_file:
        query_file_content = query_file.read().decode("utf-8")
        lines = query_file_content.splitlines()
        queries.append(lines)
    return queries
    

#@st.cache_resource
def instantiate_retrieval_model(use_expansion=False):

    faiss_index = faiss.read_index("../indexing/indexing/output/semantic_index.faiss")
    with open("../indexing/indexing/output/doc_mapping.json", "r", encoding="utf-8") as f:
        doc_mapping = json.load(f)
    
    texts = [doc["title"] for doc in doc_mapping.values()]
    urls = [doc["url"] for doc in doc_mapping.values()]
    bm25 = BM25RetrievalModel(texts, urls, use_expansion=use_expansion)
    dense = DenseRetrievalModel(faiss_index, doc_mapping, use_expansion=use_expansion)
    model = HybridReciprocalRankFusionModel(bm25, dense, doc_mapping=doc_mapping)
    return model

def start_search(queries:list,retriever,top_k=100):
    records = []
    for i, query in enumerate(queries):
        results = retriever.retrieve(query, top_k=top_k)
        for rank, item in enumerate(results, 1):
            records.append({
                "Query": i,
                "Rank": rank,
                "URL": item["url"],
                "Relevance": item.get("score", 0.0)
            })
    return pd.DataFrame.from_records(records)

def start_searching(query,queries,retriever)->list:
    if queries or query:
        queries = retrieve_queries(query,queries)
        st.toast("Started Search")
        with st.spinner(text="Search in Progress"):
            docs = start_search(queries,retriever)
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
    st.session_state.i_session = current_session
    st.session_state.i_query = 0
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
                    
                    if st.session_state.i_session==i:
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
                    

@st.cache_data
def get_keywords(meta:str,get_unique_keywords:bool):
    #get keywords first version non unique only keywords
    keywords_list = []
    if get_unique_keywords:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(metas)
        feature_names = np.array(vectorizer.get_feature_names_out())
        for doc_idx in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[doc_idx].toarray().flatten()
            top_indices = row.argsort()[::-1][:3]
            top_words = feature_names[top_indices]
            keywords_list.append(top_words.tolist())

    else:
        kw_extractor = KeywordExtractor(lan="en", n=1, top=3)
        for meta in metas:
            keywords = [x[0] for x in kw_extractor.extract_keywords(meta)]
            keywords_list.append(keywords)
    return keywords_list


def get_color_keyword_dict(keywords:list)->dict:
    keywords = set(keywords)
    color_keyword_dict = dict()
    colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4",
        "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff",
        "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
        "#000080", "#808080", "#FFFFFF", "#000000", "#a9a9a9", "#ff69b4",
        "#b0e0e6", "#98fb98", "#dda0dd", "#fa8072", "#20b2aa", "#f0e68c"
    ]
    for i,keyword in enumerate(keywords): 
        color_keyword_dict[keyword] = colors[i%len(colors)]
    return color_keyword_dict



##Code for heavy compute but high qulaity summary with huggingface run client side
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6") #mt5-small

def get_summary(content:str)->str:
    st.write(len(content))
    min_text_len = 20
    if len(content)<min_text_len:
        return content
    summarizer = load_summarizer()
    summary = summarizer(content,max_length=150,min_length=min_text_len,truncation=True,do_sample=False)
    return summary[0]["summary_text"]
###############################################################################################

#more lightweight but needs api
def get_summary_cohere(content:str)->str:
    co = cohere.Client(st.secrets["COHERE_API_KEY"])
    response = co.generate(
        model="command",
        prompt=f"Summarize the following website:n\n{content}",
        max_tokens= 150,
        temperature=0.3,
    )
    return response.generations[0].text.strip()

###############################################################################################
@st.dialog("Website Summary")
def get_summary(url:str,use_cohere=True):
    st.title(url)
    content = extract_text_from_url(url)
    if use_cohere:
        #summary = url
        summary = get_summary_cohere(content)
    else:
        summary = get_summary(content)
    st.write(summary)


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

@st.cache_data
def extract_metas_description(urls):

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
    return [extract_meta_description(url) for url in urls]


def talk_as_stream(response:str):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def LLM_query_refinement(system_prompt,prev_messages,use_cohere=True):
    system_message = [{"role":"system","content": system_prompt}]
    messages = system_message + prev_messages
    if use_cohere:
        co = cohere.ClientV2(st.secrets["COHERE_API_KEY"])
        response = co.chat_stream(
            model="command-a-03-2025", 
            messages=messages
        )

        for event in response:
            if event and event.type == "content-delta":
                yield event.delta.message.content.text
    else:
        yield from talk_as_stream("Activate Cohere")


def LLM_website_recommendation(system_prompt,prev_messages,use_cohere=True):
    """This is RAG based"""
    if st.session_state.history == []:
        print("This case")
        yield from talk_as_stream("Please search first")
    else:
        docs = st.session_state.history[st.session_state.i_session][1]
        urls = docs[docs["Query"]==0]["URL"].tolist()
        descriptions = extract_metas_description(urls)
        docs = [{"data":{"text": content, "url":url}} for url,content in zip(urls,descriptions)]
        system_message = [{"role":"system","content": system_prompt}]
        if use_cohere:
            co = cohere.ClientV2(st.secrets["COHERE_API_KEY"])
            response = co.chat_stream(
                model="command-a-03-2025",
                messages=system_message +prev_messages,
                documents=docs)
            
            for event in response:
                    if event and event.type == "content-delta":
                        yield event.delta.message.content.text
        else:
            yield from talk_as_stream("Activate Cohere")


def conversate_with_LLM(mode):
    messages = st.session_state.messages
    general_system_prompt = "You are a knowledgeable and friendly assistant that helps users with technical questions. Respond professionally, with warmth and clarity, avoiding robotic or overly formal language. Never use emojis in your responses. If you do not know the answer to a question, say so honestly, do not invent facts. Answer short but precise."
    system_prompt_query_refinement = "Your primary task is to analyze user queries and identify any arbitrary, vague, or potentially misleading terms that could confuse search engines or cause unrelated websites to rank highly. Once you highlight these terms, wait for the user's clarification. After that, suggest improved versions of the query that reduce ambiguity and improve search precision—based on the user's intent."
    system_prompt_website_recommendation = "Your primary task is to help the user find the most suitable website from a curated list of 100 options. Based on the user's query, identify and recommend the best-matching website from this set. If the query is ambiguous or missing key details, politely ask follow-up questions to clarify the user's intent. Once clarified, explain shortly why a specific website fits the user's needs and offer one or more relevant alternatives if appropriate."
    system_prompt = general_system_prompt
    if mode == "Query refinement":
        system_prompt += system_prompt_query_refinement
        yield from LLM_query_refinement(system_prompt = system_prompt, prev_messages=messages)
    if mode == "Website recommendation":
        system_prompt += system_prompt_website_recommendation
        yield from LLM_website_recommendation(system_prompt = system_prompt, prev_messages=messages)


def begin_conversation(mode:str):
    first_message_query_refinement="Please insert the query that you want to refine."
    first_message_website_recommendation = "Give me more context what you are looking for."
    n_messages = len(st.session_state.messages)
    if n_messages ==0:
        if mode == "Query refinement":
            yield from talk_as_stream(first_message_query_refinement)
        if mode == "Website recommendation":
            yield from talk_as_stream(first_message_website_recommendation)
    else:
        yield from conversate_with_LLM(mode)


def compute_all_subsets_of_query(query:list):
    n = len(query)
    subsets = []
    for r in range(n + 1):
        subsets.extend(itertools.combinations(query, r))
    return subsets[1:]


def compute_shapley_values(query:list,original_urls:str,retriever)->list:
    subsets = compute_all_subsets_of_query(query)
    computable_subset = subsets
    if len(subsets)>30: #limit to a sensible size
        computable_subset = random.sample(subsets,30)
    new_queries = [" ".join(query_tuple) for query_tuple in computable_subset]
    shap_docs = start_search(new_queries,retriever)
    
    fraction_same_docs = []
    for i in range(len(computable_subset)):
        new_urls = shap_docs[shap_docs["Query"]==i]["URL"]
        cut = set(original_urls) & set(new_urls)
        fraction_same_docs.append(len(cut)/100)
    
    in_subset = np.zeros((len(query),len(fraction_same_docs)),dtype=bool)
    fraction = np.zeros((len(query),len(fraction_same_docs)),dtype=bool)
    for i,element_query in enumerate(query):
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
        
        st.markdown("##### What can i help you with?")
        mode = st.pills("Mode",["Query refinement","Website recommendation"],default=st.session_state.chatbot_mode,label_visibility="collapsed")
        if mode != st.session_state.chatbot_mode:
            st.session_state.messages=[]
        st.session_state.chatbot_mode = mode

        st.markdown("""
            <div style='max-height: 150px; overflow-y: auto; padding-right: 10px;'>
        """, unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if len(st.session_state.messages) % 2 ==0:
            with st.chat_message("assistant"):
                response = st.write_stream(begin_conversation(st.session_state.chatbot_mode))
            st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown("</div>", unsafe_allow_html=True)

        left_side,right_side = st.columns([5,1])
        if right_side.button("",key="delete_chat",icon=":material/delete:",type="tertiary"):
            st.session_state.messages=[]
            st.rerun()
        prompt = left_side.chat_input("Get assistance")
        if prompt:
            st.session_state.messages.append({"role":"user","content": prompt})
            st.rerun()


def colored_pill(text, color):
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 0.25em 0.75em;
        margin: 0.25em;
        border-radius: 999px;
        font-size: 0.9em;
        white-space: nowrap;
        display: inline-block;
    ">{text}</span>
    """


##Initialization
st.set_page_config(page_title="Search",page_icon="🔎",layout="wide")
if "ndocs" not in st.session_state:
    st.session_state.ndocs=9042 #Look for documents
st.session_state.n_results = 100
if "i_session" not in st.session_state:
    st.session_state.i_session = None
if "i_query" not in st.session_state:
    st.session_state.i_query = None
if "history" not in st.session_state:
    st.session_state.history = []
if "use_meta_data" not in st.session_state:
    st.session_state.use_meta_data = False
if "unique_keywords" not in st.session_state:
    st.session_state.unique_keywords = False
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
    
    st.link_button("","https://github.com/TimGerne/MSE_Project",icon=":material/school:",type="tertiary",use_container_width=True)
        


#st.title("Kopernikus 🔭")
st.markdown("<h1 style='text-align: center;'>Kopernikus 🔭</h1>", unsafe_allow_html=True)

st.markdown(f':orange-badge[{st.session_state.ndocs} documents to search]')

retriever = instantiate_retrieval_model()

with st.form("search_form"):
    queries_file = st.file_uploader("Upload Queries via file 📁",help="Upload tab separated format",accept_multiple_files=False) 
    query = st.text_input("or put your Query here 👇",placeholder="What are you looking for?",help="Accept only one lined query")
    
    #Search code
    submitted = st.form_submit_button('Start Search 🔎')#,on_click=start_searching)
    if submitted:
        queries = get_queries(query,queries_file)
        #st.session_state.queries=queries
        docs = start_searching(query,queries_file,retriever)
        #docs = example_retrieved_docs #TODO: remove when search implemented
        #st.session_state.pd_docs = docs
        st.session_state.i_session = len(st.session_state.history)
        st.session_state.i_query = 0
        session_name = f"{st.session_state.i_session +1}. Session"
        st.session_state.history.append([queries,docs,session_name])
        st.rerun()


st.html("<hr>")

st.markdown(f"<h2 style='text-align: center;'>Search results</h2>", unsafe_allow_html=True)



if len(st.session_state.history)>0: #check if already searched

    i_session = st.session_state.i_session
    queries,docs,session_name = st.session_state.history[i_session]
    
    left_side,right_side = st.columns([9,1])
    left_side.markdown(f"## {session_name}")

    with right_side:
        popover = st.popover("",icon=":material/settings:")
        st.session_state.n_results = popover.number_input("number of results",min_value=1,value=20,label_visibility="visible",help="Number of results shown")
        left_side,right_side = popover.columns([2,1])
        st.session_state.use_meta_data = left_side.checkbox("Use metadata",help="Makes the search engine slightly slower",value=st.session_state.use_meta_data)
        if right_side.button(label="",icon=":material/refresh:",key="Rerun_button",type="tertiary"):
            st.rerun()
        st.session_state.unique_keywords = left_side.checkbox("Generate unique keywords",help="Only if metadata is enabled",value=st.session_state.unique_keywords)
        
    
    query_tabs_name = [f"Results {i}" for i in range(len(queries))]

    col1, col2, col3,col4 = st.columns([1, 4, 1, 2],gap="small")

    with col1:
        if st.button("", key="query_back", icon=":material/arrow_back:", use_container_width=True):
            if st.session_state.i_query>0:
                st.session_state.i_query -= 1

    with col2:
        if st.button(f"{st.session_state.i_query + 1}. Query", type="tertiary", use_container_width=True):
            st.toast("You discovered an Easter 🥚")

    with col3:
        if st.button("", key="query_next", icon=":material/arrow_forward:",use_container_width=True):
            if st.session_state.i_query<len(queries)-1:
                st.session_state.i_query += 1

    with col4:
        st.session_state.i_query = st.selectbox("",label_visibility="collapsed",options=range(len(queries)),format_func=lambda i: queries[i],index=st.session_state.i_query)

    st.html("<hr>")
    with st.container():
        ls,rs = st.columns([7,1])
        i_query = st.session_state.i_query
        query = queries[i_query]
        ls.markdown(f"##### Query: {query}")
        rs.download_button("📥",docs[docs["Query"]==i_query].to_csv(index=False,header=False,sep='\t'),type="tertiary",file_name=f"results_query_{query}.tsc")
        retrieved_docs = docs.loc[docs["Query"]==i_query].iloc[:,1:]
        shown_docs = retrieved_docs.iloc[:st.session_state.n_results,:]
        
        #will slow the browser down
        if st.session_state.use_meta_data:
            metas =extract_metas_description(shown_docs["URL"])
            keywords_list = get_keywords(metas,st.session_state.unique_keywords)
            colors_keyword_dict = get_color_keyword_dict([keyword for keywords in keywords_list for keyword in keywords])
        
        for j,element in shown_docs.iterrows():
            col1,col2,col3,col4 = st.columns([3,3,1,1])
            rank = element["Rank"]
            url = element["URL"]
            rel = element["Relevance"]
            col1.markdown(f"#### {rank}. [{url}]({url})")
            
            #keywords 
            with col2:
                if st.session_state.use_meta_data:
                    meta = metas[j]
                    keywords = keywords_list[j]

                    pills_html = "".join([colored_pill(keyword, colors_keyword_dict[keyword]) for keyword in keywords])
                    full_html = f"""
                    <div style="display: flex; flex-wrap: wrap; align-items: center;">
                        {pills_html}
                    """
                    st.markdown(full_html, unsafe_allow_html=True)


            if col3.button("Summary",key=f"summary_button{j}",icon=":material/summarize:"):
                get_summary(url)
            col4.markdown(round(float(rel),ndigits=4))

            if st.session_state.use_meta_data:
                st.markdown(f"[{meta}]({url})")
            
            

    #shapley value feature
    st.html("<hr>")
    left_side,right_side = st.columns(2)
    with right_side:
        
        if st.button("Compute Query importance",icon=":material/memory:",type="tertiary",use_container_width=True):
            queries,docs,_ = st.session_state.history[st.session_state.i_session]
            query = queries[st.session_state.i_query]
            ori_urls = docs[docs["Query"]==i_query]["URL"]
            shap_val = compute_shapley_values(query.split(),original_urls=ori_urls,retriever=retriever)
            html = format_word_importances(
                words=query.split(),
                importances=shap_val)
            st.write("Importance Indicator:")
            st.write(html, unsafe_allow_html=True)

        

    left_side.download_button('Download all results', docs.to_csv(index=False,header=False,sep='\t'),type="tertiary",icon=":material/download:",file_name="Results.tsv",use_container_width=True)

else:
    st.markdown("## No results")
    

with bottom():
    _,right_side = st.columns([3,1])
    with right_side:
        assistant()
