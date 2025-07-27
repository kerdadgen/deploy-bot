__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import json
import os
import networkx as nx
from openai import AzureOpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import datetime
import difflib

# ==== 1. CONFIGURATION ET CHARGEMENT DES RESSOURCES (MIS EN CACHE) ====
@st.cache_resource
def load_resources():
    print("--- CHARGEMENT DES RESSOURCES (une seule fois) ---")
    load_dotenv()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DIR = os.path.join(BASE_DIR, "chroma_RMA")
    GRAPH_FILE_PATH = os.path.join(BASE_DIR, "rma_knowledge_graph.graphml")

    AZURE_API_KEY = st.secrets("AZURE_OPENAI_API_KEY")
    AZURE_ENDPOINT = st.secrets("AZURE_OPENAI_ENDPOINT")
    AZURE_DEPLOYMENT_EMBEDDING = st.secrets("AZURE_DEPLOYMENT_EMBEDDING")
    AZURE_DEPLOYMENT_CHAT = st.secrets("AZURE_DEPLOYMENT_MODEL")
    AZURE_API_VERSION = st.secrets("AZURE_OPENAI_API_VERSION")

    client = AzureOpenAI(api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION, azure_endpoint=AZURE_ENDPOINT)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_function = OpenAIEmbeddingFunction(
        api_key=AZURE_API_KEY,
        api_base=AZURE_ENDPOINT,
        api_type="azure",
        api_version=AZURE_API_VERSION,
        deployment_id=AZURE_DEPLOYMENT_EMBEDDING
    )
    collection = chroma_client.get_collection(name="RMA_vectordb", embedding_function=embedding_function)
    G = nx.read_graphml(GRAPH_FILE_PATH)
    print(f"✅ Ressources chargées : Graphe avec {G.number_of_nodes()} nœuds.")
    return client, collection, G, AZURE_DEPLOYMENT_CHAT

client, collection, G, AZURE_DEPLOYMENT_CHAT = load_resources()

# ==== 2. MOTEUR INTELLIGENT DE QA ====
def decompose_question(question_utilisateur):
    system_prompt = """
    Tu es un expert en analyse de questions. Décompose la question de l'utilisateur en concepts ou entités de base.
    L'objectif est d'identifier les sujets principaux à rechercher dans une base de connaissances.
    Réponds avec un objet JSON contenant une clé "concepts".
    Exemple:
    - Question: "Quelles sont les garanties en cas d'invalidité et comment est calculée la rente ?"
    - Réponse: {"concepts": ["garanties invalidité", "calcul rente invalidité"]}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_utilisateur}
    ]
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_CHAT,
        messages=messages,
        temperature=0.0,
        max_tokens=500
    )
    try:
        content = json.loads(response.choices[0].message.content)
        return content.get("concepts", [])
    except Exception as e:
        print(f"Erreur lors du parsing JSON : {e}")
        return []

def find_context_in_graph(concepts, threshold=0.8):
    context_summary = ""
    related_nodes = set()
    all_nodes = list(G.nodes())

    for concept in concepts:
        for word in concept.split():
            matches = difflib.get_close_matches(word, all_nodes, n=3, cutoff=threshold)
            related_nodes.update(matches)

    if len(related_nodes) > 1:
        nodes_list = list(related_nodes)
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                source_node, target_node = nodes_list[i], nodes_list[j]
                if nx.has_path(G, source_node, target_node):
                    context_summary += f"- Un lien a été trouvé entre '{source_node}' et '{target_node}'.\n"
    for node in related_nodes:
        for neighbor in G.successors(node):
            relation = G.get_edge_data(node, neighbor)['label']
            context_summary += f"- Le concept '{node}' est lié à '{neighbor}' par la relation '{relation}'.\n"
    return context_summary if context_summary else "Aucun lien direct trouvé dans le graphe."

def retrieve_detailed_chunks(concepts):
    all_results = []
    for concept in concepts:
        results = collection.query(query_texts=[concept], n_results=2)
        docs = results.get('documents', [[]])[0]
        all_results.extend(docs)
    return "\n\n---\n\n".join(all_results[:5])



def final_synthesis(question, graph_context, detailed_chunks):
    system_prompt = """
    Tu es un assistant expert de l'assurance RMA. Ton rôle est de synthétiser les informations fournies pour répondre à la question de l'utilisateur.

    FORMAT DE SORTIE OBLIGATOIRE :
    Un objet JSON avec :
    1. "reponse": Une réponse claire, détaillée et informative basée sur les données fournies.
    2. "suggestions": Une liste de 2 objets, chacune étant une question de suivi pertinente que l'utilisateur pourrait poser:

    Exemple :
    {
      "reponse": "Oui, l'assurance DIM couvre l'incapacité permanente totale. Elle prévoit le versement d'une rente mensuelle proportionnelle au salaire brut, jusqu'à l'âge de la retraite. Le calcul prend en compte le taux d'invalidité reconnu et l'ancienneté dans l'entreprise.",
      "suggestions": [
        { "question": "Comment est calculé le montant de la rente d'invalidité ?", "type": "précision" },
        { "question": "Quelles sont les conditions pour déclarer une incapacité ?", "type": "exemple" },
        { "question": "La couverture DIM s'arrête-t-elle si je quitte l'entreprise ?", "type": "question liée" }
      ]
    }
    """

    history_context = ""
    for msg in st.session_state.messages[-5:]:
        history_context += f"{msg['role'].upper()} : {msg['content']}\n"

    user_prompt = f"""
    **CONTEXTE DE CONVERSATION :**
    {history_context}

    **QUESTION UTILISATEUR :**
    "{question}"

    **1. Contexte du Graphe :**
    ```
    {graph_context}
    ```

    **2. Détails des Documents :**
    ```
    {detailed_chunks}
    ```
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_CHAT,
        messages=messages,
        temperature=0.0,
        max_tokens=2000,
        response_format={"type": "json_object"}
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"reponse": response.choices[0].message.content, "suggestions": []}


# ==== 3. INTERFACE STREAMLIT ====
st.set_page_config(page_title="Chatbot RMA Avancé", layout="wide")
st.title("🤖 Chatbot Avancé RMA")
st.caption("Un assistant capable de raisonner sur l'ensemble des documents de connaissance.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "question" not in st.session_state:
    st.session_state.question = ""

def handle_suggestion_click(suggestion):
    st.session_state.question = suggestion

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("suggestions"):
            cols = st.columns(len(message["suggestions"]))
            for i, suggestion in enumerate(message["suggestions"]):
                cols[i].button(
                    suggestion, 
                    key=f"sugg_{message.get('timestamp', i)}_{i}", 
                    on_click=handle_suggestion_click, 
                    args=[suggestion]
                )

prompt = st.chat_input("Posez votre question ici...", key="chat_input")
if prompt:
    st.session_state.question = prompt

if st.session_state.question:
    current_question = st.session_state.question
    st.session_state.question = ""

    st.session_state.messages.append({"role": "user", "content": current_question})
    with st.chat_message("user"):
        st.markdown(current_question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner(""):
            concepts = decompose_question(current_question)
            graph_context = find_context_in_graph(concepts)
            detailed_chunks = retrieve_detailed_chunks(concepts)
            response_data = final_synthesis(current_question, graph_context, detailed_chunks)

            reponse_concise = response_data.get("reponse", "Désolé, une erreur est survenue.")
            suggestions_data = response_data.get("suggestions", [])
            suggestions = [s["question"] if isinstance(s, dict) else s for s in suggestions_data]

            message_placeholder.markdown(reponse_concise)
            st.session_state.messages.append({
                "role": "assistant",
                "content": reponse_concise,
                "suggestions": suggestions,
                "timestamp": datetime.datetime.now()
            })
            st.rerun()
