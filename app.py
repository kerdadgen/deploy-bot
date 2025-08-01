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
import re
# ==== 1. CONFIGURATION ET CHARGEMENT DES RESSOURCES (MIS EN CACHE) ====
@st.cache_resource
def load_resources():
    print("--- CHARGEMENT DES RESSOURCES (une seule fois) ---")
    load_dotenv()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DIR = os.path.join(BASE_DIR, "chroma_RMA")
    GRAPH_FILE_PATH = os.path.join(BASE_DIR, "rma_knowledge_graph.graphml")

    AZURE_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
    AZURE_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
    AZURE_DEPLOYMENT_EMBEDDING = st.secrets["AZURE_DEPLOYMENT_EMBEDDING"]
    AZURE_DEPLOYMENT_CHAT = st.secrets["AZURE_DEPLOYMENT_MODEL"]
    AZURE_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

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

# ==== NOUVELLE FONCTION : RÉÉCRITURE DE LA QUESTION ====
def rewrite_question_with_history(question: str, history: list):
    # Si l'historique est court ou si la question semble déjà complète, on ne réécrit pas.
    if len(history) < 2 or len(question.split()) > 10:
        return question

    # Concaténer l'historique pour le contexte
    history_str = ""
    for msg in history[-4:]: # On prend les 4 derniers messages pour le contexte
        history_str += f"{msg['role']}: {msg['content']}\n"

    system_prompt = f"""
    Étant donné l'historique de la conversation et la question de suivi, reformule la question de suivi pour qu'elle soit une question autonome et complète.
    Combine le contexte de l'historique avec la nouvelle question.

    Exemple 1:
    Historique:
    user: Parle-moi de l'assurance Multirisque Hôtel DIAFA.
    assistant: Bien sûr, l'assurance DIAFA couvre les bâtiments, le mobilier...
    Question de suivi: et pour les garanties ?
    Question reformulée: Quelles sont les garanties de l'assurance Multirisque Hôtel DIAFA ?

    Exemple 2:
    Historique:
    user: Quelles sont les conditions pour la perte d'exploitation ?
    assistant: Il faut une police directe préalable et déclarer une marge brute.
    Question de suivi: en cas de sinistre ?
    Question reformulée: Que faut-il faire en cas de sinistre pour l'assurance Perte d'Exploitation ?
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Historique:\n{history_str}\nQuestion de suivi: {question}"}
    ]

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_CHAT,
            messages=messages,
            temperature=0.0,
            max_tokens=100
        )
        rewritten_question = response.choices[0].message.content.replace("Question reformulée:", "").strip()
        print(f"--- Question originale: '{question}'")
        print(f"--- Question réécrite: '{rewritten_question}'")
        return rewritten_question
    except Exception as e:
        print(f"Erreur lors de la réécriture de la question : {e}")
        return question # En cas d'erreur, on retourne la question originale

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



def final_synthesis(question, standalone_question, graph_context, detailed_chunks):
    system_prompt = """
    Tu es un assistant expert de l'assurance RMA. Ton rôle est de synthétiser les informations fournies pour répondre à la question ORIGINALE de l'utilisateur.
    Utilise le contexte fourni, qui a été récupéré sur la base de la "Question Complète pour Recherche", merci de ne pas dire et indiquer que ces informations viennent des documents réponds juste à la question.

    FORMAT DE SORTIE OBLIGATOIRE :
    Un objet JSON avec :
    1. "reponse": Une réponse claire et détaillée à la question originale. Si la réponse est détaillée, utilise une liste numérotée (1., 2., ...).
    2. "suggestions": Une liste de 2 questions de suivi pertinentes.
    """

    history_context = ""
    for msg in st.session_state.messages[-5:]:
        history_context += f"{msg['role'].upper()} : {msg['content']}\n"

    user_prompt = f"""
    **CONTEXTE DE CONVERSATION :**
    {history_context}

    **QUESTION ORIGINALE DE L'UTILISATEUR :**
    "{question}"

    **QUESTION COMPLÈTE POUR RECHERCHE (générée à partir de l'historique) :**
    "{standalone_question}"

    **1. Contexte du Graphe (basé sur la question complète) :**
    ```    {graph_context}
    ```

    **2. Détails des Documents (basés sur la question complète) :**
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
# ==== 3. INTERFACE STREAMLIT (Version avec Mémoire de Conversation) ====

st.set_page_config(page_title="Chatbot RMA Avancé", layout="wide")
st.title("🤖 Chatbot Avancé RMA")
st.caption("Un assistant capable de raisonner sur l'ensemble des documents de connaissance.")

# Initialisation de l'état de la session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question" not in st.session_state:
    st.session_state.question = ""

# Fonction pour gérer le clic sur une suggestion
def handle_suggestion_click(suggestion):
    st.session_state.question = suggestion

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Afficher les suggestions sous la réponse de l'assistant
        if message.get("suggestions"):
            cols = st.columns(len(message["suggestions"]))
            for i, suggestion in enumerate(message["suggestions"]):
                # Utiliser un timestamp ou un ID unique pour les clés des boutons
                unique_key = f"sugg_{message.get('timestamp', i)}_{i}"
                cols[i].button(
                    suggestion, 
                    key=unique_key, 
                    on_click=handle_suggestion_click, 
                    args=[suggestion]
                )

# Champ de saisie du chat
prompt = st.chat_input("Posez votre question ici...", key="chat_input")
if prompt:
    st.session_state.question = prompt

# Logique de traitement si une nouvelle question est posée
if st.session_state.question:
    current_question = st.session_state.question
    st.session_state.question = "" # Réinitialiser pour éviter une ré-exécution en boucle

    # Ajouter la question de l'utilisateur à l'historique et l'afficher
    st.session_state.messages.append({"role": "user", "content": current_question})
    with st.chat_message("user"):
        st.markdown(current_question)

    # Afficher la réponse de l'assistant
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner(""):
            
            # ==================================================================
            # ==== ÉTAPE 1 : RÉÉCRIRE LA QUESTION AVEC LE CONTEXTE (LA CLÉ) ====
            # ==================================================================
            # On passe l'historique complet pour que la fonction ait tout le contexte.
            standalone_question = rewrite_question_with_history(current_question, st.session_state.messages)

            # ==================================================================
            # ==== ÉTAPE 2 : RECHERCHE BASÉE SUR LA QUESTION AUTONOME ========
            # ==================================================================
            concepts = decompose_question(standalone_question)
            graph_context = find_context_in_graph(concepts)
            detailed_chunks = retrieve_detailed_chunks(concepts)
            
            # ==================================================================
            # ==== ÉTAPE 3 : SYNTHÈSE FINALE ===================================
            # ==================================================================
            # On passe la question originale (pour la réponse) et la question réécrite (pour le contexte)
            response_data = final_synthesis(
                question=current_question,
                standalone_question=standalone_question,
                graph_context=graph_context, 
                detailed_chunks=detailed_chunks
            )

            raw_response = response_data.get("reponse", "Désolé, une erreur est survenue.")

            # 1. On s'assure que la réponse est bien une chaîne de caractères
            if isinstance(raw_response, list):
                # Si c'est une liste, on joint tous les éléments avec un saut de ligne
                reponse_concise = "\n".join(map(str, raw_response))
            else:
                # Sinon, on s'assure que c'est bien une chaîne (au cas où ce serait autre chose)
                reponse_concise = str(raw_response)

            # 🔧 Nettoyage du Markdown pour un affichage propre
            reponse_concise = re.sub(r"(#+)([^\s#])", r"\1 \2", reponse_concise)
            reponse_concise = re.sub(r"(##[^\n]*)", r"\n\n\1\n\n", reponse_concise)
            reponse_concise = re.sub(r"([^\n])(\n- )", r"\1\n\n\2", reponse_concise)
            reponse_concise = re.sub(r"\n{3,}", r"\n\n", reponse_concise)
            reponse_concise = re.sub(r"(?<!\n)(\d+\. )", r"\n\n\1", reponse_concise)
            reponse_concise = reponse_concise.strip()

            # Extraire les suggestions de la réponse
            suggestions_data = response_data.get("suggestions", [])
            suggestions = [s["question"] if isinstance(s, dict) else s for s in suggestions_data]

            # Afficher la réponse finale
            message_placeholder.markdown(reponse_concise, unsafe_allow_html=False)

            # Ajouter la réponse complète de l'assistant à l'historique
            st.session_state.messages.append({
                "role": "assistant",
                "content": reponse_concise,
                "suggestions": suggestions,
                "timestamp": datetime.datetime.now().isoformat() # Utiliser un format standard pour la clé
            })
            
            # Forcer la ré-exécution du script pour afficher les boutons de suggestion
            st.rerun()

