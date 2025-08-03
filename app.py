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

def retrieve_detailed_chunks_alternative(standalone_question: str):
    """
    Cherche les chunks pertinents en utilisant la question complète et autonome,
    ce qui est souvent plus efficace que de chercher des concepts isolés.
    """
    # On utilise directement la question complète pour la recherche sémantique.
    # On augmente n_results pour s'assurer de récupérer un contexte riche.
    results = collection.query(
        query_texts=[standalone_question], 
        n_results=10  # On récupère directement 10 chunks pertinents pour la question globale
    )
    
    docs = results.get('documents', [[]])[0]
    
    if not docs:
        return "Aucun document pertinent n'a été trouvé dans la base de connaissances."
        
    return "\n\n---\n\n".join(docs)




def final_synthesis(question, standalone_question, graph_context, detailed_chunks):
    # (Le system_prompt et le user_prompt restent les mêmes)
    system_prompt = """
    Tu es un assistant expert pour les courtiers de l'assurance RMA. Ton rôle est de fournir des réponses précises, complètes et basées **exclusivement** sur les documents fournis.

    RÈGLES :
    1.  **Exhaustivité :** Synthétise TOUTES les informations pertinentes des documents pour répondre à la question.
    2.  **Clarté :** Structure ta réponse avec des titres, des listes à puces ou numérotées pour une lisibilité maximale.
    3.  **Ne jamais inventer :** Si l'information n'est pas dans les documents, dis-le clairement.
    4.  **Tu peux répondre aux remerciements.**

    FORMAT DE SORTIE OBLIGATOIRE :
    Un objet JSON avec :
    1. "reponse": Une réponse claire et détaillée à la question originale.
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
    ```
    {graph_context}
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

    # --- Étape 1 : Tentative d'appel avec le mode JSON forcé ---
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_CHAT,
            messages=messages,
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        # Si l'appel réussit, on essaie de parser le JSON
        response_content = response.choices[0].message.content
        return json.loads(response_content)

    except Exception as e:
        print(f"⚠️ AVERTISSEMENT : Le modèle n'a pas retourné un JSON valide. Erreur : {e}")
        print("--- Tentative de réparation : nouvel appel au LLM pour extraire la réponse. ---")
        
        # --- Étape 2 : Plan B - Le modèle n'a pas respecté le format JSON ---
        # On refait un appel en lui demandant d'extraire la réponse de son propre texte confus.
        repair_prompt = f"""
        Le texte suivant devait être un JSON mais a échoué. Extrais-en la réponse principale destinée à l'utilisateur.
        Ignore les clés JSON comme "reponse" ou "suggestions". Donne juste le texte de la réponse.

        Texte à analyser :
        ---
        {response.choices[0].message.content if 'response' in locals() else 'Contenu non disponible'}
        ---
        """
        try:
            repair_response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_CHAT,
                messages=[{"role": "system", "content": "Tu es un expert en nettoyage de texte."},
                          {"role": "user", "content": repair_prompt}],
                temperature=0.0,
                max_tokens=1500
            )
            # On retourne un dictionnaire formaté correctement avec la réponse "réparée".
            repaired_text = repair_response.choices[0].message.content
            return {"reponse": repaired_text, "suggestions": []}
        except Exception as final_e:
            print(f"❌ ERREUR : La tentative de réparation a également échoué. Erreur : {final_e}")
            # En dernier recours, on retourne un message d'erreur clair.
            return {"reponse": "Désolé, je n'ai pas pu formater la réponse correctement. Veuillez réessayer.", "suggestions": []}




def json_to_markdown(data):
    """
    Transforme un dictionnaire ou une liste Python en une chaîne de caractères
    formatée en Markdown pour un affichage clair.
    """
    if not isinstance(data, (dict, list)):
        # Si ce n'est pas un dict ou une liste, c'est probablement déjà du texte.
        return str(data)

    markdown_output = ""
    
    # Gérer le cas où la donnée est un dictionnaire
    if isinstance(data, dict):
        for key, value in data.items():
            # Utiliser les clés comme des titres
            markdown_output += f"### {key.replace('_', ' ').title()}\n"
            if isinstance(value, list):
                # Formater les listes avec des puces
                for item in value:
                    markdown_output += f"- {item}\n"
            else:
                # Afficher les autres valeurs directement
                markdown_output += f"{value}\n"
            markdown_output += "\n"
            
    # Gérer le cas où la donnée est une liste (moins probable mais plus sûr)
    elif isinstance(data, list):
        for item in data:
            markdown_output += f"- {item}\n"

    return markdown_output.strip()


# ==== 3. INTERFACE STREAMLIT ====

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
        with st.spinner("Analyse de la conversation et recherche d'informations..."):
            
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
            detailed_chunks = retrieve_detailed_chunks_alternative(standalone_question)
            
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

            # 1. On utilise notre nouvelle fonction pour transformer la réponse en Markdown propre.
            #    Cette fonction gère tous les cas (texte, liste, ou dictionnaire/JSON).
            reponse_concise = json_to_markdown(raw_response)

            # 2. Le nettoyage avec re.sub n'est plus nécessaire car le formatage est déjà fait.
            #    On peut garder un simple .strip() pour enlever les espaces superflus.
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

