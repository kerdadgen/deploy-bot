# -*- coding: utf-8 -*-

# =======================
# FastAPI backend (RMA) – compatible avec l'ancien widget + nouvelles features (réécriture, graphe, suggestions)
# =======================


import os
import json
import uuid
import logging
import datetime
import difflib
import re
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from openai import AzureOpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import networkx as nx

# =======================
# 1) LOGS & ENV
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rma_backend")

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_RMA")
GRAPH_FILE_PATH = os.path.join(BASE_DIR, "rma_knowledge_graph.graphml")

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_EMBEDDING = os.getenv("AZURE_DEPLOYMENT_EMBEDDING")
AZURE_DEPLOYMENT_CHAT = os.getenv("AZURE_DEPLOYMENT_MODEL")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

CHROMA_HOST = os.getenv("CHROMA_HOST", "")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

if not all([AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_EMBEDDING, AZURE_DEPLOYMENT_CHAT, AZURE_API_VERSION]):
    logger.warning("⚠️ Variables d'environnement Azure incomplètes. Vérifiez votre .env.")

# =======================
# 2) FASTAPI APP
# =======================
app = FastAPI(title="RMA Chat Backend (FastAPI)")

# Static & Templates


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =======================
# 3) RESSOURCES PARTAGÉES (chargées 1x)
# =======================
def load_resources():
    """
    Charge les clients (Azure, Chroma) et le graphe (NetworkX) une seule fois.
    """
    logger.info("--- CHARGEMENT DES RESSOURCES ---")

    # Client Azure OpenAI
    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    # Chroma: privilégier le même mode que l'ancien backend (HttpClient),
    # sinon fallback en local PersistentClient.
    logger.info(f"Connexion Chroma via HTTP -> {CHROMA_HOST}:{CHROMA_PORT}")
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
    embedding_function = OpenAIEmbeddingFunction(
        api_key=AZURE_API_KEY,
        api_base=AZURE_ENDPOINT,
        api_type="azure",
        api_version=AZURE_API_VERSION,
        deployment_id=AZURE_DEPLOYMENT_EMBEDDING
    )

    collection = chroma_client.get_collection(
        name="RMA_vectordb",
        embedding_function=embedding_function
    )

    # Graphe de connaissance
    G = None
    if os.path.exists(GRAPH_FILE_PATH):
        try:
            G = nx.read_graphml(GRAPH_FILE_PATH)
            logger.info(f"✅ Graphe chargé: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes.")
        except Exception as e:
            logger.warning(f"⚠️ Échec chargement graphe: {e}")
            G = None
    else:
        logger.warning("⚠️ Fichier graphe introuvable, fonctionnalités du graphe limitées.")

    return client, collection, G

client, collection, G = load_resources()

# =======================
# 4) SESSIONS & HISTORIQUES
# =======================
class ConversationHistory:
    """
    Conserve deux vues:
      - history: liste Q/R (compatibilité ancien widget)
      - messages: format messages role/content (user/assistant) pour LLM
    """
    def __init__(self, max_history_pairs: int = 5):
        self.max_history_pairs = max_history_pairs
        self.history: List[Dict] = []  # [{"question":..., "answer":..., "timestamp":...}, ...]
        self.messages: List[Dict] = [] # [{"role": "user"/"assistant", "content": "..."}]
        self.created_at = datetime.datetime.now()

    def add_user(self, question: str):
        self.messages.append({"role": "user", "content": question})

    def add_assistant(self, answer: str, suggestions: Optional[List[str]] = None):
        self.messages.append({"role": "assistant", "content": answer})
        self.history.append({
            "question": self._last_user_question(),
            "answer": answer,
            "timestamp": datetime.datetime.now().isoformat(),
            "suggestions": suggestions or []
        })
        if len(self.history) > self.max_history_pairs:
            self.history.pop(0)

    def _last_user_question(self) -> str:
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def to_dict(self) -> Dict:
        return {
            "history": self.history,
            "created_at": self.created_at.isoformat()
        }

user_sessions: Dict[str, ConversationHistory] = {}

def get_or_create_session(session_id: Optional[str]) -> (str, ConversationHistory):
    if not session_id or session_id not in user_sessions:
        new_id = str(uuid.uuid4())
        user_sessions[new_id] = ConversationHistory()
        return new_id, user_sessions[new_id]
    return session_id, user_sessions[session_id]

# =======================
# 5) INTELLIGENCE (réécriture, décomposition, graphe, retrieval, synthèse)
# =======================
def rewrite_question_with_history(question: str, history_messages: List[Dict]) -> str:
    """
    Réécrit la question si besoin, en s'appuyant sur les derniers messages pour créer une question autonome.
    """
    try:
        if len(history_messages) < 2 or len(question.split()) > 10:
            return question

        hist_str = ""
        for msg in history_messages[-4:]:  # 4 derniers messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            hist_str += f"{role}: {content}\n"

        system_prompt = """
        Étant donné l'historique de la conversation et la question de suivi, reformule la question de suivi pour qu'elle soit une question autonome et complète.
        Combine le contexte de l'historique avec la nouvelle question.

        Exemple 1:
        Historique:
        user: Parle-moi de l'assurance Multirisque Hôtel DIAFA.
        system: Bien sûr, l'assurance DIAFA couvre les bâtiments, le mobilier...
        Question de suivi: et pour les garanties ?
        Question reformulée: Quelles sont les garanties de l'assurance Multirisque Hôtel DIAFA ?

        Exemple 2:
        Historique:
        user: Quelles sont les conditions pour la perte d'exploitation ?
        system: Il faut une police directe préalable et déclarer une marge brute.
        Question de suivi: en cas de sinistre ?
        Question reformulée: Que faut-il faire en cas de sinistre pour l'assurance Perte d'Exploitation ?
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Historique:\n{hist_str}\nQuestion de suivi: {question}"}
        ]
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_CHAT,
            messages=messages,
            temperature=0.0,
            max_tokens=100
        )
        rewritten = resp.choices[0].message.content or question
        rewritten = rewritten.replace("Question reformulée:", "").strip()
        logger.debug(f"Rewriting: '{question}' -> '{rewritten}'")
        return rewritten or question
    except Exception as e:
        logger.warning(f"Réécriture échouée: {e}")
        return question

def decompose_question(question_utilisateur: str) -> List[str]:
    system_prompt = """
    Tu es un expert en analyse de questions. Décompose la question de l'utilisateur en concepts ou entités de base.
    L'objectif est d'identifier les sujets principaux à rechercher dans une base de connaissances.
    Réponds avec un objet JSON contenant une clé "concepts".
    Exemple:
    - Question: "Quelles sont les garanties en cas d'invalidité et comment est calculée la rente ?"
    - Réponse: {"concepts": ["garanties invalidité", "calcul rente invalidité"]}
    """
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_utilisateur}
        ]
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_CHAT,
            messages=messages,
            temperature=0.0,
            max_tokens=500
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        return data.get("concepts", []) if isinstance(data, dict) else []
    except Exception as e:
        logger.warning(f"Erreur de décomposition: {e}")
        return []


def find_context_in_graph(concepts, threshold=0.8):
    context_summary = ""
    related_nodes = set()
    all_nodes = list(G.nodes())

    for concept in concepts:
        for word in concept.split():
            matches = difflib.get_close_matches(word, all_nodes, n=5, cutoff=threshold)
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

def retrieve_detailed_chunks_alternative(standalone_question: str, product_name=None):
    where_filter = None
    if product_name:
        where_filter = {
            "$and": [
                {"product_name": {"$eq": product_name}}
            ]
        }
        
    results = collection.query(
        query_texts=[standalone_question],
        n_results=10,
        where=where_filter  # Filtre sur les métadonnées
    )
    
    docs = results.get('documents', [[]])[0]
    
    if not docs:
        return "Aucun document pertinent n'a été trouvé dans la base de connaissances."
        
    return "\n\n---\n\n".join(docs)


def final_synthesis(question: str,
                    standalone_question: str,
                    graph_context: str,
                    detailed_chunks: str,
                    recent_messages: List[Dict]) -> Dict:
    system_prompt = """
    Tu es un assistant expert de l'assurance RMA, un outil de support destiné exclusivement aux courtiers et intermédiaires professionnels. Ta mission est de fournir des réponses factuelles, précises et immédiatement exploitables.

    **PRINCIPES DIRECTEURS :**

    1. **ADOPTE LA PERSPECTIVE DU COURTIER :** 
    - C'est la règle la plus importante. Tu parles à un professionnel.
    - **Ne lui explique jamais son propre rôle ou des procédures qu'il exécute lui-même.**
    - Concentre-toi sur les informations que le courtier doit communiquer à son **client final** ou sur les spécificités du produit.
    - Utilise un ton professionnel et formel, adapté à la communication B2B.

    2. **PERTINENCE AVANT TOUT :** 
    - Réponds **précisément et uniquement** à la QUESTION ORIGINALE DE L'UTILISATEUR.
    - Sois concis pour les questions simples (max 3-4 lignes).
    - Pour les questions complexes, structure ta réponse en sections clairement définies.

    3. **EXHAUSTIVITÉ CONTRÔLÉE :** 
    - Questions larges : présente les informations clés de manière structurée.
    - Questions spécifiques : fournis uniquement l'information demandée.
    - Longueur maximale recommandée : 15-20 lignes pour les réponses détaillées.

    4. **PRÉCISION ET CITATIONS :**
    - Base ta réponse **exclusivement** sur les documents fournis.
    - Format des citations : `> citation exacte en italique`
    - Ne jamais inventer ou supposer d'informations.

    5. **CLARTÉ ET STRUCTURE :**
    - Utilise des titres `##` pour les titres
    - Utilise des sections `###` pour les sections
    - Emploie des listes à puces `-` pour les énumérations
    - Mets en gras les **termes clés** et informations critiques
    - Structure : Introduction → Points clés → Détails → Conclusion (si pertinent)

    6. **VOCABULAIRE ET TERMINOLOGIE :**
    - Utilise le vocabulaire technique de l'assurance sans le définir
    - Réserve les explications pour les termes très spécialisés ou nouveaux
    - Maintiens une cohérence terminologique avec les documents sources

    7. **GESTION DES CAS PARTICULIERS :**
    - Question ambiguë : demande une clarification précise
    - Information partielle : indique clairement ce qui est disponible et ce qui manque
    - Réponse négative : "Je n'ai pas trouvé d'information pertinente dans les documents à ma disposition."

    8. **SUGGESTIONS DE SUIVI :**
    - Propose 2-3 questions pertinentes liées au sujet principal
    - Format : question courte et actionnable
    - Privilégie les questions complémentaires utiles pour le courtier
    ================================================================
    **INSTRUCTION DE FORMATAGE CRITIQUE :**
    Ta sortie DOIT être un objet JSON valide. L'intégralité de la réponse textuelle DOIT être contenue dans une SEULE chaîne de caractères sous la clé "reponse".
    ================================================================

    **FORMAT DE SORTIE OBLIGATOIRE :**
    {
      "reponse": "...", // TOUT le contenu pour l'utilisateur va ici en tant que chaîne de caractères formatée en Markdown.
      "suggestions": [
        {"question": "Question de suivi 1"},
        {"question": "Question de suivi 2"}
      ]
    }
    """

    history_context = ""
    for msg in recent_messages[-5:]:
        role = (msg.get("role") or "").upper()
        content = msg.get("content") or ""
        history_context += f"{role} : {content}\n"

    user_prompt = f"""
    **CONTEXTE DE CONVERSATION :**
    {history_context}
    **QUESTION ORIGINALE DE L'UTILISATEUR :**
    "{question}"
    **1. Contexte du Graphe (basé sur la question complète) :**
    ```
    {graph_context}
    ```
    **2. Détails des Documents (basés sur la question complète) :**
    ```
    {detailed_chunks}
    ```
    IMPORTANT : Si les informations dans {detailed_chunks} ne permettent pas de répondre à {question}, appliquer la règle 6.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_CHAT,
            messages=messages,
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.warning(f"⚠️ JSON strict non renvoyé, tentative de réparation: {e}")
        # Réparation simple : extraire du texte
        try:
            repair_prompt = f"""
            Le texte suivant devait être un JSON mais a échoué. Extrais la réponse principale destinée à l'utilisateur.
            Ignore les clés JSON comme "reponse" ou "suggestions". Donne juste le texte de la réponse.

            Texte à analyser :
            ---
            {content if 'content' in locals() else ''}
            ---
            """
            rep = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_CHAT,
                messages=[{"role": "system", "content": "Tu es un expert en nettoyage de texte."},
                          {"role": "user", "content": repair_prompt}],
                temperature=0.0,
                max_tokens=1500
            )
            repaired_text = rep.choices[0].message.content
            return {"reponse": repaired_text, "suggestions": []}
        except Exception as final_e:
            logger.error(f"❌ Échec de réparation: {final_e}")
            return {"reponse": "Désolé, la réponse n'a pas pu être formatée correctement.", "suggestions": []}

def format_response_robustly(response_data):
    """
    Formate une réponse potentiellement hétérogène (dict/list/str) en texte.
    """
    if isinstance(response_data, dict):
        content = response_data.get("reponse", "Aucun contenu de réponse trouvé.")
        return format_response_robustly(content)

    if isinstance(response_data, list):
        return "\n".join([f"- {item}" for item in response_data])

    if isinstance(response_data, str):
        pattern = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ\s]+)\n(\{.*?\})", re.DOTALL)
        matches = pattern.findall(response_data)
        if not matches:
            return response_data

        markdown_output = ""
        for title, dict_str in matches:
            markdown_output += f"### {title.strip()}\n"
            try:
                data_dict = eval(dict_str)  # même logique que votre version Streamlit
                if isinstance(data_dict, dict):
                    for key, value in data_dict.items():
                        markdown_output += f"**{key}**\n"
                        if isinstance(value, list):
                            for item in value:
                                markdown_output += f"- {item}\n"
                        else:
                            markdown_output += f"{value}\n"
                        markdown_output += "\n"
            except Exception:
                markdown_output += f"{dict_str}\n\n"
        return markdown_output.strip()

    return str(response_data)

# =======================
# 6) ENDPOINTS
# =======================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Page d'accueil (si un template 'index.html' existe).
    """
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        return templates.TemplateResponse("index.html", {"request": request})
    # fallback simple
    return HTMLResponse("<h1>RMA Chat Backend (FastAPI)</h1>", status_code=200)

@app.get("/chat", response_class=HTMLResponse)
def chat_interface(request: Request, theme: str = "light"):
    """
    Endpoint pour l'interface de chat (si 'chat.html' existe).
    """
    chat_tpl = os.path.join(TEMPLATES_DIR, "chat.html")
    if os.path.exists(chat_tpl):
        return templates.TemplateResponse("chat.html", {"request": request, "theme": theme})
    return HTMLResponse(f"<h1>Chat (theme={theme})</h1>", status_code=200)

@app.get("/widget.js")
def widget_js():
    path = os.path.join(STATIC_DIR, "js", "chat.js")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="widget js introuvable")
    return FileResponse(path, media_type="application/javascript")

@app.get("/widget.css")
def widget_css():
    path = os.path.join(STATIC_DIR, "css", "chat.css")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="widget css introuvable")
    return FileResponse(path, media_type="text/css")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/api/chat/history")
def get_history(session_id: Optional[str] = None):
    """
    Récupère l'historique d'une session.
    """
    if not session_id or session_id not in user_sessions:
        raise HTTPException(status_code=400, detail="Session invalide")
    return user_sessions[session_id].to_dict()

@app.post("/api/chat/clear")
async def clear_history(payload: Dict):
    """
    Efface l'historique d'une session (réinitialise).
    """
    session_id = payload.get("session_id")
    if not session_id or session_id not in user_sessions:
        raise HTTPException(status_code=400, detail="Session invalide")
    user_sessions[session_id] = ConversationHistory()
    return {"message": "Historique effacé", "session_id": session_id}

@app.post("/api/chat")
async def chat(payload: Dict):
    """
    Pose une question au chatbot.
    Requête JSON: { "question": "...", "session_id": "..."? }
    Réponse JSON: { "session_id": "...", "response": "...", "suggestions": [...], "history": {...} }
    """
    question = (payload or {}).get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question manquante")

    session_id = (payload or {}).get("session_id")
    session_id, conv = get_or_create_session(session_id)

    # Historisation (user)
    conv.add_user(question)

    try:
        # 1) Réécriture autonome
        standalone_question = rewrite_question_with_history(question, conv.messages)

        # 2) Décomposition & graphe (optionnel)
        concepts = decompose_question(standalone_question)
        graph_context = find_context_in_graph(concepts)

        # 3) Retrieval (Chroma) basé sur la question autonome
        detailed_chunks = retrieve_detailed_chunks_alternative(standalone_question)

        # 4) Synthèse finale en JSON strict (avec suggestions)
        response_json = final_synthesis(
            question=question,
            standalone_question=standalone_question,
            graph_context=graph_context,
            detailed_chunks=detailed_chunks,
            recent_messages=conv.messages
        )

        # Format texte robuste pour l'affichage
        response_text = format_response_robustly(response_json).strip()

        # Extraction des suggestions (liste de str)
        suggestions_raw = []
        if isinstance(response_json, dict):
            suggestions_raw = response_json.get("suggestions", [])
        suggestions = [s["question"] if isinstance(s, dict) and "question" in s else str(s) for s in suggestions_raw]

        # Historisation (assistant)
        conv.add_assistant(response_text, suggestions=suggestions)

        return {
            "session_id": session_id,
            "response": response_text,
            "suggestions": suggestions,
            "history": conv.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur /api/chat")
        raise HTTPException(status_code=500, detail=str(e))
