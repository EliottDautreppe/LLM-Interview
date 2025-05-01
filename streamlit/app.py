import io
import time
import streamlit as st
import numpy as np
import openai
import sys
import threading
import os
import PyPDF2
import tempfile
from gtts import gTTS
from dataclasses import dataclass

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Simulateur d'Entretien d'Embauche IA",
    page_icon="🤖",
    layout="wide"
)

# --- Dataclass pour les fichiers uploadés ---
@dataclass()
class DTO:  # For Data Transfer Object
    resume_filename: str | None = None
    motivation_filename: str | None = None
    job_offer_filename: str | None = None

# Initialisation des variables de session
if "dto" not in st.session_state:
    st.session_state.dto = DTO()

if "api_key_groq" not in st.session_state:
    st.session_state.api_key_groq = "VOTRE API GROK ICI"

if "llm_model" not in st.session_state:
    st.session_state.llm_model = "compound-beta-mini"

if "tss_model" not in st.session_state:
    st.session_state.tss_model = "playai-tts"

if "tts_voice" not in st.session_state:
    st.session_state.tts_voice = "Mikail-PlayAI"

if "audio_activated" not in st.session_state:
    st.session_state.audio_activated = True

if "client" not in st.session_state:
    st.session_state.client = None

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

if "files_ready" not in st.session_state:
    st.session_state.files_ready = False

if "interview_ended" not in st.session_state:
    st.session_state.interview_ended = False

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

# --- Fonctions pour le traitement des fichiers ---
def save_uploaded_file(uploaded_file, directory="temp_files"):
    """Sauvegarde un fichier uploadé dans un répertoire temporaire et retourne le chemin."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def getCV():
    """Extrait le texte du CV."""
    if st.session_state.dto.resume_filename is None:
        return ""
    
    file_extension = os.path.splitext(st.session_state.dto.resume_filename)[1]
    if file_extension == '.pdf':
        # Ouvrir le fichier PDF
        with open(st.session_state.dto.resume_filename, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Parcourir toutes les pages et extraire le texte
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    if file_extension == '.txt':
        # Ouvrir le fichier TXT
        with open(st.session_state.dto.resume_filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    
    return ""

def getJobOffer():
    """Extrait le texte de l'offre d'emploi."""
    if st.session_state.dto.job_offer_filename is None:
        return ""
    
    file_extension = os.path.splitext(st.session_state.dto.job_offer_filename)[1]
    if file_extension == '.pdf':
        # Ouvrir le fichier PDF
        with open(st.session_state.dto.job_offer_filename, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Parcourir toutes les pages et extraire le texte
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    if file_extension == '.txt':
        # Ouvrir le fichier TXT
        with open(st.session_state.dto.job_offer_filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    
    return ""

def getMotivationLetter():
    """Extrait le texte de la lettre de motivation."""
    if st.session_state.dto.motivation_filename is None:
        return ""
    
    file_extension = os.path.splitext(st.session_state.dto.motivation_filename)[1]
    if file_extension == '.pdf':
        # Ouvrir le fichier PDF
        with open(st.session_state.dto.motivation_filename, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Parcourir toutes les pages et extraire le texte
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    if file_extension == '.txt':
        # Ouvrir le fichier TXT
        with open(st.session_state.dto.motivation_filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    
    return ""

# --- Fonction pour la synthèse vocale ---
def tts(text):
    """Convertit le texte en audio et le joue."""
    if not st.session_state.audio_activated:
        return
    
    try:
        tts = gTTS(text, lang='fr')
        
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            tts.save(temp_file.name)
            
            # Jouer l'audio avec st.audio
            audio_file = open(temp_file.name, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            audio_file.close()
            
        # Nettoyer le fichier temporaire
        os.unlink(temp_file.name)
    except Exception as e:
        st.error(f"Erreur lors de la synthèse vocale: {e}")

# --- Configuration du client OpenAI ---
def init_client():
    """Initialise le client API avec les paramètres actuels."""
    try:
        st.session_state.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=st.session_state.api_key_groq
        )
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du client API: {e}")
        return False

# --- Fonction pour streamer la réponse LLM ---
def ask_llm(message_history, max_tokens=150, message_placeholder=None):
    """
    Envoie message_history à l'API et affiche la réponse en streaming.
    Retourne la réponse complète.
    """
    if st.session_state.client is None:
        if not init_client():
            return "Erreur de connexion à l'API. Veuillez vérifier votre clé API."
    
    try:
        stream = st.session_state.client.chat.completions.create(
            model=st.session_state.llm_model,
            messages=message_history,
            stream=True,
            max_tokens=max_tokens,
            temperature=0.5,
            top_p=0.9
        )
        
        # Si aucun placeholder n'est fourni, on en crée un
        if message_placeholder is None:
            message_placeholder = st.empty()
        
        full_response = ""
        
        # Afficher le stream progressivement
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_response += delta.content
                message_placeholder.markdown(f"**Recruteur:** {full_response}")
        
        return full_response
    
    except Exception as e:
        st.error(f"Erreur lors de la communication avec l'API: {e}")
        return f"Erreur: {str(e)}"

# --- Fonction pour gérer la soumission du message utilisateur ---
def handle_user_input():
    user_input = st.session_state.user_input
    
    if user_input.strip():
        # Ajouter la réponse de l'utilisateur à l'historique
        st.session_state.message_history.append({"role": "user", "content": user_input})
        
        # Réinitialiser l'input
        st.session_state.user_input = ""
        
        # Créer un placeholder pour la réponse du recruteur
        with st.chat_message("assistant", avatar="👔"):
            message_placeholder = st.empty()
            
            # Obtenir la réponse du recruteur
            recruteur_reply = ask_llm(st.session_state.message_history, message_placeholder=message_placeholder)
            
            # Vérifier si l'entretien se termine
            if recruteur_reply.lower() == "merci" or len(st.session_state.message_history) >= 20:
                st.session_state.interview_ended = True
                st.session_state.message_history.append({"role": "assistant", "content": "Merci"})
            else:
                st.session_state.message_history.append({"role": "assistant", "content": recruteur_reply})
                
                if st.session_state.audio_activated:
                    tts(recruteur_reply)

# --- Fonction pour générer un feedback après l'entretien ---
def generate_feedback():
    if not st.session_state.feedback_given:
        st.session_state.message_history.append({"role": "user", "content": 
            "L'entretien est terminé, fais moi un retour sur la manière dont ça s'est passé et donne moi des conseils pour mieux performer la prochaine fois, éventuellement des modifications à apporter à mon CV ou ma lettre de motivation."
        })
        
        with st.chat_message("assistant", avatar="👔"):
            message_placeholder = st.empty()
            feedback = ask_llm(st.session_state.message_history, max_tokens=500, message_placeholder=message_placeholder)
            
        st.session_state.message_history.append({"role": "assistant", "content": feedback})
        
        if st.session_state.audio_activated:
            tts(feedback)
            
        st.session_state.feedback_given = True

# --- Fonction pour l'entretien interactif ---
def interactive_interview():
    """
    Lance une interface interactive d'entretien.
    Garde l'historique des messages pour contexte.
    """
    CV = getCV()
    job_offer = getJobOffer()
    motivation_letter = getMotivationLetter()
    
    if not CV or not job_offer:
        st.error("Impossible de récupérer le CV ou l'offre d'emploi. Vérifiez les fichiers.")
        return
    
    st.markdown("### Simulation d'entretien d'embauche")
    
    # Initialiser l'historique des messages si ce n'est pas déjà fait
    if len(st.session_state.message_history) == 0:
        system_message = {
            "role": "user", 
            "content": (
                "Vous êtes un recruteur professionnel qui recrute un développeur python spécialisé en machine learning. "
                "Posez des questions d'entretien, une par une, "
                "et simulez un contexte d'entretien d'embauche. "
                "Attention à ne pas trop parler, soyez concis et faites en sorte que l'interlocuteur parle ben plus que vous. "
                "Voici le CV du candidat : "
                f"{CV} "
                "Voici la lettre de motivation du candidat : "
                f"{motivation_letter} "
                "Voici l'offre d'emploi à laquelle il postule : "
                f"{job_offer} "
                "Quand vous considérez que l'entretien est terminé, envoyez uniquement 'Merci', cela terminera l'entretien."
            )
        }
        st.session_state.message_history.append(system_message)
        
        with st.chat_message("assistant", avatar="👔"):
            message_placeholder = st.empty()
            recruteur_reply = ask_llm(st.session_state.message_history, message_placeholder=message_placeholder)
            
        st.session_state.message_history.append({"role": "assistant", "content": recruteur_reply})
        
        if st.session_state.audio_activated:
            tts(recruteur_reply)
    
    # Conteneur pour afficher l'historique des messages
    chat_container = st.container()
    
    with chat_container:
        # Afficher l'historique des messages (en ignorant le message système)
        for i, message in enumerate(st.session_state.message_history[1:], 1):
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant", avatar="👔"):
                    st.markdown(message["content"])
    
    # Vérifier si l'entretien est terminé
    if st.session_state.interview_ended and not st.session_state.feedback_given:
        st.info("L'entretien est terminé. Le recruteur va vous donner un feedback.")
        generate_feedback()
        
        if st.button("Terminer l'entretien"):
            st.session_state.interview_started = False
            st.session_state.interview_ended = False
            st.session_state.feedback_given = False
            st.session_state.message_history = []
            st.rerun()
    
    # Si l'entretien n'est pas terminé, afficher la zone de saisie
    if not st.session_state.interview_ended:
        st.text_input(
            "Votre réponse:",
            key="user_input",
            on_change=handle_user_input
        )

# --- Interface principale ---
st.title("🤖 Simulateur d'Entretien d'Embauche IA")

# Barre latérale pour les configurations
with st.sidebar:
    st.header("Configuration")
    
    st.session_state.api_key_groq = st.text_input("Clé API Groq", value=st.session_state.api_key_groq, type="password")
    
    st.session_state.llm_model = st.selectbox(
        "Modèle LLM",
        ["compound-beta-mini", "mistral-saba-24b", "llama-3.3-70b-versatile"],
        index=0
    )
    
    st.session_state.audio_activated = st.checkbox("Activer l'audio", value=st.session_state.audio_activated)
    
    # Upload des fichiers
    st.header("Documents")
    cv_file = st.file_uploader("Chargez votre CV (PDF ou TXT)", type=['pdf', 'txt'])
    job_offer_file = st.file_uploader("Chargez l'offre d'emploi (PDF ou TXT)", type=['pdf', 'txt'])
    motivation_file = st.file_uploader("Chargez votre lettre de motivation (facultatif) (PDF ou TXT)", type=['pdf', 'txt'])
    
    if st.button("Préparer l'entretien"):
        if cv_file and job_offer_file:
            # Créer le répertoire temp_files s'il n'existe pas
            if not os.path.exists("temp_files"):
                os.makedirs("temp_files")
            
            # Sauvegarder les fichiers uploadés
            st.session_state.dto.resume_filename = save_uploaded_file(cv_file)
            st.session_state.dto.job_offer_filename = save_uploaded_file(job_offer_file)
            
            if motivation_file:
                st.session_state.dto.motivation_filename = save_uploaded_file(motivation_file)
            
            # Initialiser le client API
            if init_client():
                st.session_state.files_ready = True
                st.success("Documents préparés avec succès!")
            else:
                st.error("Erreur lors de l'initialisation du client API.")
        else:
            st.error("Veuillez uploader au moins votre CV et l'offre d'emploi.")

# Section principale
if not st.session_state.interview_started:
    st.info("👈 Commencez par configurer et préparer votre entretien dans le panneau de gauche.")
    
    if st.session_state.files_ready:
        if st.button("🚀 Démarrer l'entretien"):
            st.session_state.interview_started = True
            st.session_state.message_history = []  # Réinitialiser l'historique
            st.session_state.interview_ended = False
            st.session_state.feedback_given = False
            st.rerun()  # Nécessaire uniquement pour initialiser l'entretien

else:
    # Si l'entretien est démarré, lancer la fonction interactive
    interactive_interview()