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
    resume_content: str = None
    motivation_content: str = None
    job_offer_content: str = None


# --- Initialisation des variables de session ---
if "dto" not in st.session_state:
    st.session_state.dto = DTO()

if "api_key" not in st.session_state:
    st.session_state.api_key = "..."

if "llm_model" not in st.session_state:
    st.session_state.llm_model = "compound-beta-mini"

if "audio_activated" not in st.session_state:
    st.session_state.audio_activated = True

if "client" not in st.session_state:
    st.session_state.client = None

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

if "interview_ended" not in st.session_state:
    st.session_state.interview_ended = False

if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = False


# --- Fonctions pour l'API et les traitements ---
def init_client():
    """Initialise le client API avec les paramètres actuels."""
    try:
        st.session_state.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=st.session_state.api_key
        )
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du client API: {e}")
        return False


def extract_text_from_pdf(pdf_file):
    """Extrait le texte d'un fichier PDF."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text


def extract_text_from_file(file):
    """Extrait le texte d'un fichier (PDF ou TXT)."""
    if file is None:
        return ""

    file_extension = os.path.splitext(file.name)[1].lower()

    if file_extension == '.pdf':
        return extract_text_from_pdf(file)
    elif file_extension == '.txt':
        return file.getvalue().decode('utf-8')
    else:
        st.error(f"Format de fichier non pris en charge: {file_extension}")
        return ""


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


def ask_llm(message_history, max_tokens=150):
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

        # Créer un placeholder pour le streaming
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


# --- Fonctions pour l'entretien ---
def prepare_documents(cv_file, job_offer_file, motivation_file):
    """Prépare les documents pour l'entretien."""
    with st.spinner("Traitement des documents..."):
        if cv_file and job_offer_file:
            st.session_state.dto.resume_content = extract_text_from_file(cv_file)
            st.session_state.dto.job_offer_content = extract_text_from_file(job_offer_file)

            if motivation_file:
                st.session_state.dto.motivation_content = extract_text_from_file(motivation_file)
            else:
                st.session_state.dto.motivation_content = "Non fournie"

            if init_client():
                st.session_state.documents_ready = True
                st.success("Documents chargés avec succès! Vous pouvez maintenant démarrer l'entretien.")
                return True
            else:
                st.error("Erreur lors de l'initialisation du client API.")
                return False
        else:
            st.error("Veuillez charger au moins votre CV et l'offre d'emploi.")
            return False


def start_interview():
    """Démarre l'entretien avec le recruteur IA."""
    if not st.session_state.documents_ready:
        st.error("Veuillez d'abord préparer les documents.")
        return

    st.session_state.message_history = [
        {"role": "system", "content": (
            "Vous êtes un recruteur professionnel qui recrute un développeur python spécialisé en machine learning. "
            "Posez des questions d'entretien, une par une, et simulez un contexte d'entretien d'embauche. "
            "Attention à ne pas trop parler, soyez concis et faites en sorte que l'interlocuteur parle ben plus que vous. "
            "Voici le CV du candidat :\n"
            f"{st.session_state.dto.resume_content}\n"
            "Voici la lettre de motivation du candidat :\n"
            f"{st.session_state.dto.motivation_content}\n"
            "Voici l'offre d'emploi à laquelle il postule :\n"
            f"{st.session_state.dto.job_offer_content}\n"
            "Quand vous considérez que l'entretien est terminé, envoyez uniquement 'Merci', cela terminera l'entretien."
        )}
    ]

    st.session_state.interview_started = True
    st.session_state.interview_ended = False

    # Première question du recruteur
    st.markdown("### L'entretien commence")
    st.session_state.message_history.append({"role": "user", "content": "Bonjour"})
    st.markdown(st.session_state.message_history)
    recruteur_reply = ask_llm(st.session_state.message_history)
    st.markdown(recruteur_reply)
    st.session_state.message_history.append({"role": "assistant", "content": recruteur_reply})

    if st.session_state.audio_activated:
        tts(recruteur_reply)


def submit_response():
    """Traite la réponse du candidat et génère la réponse du recruteur."""
    user_input = st.session_state.user_input
    if not user_input.strip():
        return

    # Afficher la réponse de l'utilisateur
    st.markdown(f"**Vous:** {user_input}")

    # Si la réponse est 'merci' ou 'exit', on demande un feedback automatiquement
    if user_input.lower() in {"merci", "exit"}:
        st.session_state.message_history.append({"role": "user",
                                                 "content": "L'entretien est terminé, fais moi un retour sur la manière dont ça s'est passé et donne moi des conseils pour mieux performer la prochaine fois, éventuellement des modifications à apporter à mon CV ou ma lettre de motivation."})
    else:
        st.session_state.message_history.append({"role": "user", "content": user_input})

    # Le recruteur écrit sa réponse selon l'historique de conversation
    recruteur_reply = ask_llm(st.session_state.message_history)

    # Si le recruteur décide que l'interview est terminé, il envoie merci
    if recruteur_reply.lower() == "merci" or len(st.session_state.message_history) >= 10:
        st.session_state.message_history.append({"role": "assistant", "content": "Merci, l'entretien est terminé."})
        st.session_state.message_history.append({"role": "user",
                                                 "content": "L'entretien est terminé, fais moi un retour sur la manière dont ça s'est passé et donne moi des conseils pour mieux performer la prochaine fois, éventuellement des modifications à apporter à mon CV ou ma lettre de motivation."})

        feedback = ask_llm(st.session_state.message_history, max_tokens=500)
        st.session_state.message_history.append({"role": "assistant", "content": feedback})

        if st.session_state.audio_activated:
            tts(feedback)

        st.session_state.interview_ended = True
    else:
        st.session_state.message_history.append({"role": "assistant", "content": recruteur_reply})

        if st.session_state.audio_activated:
            tts(recruteur_reply)

    # Vider le champ de saisie
    st.session_state.user_input = ""


# --- Interface principale ---
st.title("🤖 Simulateur d'Entretien d'Embauche IA")

# Barre latérale pour les configurations
with st.sidebar:
    st.header("Configuration")

    st.session_state.api_key = st.text_input("Clé API Groq", value=st.session_state.api_key, type="password")
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
    motivation_file = st.file_uploader("Chargez votre lettre de motivation (facultatif) (PDF ou TXT)",
                                       type=['pdf', 'txt'])

    if st.button("Préparer l'entretien"):
        prepare_documents(cv_file, job_offer_file, motivation_file)

# Section principale - Affichage conditionnel
if not st.session_state.interview_started:
    st.info("👈 Configurez votre entretien dans le panneau de gauche.")

    # Bouton pour démarrer l'entretien uniquement si les documents sont prêts
    if st.session_state.documents_ready:
        if st.button("🚀 Démarrer l'entretien"):
            start_interview()

elif st.session_state.interview_ended:
    st.success("✅ L'entretien est terminé!")

    # Afficher le feedback final
    for message in st.session_state.message_history:
        if message["role"] == "assistant" and st.session_state.message_history[-1]["content"] == message["content"]:
            st.markdown("### Feedback du recruteur")
            st.markdown(message["content"])

    if st.button("🔄 Recommencer un nouvel entretien"):
        st.session_state.interview_started = False
        st.session_state.interview_ended = False
        st.session_state.documents_ready = False
        st.experimental_rerun()

else:
    # Afficher l'historique des messages pendant l'entretien
    for message in st.session_state.message_history:
        if message["role"] == "user" and message[
            "content"] != "L'entretien est terminé, fais moi un retour sur la manière dont ça s'est passé et donne moi des conseils pour mieux performer la prochaine fois, éventuellement des modifications à apporter à mon CV ou ma lettre de motivation.":
            st.markdown(f"**Vous:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**Recruteur:** {message['content']}")

    # Zone de saisie pour la réponse du candidat
    st.text_input("Votre réponse:", key="user_input", on_change=submit_response)