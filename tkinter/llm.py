from dataclasses import dataclass

import openai
import sys
import os
import PyPDF2
from gtts import gTTS

import tempfile
import dotenv

dotenv.load_dotenv()

# --- 1. Configuration client Groq/OpenAI ---
api_key_groq = os.getenv('GROQ_API_KEY')
#llm_model = "compound-beta-mini"
#llm_model = "mistral-saba-24b"
#llm_model = "llama-3.3-70b-versatile"
llm_model = "llama-3.1-8b-instant"
tss_model = "playai-tts"
tts_voice = "Mikail-PlayAI"

audio_activated = True

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key_groq
)

# --- 2. Fonction pour streamer la réponse LLM ---
def ask_llm(message_history, max_tokens=150):
    """
    Envoie message_history à l'API et affiche la réponse en streaming.
    Retourne la réponse complète.
    """
    stream = client.chat.completions.create(
        model=llm_model,
        messages=message_history,
        stream=True,
        max_tokens=max_tokens,           # ← limite en tokens de génération
        temperature=0.5,                 # optionnel : ajustez la créativité
        top_p=0.9                        # optionnel : ajustez la diversité
    )
    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            sys.stdout.write(delta.content)
            sys.stdout.flush()
            full_response += delta.content
    print()  # retour à la ligne
    return full_response

@dataclass()
class DTO: # For Data Transfer Object
    resume_filename: str | None = None
    motivation_filename: str | None = None
    job_offer_filename: str | None = None


dto = DTO()

def getCV():
    file_extension = os.path.splitext(dto.resume_filename)[1]
    if file_extension == '.pdf':
        # Ouvrir le fichier PDF
        with open(dto.resume_filename, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Parcourir toutes les pages et extraire le texte
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    if file_extension == '.txt':
        # Ouvrir le fichier TXT
        with open(dto.resume_filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content

def getJobOffer():
    file_extension = os.path.splitext(dto.job_offer_filename)[1]
    if file_extension == '.pdf':
        # Ouvrir le fichier PDF
        with open(dto.job_offer_filename, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Parcourir toutes les pages et extraire le texte
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    if file_extension == '.txt':
        # Ouvrir le fichier TXT
        with open(dto.job_offer_filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content

def getMotivationLetter():
    file_extension = os.path.splitext(dto.motivation_filename)[1]
    if file_extension == '.pdf':
        # Ouvrir le fichier PDF
        with open(dto.motivation_filename, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Parcourir toutes les pages et extraire le texte
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    if file_extension == '.txt':
        # Ouvrir le fichier TXT
        with open(dto.motivation_filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content

def tts(text):
    if audio_activated:
        text = text.replace("CV", "c'est vais").replace("*", "")
        tts = gTTS(text, lang='fr')

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tts.save(temp_file.name)

        # Jouer l'audio avec ffplay (ou vlc)
        os.system(f"ffplay -autoexit -nodisp -loglevel quiet -af \"atempo=1.5\" {temp_file.name}")
        return

# --- 3. Fonction interactive ---
def interactive_interview(fetch_text=lambda: input("(debug) Vous: ")):
    """
    Lance une boucle interactive d'entretien.
    Garde l'historique des messages pour contexte.
    """
    CV = getCV()
    job_offer = getJobOffer()
    motivation_letter = getMotivationLetter()
    print("=== Simulation d'entretien d'embauche ===")
    message_history = [
        {"role": "system", "content": (
            "Vous êtes un recruteur professionnel qui recrute un développeur python spécialisé en machine learning. "
            "Posez des questions d'entretien, une par une,"
            "et simulez un contexte d'entretien d'embauche."
            "Attention à ne pas trop parler, soyez concis et faites en sorte que l'interlocuteur parle ben plus que vous."
            "Voici le CV du candidat :"
            f"{CV}"
            "Voici la lettre de motivation du candidat :"
            f"{motivation_letter}"
            "Voici l'offre d'emploi à laquelle il postule :"
            f"{job_offer}"
            "Quand vous considérez que l'entretien est terminé, envoyez uniquement 'Merci', cela terminera l'entretien."
        )}
    ]


    end = False
    while True:
        try:
            # L'utilisateur saisit sa réponse
            user_input = fetch_text().strip()
            # Si la réponse est 'merci' ou 'exit', on demande un feedback automatiquement
            if user_input.lower() in {"merci", "exit"}:
                message_history.append({"role": "user", "content": "L'entretien est terminé, fais moi un retour sur la manière dont ça s'est passé et donne moi des conseils pour mieux performer la prochaine fois, éventuellement des modifications à apporter à mon CV ou ma lettre de motivation."})

            else:
                message_history.append({"role": "user", "content": user_input})

            print("\nRecruteur:", end=" ")
            # Le recruteur écrit sa réponse selon l'historique de conversation
            recruteur_reply = ask_llm(message_history)
            #print(f"Nombre de messages dans l'historique : {len(message_history)}")
            # Si le recruteur décide que l'interview est terminé, il envoie merci, puis on demande un feedback de la même manière qu'avant en s'arrangeant avec l'historique
            if recruteur_reply.lower() == "merci" or len(message_history) == 10:
                message_history.append({"role": "assistant", "content": "Merci, l'entretien est terminé."})
                message_history.append({"role": "user", "content": "L'entretien est terminé, fais moi un retour sur la manière dont ça s'est passé et donne moi des conseils pour mieux performer la prochaine fois, éventuellement des modifications à apporter à mon CV ou ma lettre de motivation."})
                recruteur_reply = ask_llm(message_history)

                #recruteur_reply

                tts(recruteur_reply)
                message_history.append({"role": "assistant", "content": recruteur_reply})
            else:
                tts(recruteur_reply)
                message_history.append({"role": "assistant", "content": recruteur_reply})

        except KeyboardInterrupt:
            print("\nInterrompu par l'utilisateur. Au revoir!")
            break

if __name__ == "__main__":
    dto.resume_filename = os.path.abspath('../dummies/CV.pdf')
    dto.job_offer_filename = os.path.abspath('../dummies/job_offer.pdf')
    dto.motivation_filename = os.path.abspath('../dummies/motivation_letter.pdf')
    interactive_interview()
