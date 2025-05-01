import os.path
import tempfile
import threading
import time
import tkinter as tk
import tkinter.filedialog
from tkinter.constants import DISABLED, NORMAL

import numpy as np
import scipy
import sounddevice as sd

from llm import dto, interactive_interview, audio_activated, client

main = tk.Tk()

job_offer_label_variable = tk.StringVar(None)
resume_label_variable = tk.StringVar(None)
motivation_label_variable = tk.StringVar(None)


def upload_job_offer():
    path = tk.filedialog.askopenfilename(filetypes=(("PDF", ".pdf"), ("Texte", ".txt"),), title="Select job offer")
    if path is None:
        return
    path = os.path.abspath(path)
    dto.job_offer_filename = path
    job_offer_label_variable.set(path)


def upload_resume():
    path = tk.filedialog.askopenfilename(filetypes=(("PDF", ".pdf"), ("Texte", ".txt"),), title="Select resume")
    if path is None:
        return
    path = os.path.abspath(path)
    dto.resume_filename = path
    resume_label_variable.set(path)


def upload_motivation():
    path = tk.filedialog.askopenfilename(filetypes=(("PDF", ".pdf"), ("Texte", ".txt"),), title="Select motivation letter")
    if path is None:
        return
    path = os.path.abspath(path)
    dto.motivation_filename = path
    motivation_label_variable.set(path)

def fetch_text():
    global is_recording
    recording_button.configure(state=NORMAL)
    blablating_turn.configure(text="SPEAK !!")
    start_recording()
    while not is_recording:
        time.sleep(1)
    while is_recording:
        time.sleep(1)

    if audio_activated:
        time.sleep(1)
        all_data = np.concat(recording)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        scipy.io.wavfile.write(temp_wav.name, 44100, np.array(list(all_data)))

        # Open the audio file
        with open(temp_wav.name, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3-turbo",
                prompt="",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language="fr",
                temperature=0.0
            )
        result_text = transcription.text
        print("\nVous:", result_text)
    else:
        result_text = input("\nVous: ")
    return result_text

def go_to_interview():
    global recording_button, blablating_turn
    for widget in main.winfo_children():
        widget.destroy()
    recording_button = tk.Button(main, text="Stop recording", command=stop_recording, state=DISABLED)
    recording_button.pack()
    blablating_turn = tk.Label(main, text="SPEAK !!")
    blablating_turn.pack()
    def func():
        interactive_interview(fetch_text)
    threading.Thread(target=func).start()


is_recording = False
recording_button: tk.Button | None = None
blablating_turn: tk.Label | None = None
recording = []

def start_recording():
    global is_recording
    is_recording = True

    if audio_activated:
        # Start recording in a separate thread
        def record_thread():
            global is_recording
            recording.clear()
            muted_for = 0
            may_have_started_for = 0
            started_recording = False
            while is_recording:
                data = sd.rec(10000, 44100, channels=1, blocking=True)
                recording.append(data)
                if data.max() < .28:
                    if started_recording:
                        muted_for += 1
                        if muted_for == 5:
                            stop_recording()
                    else:
                        started_recording = False
                else:
                    if not started_recording:
                        may_have_started_for += 1
                        if may_have_started_for == 5:
                            started_recording = True
                    else:
                        muted_for = 0


        threading.Thread(target=record_thread).start()


def stop_recording():
    global is_recording
    is_recording = False
    recording_button.configure(state=DISABLED)
    recording_button.configure(text="Start recording", command=start_recording)
    blablating_turn.configure(text="Listen to michel or michelle or mychel or mychelle or michL (you got it)")


tk.Button(main, text="Upload the job offer", command=upload_job_offer, width=40, height=5).pack()
tk.Label(main, textvariable=job_offer_label_variable, background="orange").pack()
tk.Button(main, text="Upload your resume", command=upload_resume, width=40, height=5).pack()
tk.Label(main, textvariable=resume_label_variable, background="orange").pack()
tk.Button(main, text="Upload the motivation letter", command=upload_motivation, width=40, height=5).pack()
tk.Label(main, textvariable=motivation_label_variable, background="orange").pack()
tk.Button(main, text="Go to interview", command=go_to_interview, background="pink", width=40, height=10).pack()

main.attributes("-fullscreen", True)
main.configure(background="orange")

dto.resume_filename = os.path.abspath('../dummies/CV.pdf')
dto.job_offer_filename = os.path.abspath('../dummies/job_offer.pdf')
dto.motivation_filename = os.path.abspath('../dummies/motivation_letter.pdf')

tk.mainloop()
