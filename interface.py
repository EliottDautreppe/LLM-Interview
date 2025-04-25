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

from python import dto, interactive_interview, audio_activated, client

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
    recording_button.configure(state=NORMAL)
    while not is_recording:
        time.sleep(1)
    while is_recording:
        time.sleep(1)
    recording_button.configure(state=DISABLED)

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
    global main, recording_button
    for widget in main.winfo_children():
        widget.destroy()
    recording_button = tk.Button(main, text="Start recording", command=start_recording, state=DISABLED)
    recording_button.pack()
    def func():
        interactive_interview(fetch_text)
    threading.Thread(target=func).start()


is_recording = False
recording_button = None
recording = []

def start_recording():
    global is_recording
    is_recording = True
    recording_button.configure(text="Stop recording", command=stop_recording)

    if audio_activated:
        # Start recording in a separate thread
        def record_thread():
            while is_recording:
                data = sd.rec(10000, 44100, channels=1, blocking=True)
                recording.append(data)

        threading.Thread(target=record_thread).start()

def stop_recording():
    global is_recording
    is_recording = False
    recording_button.configure(text="Start recording", command=start_recording)



tk.Button(main, text="Upload the job offer", command=upload_job_offer, width=40, height=5).pack()
tk.Label(main, textvariable=job_offer_label_variable, background="orange").pack()
tk.Button(main, text="Upload your resume", command=upload_resume, width=40, height=5).pack()
tk.Label(main, textvariable=resume_label_variable, background="orange").pack()
tk.Button(main, text="Upload the motivation letter", command=upload_motivation, width=40, height=5).pack()
tk.Label(main, textvariable=motivation_label_variable, background="orange").pack()
tk.Button(main, text="Go to interview", command=go_to_interview, background="pink", width=40, height=10).pack()

main.attributes("-fullscreen", True)
main.configure(background="orange")

tk.mainloop()
