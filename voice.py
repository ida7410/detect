import platform
import subprocess

_speaker = None

def speak(text):
    global _speaker
    system = platform.system()
    if system == "Darwin":      # Mac
        cmd = ["say", text]
    elif system == "Windows":
        safe = text.replace("'", "''") # escape apostrophes for PowerShell
        cmd = ["powershell", "-Command",
               "Add-Type -AssemblyName System.Speech; "
               f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"]
    else:                       # Linux (sudo apt install espeak)
        cmd = ["espeak", text]
    _speaker = subprocess.Popen(cmd)


def is_speaking():
    return _speaker is not None and _speaker.poll() is None

def stop():
    if is_speaking():
        _speaker.terminate()
