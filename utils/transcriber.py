import whisper

# Load the model globally so it's reused
model = whisper.load_model("base")  # You can also use: tiny, small, medium, large

def transcribe_audio(audio_file_path):
    """
    Transcribe an audio file using local Whisper.
    
    Parameters:
        audio_file_path (str): Path to the downloaded audio file.
    
    Returns:
        str: Transcribed text.
    """
    print(f"Transcribing {audio_file_path} with local Whisper...")
    result = model.transcribe(audio_file_path)
    return result["text"]
