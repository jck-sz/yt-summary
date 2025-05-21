import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript
from pytube import YouTube
from datetime import timedelta

def extract_video_id(url):
    """Extract video ID from a YouTube URL."""
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL format")

from datetime import timedelta

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

def get_transcript(video_id, languages=["en", "pl"]):
    """Try to get the transcript in preferred languages."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return [
            f"[{format_timestamp(item['start'])}] {item['text'].strip()}"
            for item in transcript
        ]
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript, Exception) as e:
        print(f"[WARN] Could not retrieve transcript: {e}")
        return None


def download_audio(video_url, output_path="data/audio/"):
    """Download audio from a YouTube video using pytube."""
    os.makedirs(output_path, exist_ok=True)
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = audio_stream.download(output_path=output_path)
    return audio_file
