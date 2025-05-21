import os
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript
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
        print(f"[INFO] Generating transcript from YouTube Transcript API...")
        return [
            f"[{format_timestamp(item['start'])}] {item['text'].strip()}"
            for item in transcript
        ]
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript, Exception) as e:
        print(f"[WARN] Could not retrieve transcript: {e}")
        return None


def download_audio(video_url, output_path="data/audio/"):
    """Download audio from a YouTube video using yt-dlp."""
    os.makedirs(output_path, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        # prepare_filename will give you the actual filename
        filename = ydl.prepare_filename(info)
    return filename
