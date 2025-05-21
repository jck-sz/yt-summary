import os
import argparse
from dotenv import load_dotenv

from utils.youtube import extract_video_id, get_transcript, download_audio
from utils.transcriber import transcribe_audio
from utils.summarizer import summarize_text

load_dotenv()

def read_from_file(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None


def save_output(video_id, transcript, summary, base_dir="data"):
    summaries_dir = os.path.join(base_dir, "summaries")
    transcripts_dir = os.path.join(base_dir, "transcripts")
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)

    transcript_path = os.path.join(transcripts_dir, f"{video_id}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript.strip())
    print(f"\n[INFO] Transcript saved to: {transcript_path}")

    summary_path = os.path.join(summaries_dir, f"{video_id}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary.strip())
    print(f"[INFO] Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="YouTube video transcription and summarization tool.")
    parser.add_argument("url", help="The full YouTube video URL")
    parser.add_argument("--language", choices=["en", "pl"], required=True, help="Language of the video (en or pl)")
    args = parser.parse_args()

    url = args.url
    language = args.language

    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        print(f"[ERROR] Invalid URL: {e}")
        return

    transcript_path = os.path.join("data", "transcripts", f"{video_id}_transcript.txt")
    summary_path = os.path.join("data", "summaries", f"{video_id}_summary.txt")

    transcript_text = read_from_file(transcript_path)
    summary = read_from_file(summary_path)

    if transcript_text:
        print(f"[INFO] Loaded cached transcript from: {transcript_path}")
    else:
        print("\n--- Attempting to retrieve transcript ---")
        transcript_data = get_transcript(video_id)
        if transcript_data:
            print("[INFO] Transcript retrieved from YouTube.")
            transcript_text = "\n".join(transcript_data)
        else:
            print("[WARN] No transcript available. Downloading audio and using Whisper...")
            audio_file = download_audio(url)
            print(f"[INFO] Audio downloaded to: {audio_file}")
            transcript_text = transcribe_audio(audio_file)

    if summary:
        print(f"[INFO] Loaded cached summary from: {summary_path}")
    else:
        print("\n--- Generating summary ---")
        summary = summarize_text(transcript_text, language=language)
        print("[INFO] Summary generated.")
        save_output(video_id, transcript_text, summary)


if __name__ == "__main__":
    main()