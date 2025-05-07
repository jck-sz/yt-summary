import os
import argparse
from dotenv import load_dotenv

from utils.youtube import extract_video_id, get_transcript, download_audio
from utils.transcriber import transcribe_audio
from utils.summarizer import summarize_text

load_dotenv()

def save_output(video_id, transcript, summary, base_dir="data"):
    """Save transcript and summary to disk with video ID in filename."""
    summaries_dir = os.path.join(base_dir, "summaries")
    transcripts_dir = os.path.join(base_dir, "transcripts")
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)

    # Save full transcript
    transcript_path = os.path.join(transcripts_dir, f"{video_id}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript.strip())
    print(f"\n[INFO] Transcript saved to: {transcript_path}")

    # Save summary
    summary_path = os.path.join(summaries_dir, f"{video_id}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== SUMMARY ===\n\n")
        f.write(summary.strip())
    print(f"[INFO] Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="YouTube video transcription and summarization tool.")
    parser.add_argument("url", help="The full YouTube video URL")
    args = parser.parse_args()

    url = args.url
    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        print(f"[ERROR] Invalid URL: {e}")
        return

    print("\n--- Attempting to retrieve transcript ---")
    transcript_data = get_transcript(video_id)

    if transcript_data:
        print("\n[INFO] Transcript retrieved (preview):")
        print("\n".join(transcript_data[:10]), "...\n")  # preview first 10 lines
        transcript_text = "\n".join(transcript_data)
    else:
        print("\n[WARN] No transcript available. Downloading audio and using Whisper...")
        audio_file = download_audio(url)
        print(f"[INFO] Audio downloaded to: {audio_file}")
        transcript_text = transcribe_audio(audio_file)
        print("\n[INFO] Transcription complete (preview):")
        print(transcript_text[:500], "...\n")

    print("\n--- Generating summary ---")
    summary = summarize_text(transcript_text, style="detailed")
    print("\n[INFO] Summary generated:\n")
    print(summary)

    save_output(video_id, transcript_text, summary)

if __name__ == "__main__":
    main()
