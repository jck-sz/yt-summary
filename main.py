# utils/main.py
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from utils.youtube import extract_video_id, get_transcript, download_audio
from utils.transcriber import transcribe_audio
from utils.summarizer import summarize_text, generate_article

load_dotenv()

def read_from_file(path: str) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def save_output(video_id: str, transcript: str, summary: str, base_dir: str = "data") -> None:
    (
        Path(base_dir)/"transcripts"
    ).mkdir(parents=True, exist_ok=True)
    (
        Path(base_dir)/"summaries"
    ).mkdir(parents=True, exist_ok=True)

    tpath = Path(base_dir)/"transcripts"/f"{video_id}_transcript.txt"
    wpath = Path(base_dir)/"summaries"/f"{video_id}_summary.txt"
    tpath.write_text(transcript, encoding="utf-8")
    wpath.write_text(summary, encoding="utf-8")
    print(f"[INFO] Transcript saved: {tpath}")
    print(f"[INFO] Summary saved:    {wpath}")


def save_article(video_id: str, article: str, base_dir: str = "data") -> None:
    (
        Path(base_dir)/"articles"
    ).mkdir(parents=True, exist_ok=True)

    apath = Path(base_dir)/"articles"/f"{video_id}_article.txt"
    apath.write_text(article, encoding="utf-8")
    print(f"[INFO] Article saved:    {apath}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube transcription, summarization, and article-generation CLI."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--language",
        choices=["en", "pl"],
        required=True,
        help="Language of the video",
    )
    parser.add_argument(
        "--article",
        action="store_true",
        help="Also generate a newspaper-style article",
    )
    parser.add_argument(
    "--model",
    default="gpt-4o-mini",
    help="Which OpenAI model to use for article generation"
    )
    args = parser.parse_args()

    try:
        vid = extract_video_id(args.url)
    except ValueError as e:
        print(f"[ERROR] Invalid URL: {e}")
        return

    # Prepare cache dirs
    base = Path("data")
    tx_dir = base/"transcripts"
    sm_dir = base/"summaries"
    tx_dir.mkdir(parents=True, exist_ok=True)
    sm_dir.mkdir(parents=True, exist_ok=True)

    tfile = tx_dir/f"{vid}_transcript.txt"
    sfile = sm_dir/f"{vid}_summary.txt"

    transcript = read_from_file(tfile)
    if transcript:
        print(f"[INFO] Loaded transcript from cache: {tfile}")
    else:
        print("[INFO] Fetching transcript...")
        data = get_transcript(vid)
        if data:
            transcript = "\n".join(data)
            print("[INFO] Transcript retrieved from YouTube.")
        else:
            print("[WARN] No transcript; using Whisper fallback.")
            audio = download_audio(args.url)
            print(f"[INFO] Audio saved: {audio}")
            transcript = transcribe_audio(audio)

    if not transcript:
        print("[ERROR] Could not obtain transcript.")
        return

    summary = read_from_file(sfile)
    if summary:
        print(f"[INFO] Loaded summary from cache: {sfile}")
    else:
        print("[INFO] Generating summary...")
        summary = summarize_text(transcript, language=args.language)
        print("[INFO] Summary generated.")
        save_output(vid, transcript, summary)

    if args.article:
        print("[INFO] Generating newspaper-style article...")
        article = generate_article(transcript, language=args.language)
        print("[INFO] Article generated.")
        save_article(vid, article)


if __name__ == "__main__":
    main()
