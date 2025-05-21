import os
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text, max_tokens=2000):
    """Split text into smaller chunks without breaking on newline, based on character length."""
    max_chars = max_tokens * 4
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for the space
        if current_length + word_length > max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_text(text, language="en", model="gpt-3.5-turbo"):
    """
    Summarize transcript in chunks, respecting language and token limits.
    """
    if language == "pl":
        prompt = (
            "Stwórz szczegółowe, uporządkowane streszczenie poniższej transkrypcji w języku polskim. "
            "Podziel streszczenie na sekcje z nagłówkami i znacznikami czasowymi w formacie [MM:SS]. "
            "Przed każdą sekcją oznacz odnośnie jakiego zakresu czasu odnosi się dane streszczenie, w formacie [MM:SS]."
            "Używaj punktów i podpunktów, jeśli to możliwe.\n\n"
        )
    else:
        prompt = (
            "Create a detailed, structured summary of this transcript section in English. "
            "Split the summary into sections with headers and timestamps in the format [MM:SS]. "
            "Before each section, indicate the time range it covers, in the format [MM:SS]. "
            "Divide it into logical sections with headers. Include [MM:SS] timestamps and use bullet points if helpful.\n\n"
        )

    chunks = chunk_text(text)
    all_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"[INFO] Summarizing chunk {i+1} of {len(chunks)}...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes transcripts."},
                {"role": "user", "content": prompt + chunk}
            ],
            temperature=0.3
        )
        all_summaries.append(response.choices[0].message.content)

    return "\n\n".join(all_summaries)