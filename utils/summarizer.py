import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text, max_tokens=2000):
    """Split text into chunks based on rough token limits (~4 chars per token)."""
    max_chars = max_tokens * 4
    lines = text.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) < max_chars:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def summarize_text(text, language="en", model="gpt-3.5-turbo"):
    """
    Summarize transcript in chunks, respecting language and token limits.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    if language == "pl":
        prompt = (
            "Stwórz szczegółowe, uporządkowane streszczenie poniższej transkrypcji w języku polskim. "
            "Podziel streszczenie na sekcje z nagłówkami i znacznikami czasowymi w formacie [MM:SS]. "
            "Używaj punktów i podpunktów, jeśli to możliwe.\n\n"
        )
    else:
        prompt = (
            "Create a detailed, structured summary of this transcript section in English. "
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
