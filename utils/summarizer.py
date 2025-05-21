# utils/summarizer.py
import os
import tiktoken
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model context windows (token limits)
# You can extend this mapping with additional models as needed
CONTEXT_WINDOW = {
    "gpt-4o-mini": 131072,
    "gpt-4o": 131072,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo": 4096,
}
# Reserve tokens for prompt overhead
OVERHEAD = 2000


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text for the specified model using tiktoken."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def chunk_text(text: str, max_tokens: int = 2000) -> list[str]:
    """Split text into smaller chunks (approx. max_tokens each) based on char heuristic."""
    max_chars = max_tokens * 4  # heuristic: 1 token ~ 4 chars
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        wl = len(word) + 1
        if current_length + wl > max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = wl
        else:
            current_chunk.append(word)
            current_length += wl

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def summarize_text(text: str, language: str = "en", model: str = "gpt-3.5-turbo") -> str:
    """
    Summarize transcript in chunks, respecting language and token limits.
    """
    if language == "pl":
        prompt = (
            "Stwórz szczegółowe, uporządkowane streszczenie poniższej transkrypcji w języku polskim. "
            "Podziel streszczenie na sekcje z nagłówkami i znacznikami czasowymi w formacie [MM:SS]. "
            "Przed każdą sekcją oznacz zakres czasu, w formacie [MM:SS]. Używaj punktów i podpunktów, jeśli możliwe.\n\n"
        )
    else:
        prompt = (
            "Create a detailed, structured summary of this transcript section in English. "
            "Split the summary into sections with headers and timestamps in the format [MM:SS]. "
            "Before each section, indicate the time range. Use bullet points if helpful.\n\n"
        )

    chunks = chunk_text(text)
    summaries = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"[INFO] Summarizing chunk {i}/{len(chunks)}...")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes transcripts."},
                {"role": "user",   "content": prompt + chunk}
            ],
            temperature=0.3
        )
        summaries.append(resp.choices[0].message.content)

    return "\n\n".join(summaries).strip()


def _call_article_api(text: str, language: str, model: str) -> str:
    """Internal helper: send article-generation prompt."""
    if language == "pl":
        system_prompt = (
            "Jesteś doświadczonym dziennikarzem piszącym artykuł w stylu gazetowym. "
            "Użyj struktury odwróconej piramidy: zacznij od kluczowych informacji kto/co/gdzie/kiedy/dlaczego, "
            "następnie dodaj szczegóły. Zacznij od nagłówka, potem treść. Nie ograniczaj długości artykułu."
        )
    else:
        system_prompt = (
            "You are an experienced journalist writing a clear, concise newspaper article. "
            "Use an inverted-pyramid structure: start with the key who/what/where/when/why, then provide supporting details. "
            "Begin with a headline, then write the article body. No length limit."
        )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text}
        ],
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()


def generate_article(
    transcript_text: str,
    language: str = "en",
    model: str = "gpt-4o-mini"
) -> str:
    """
    Generate a newspaper-style article using one-shot if within context window,
    otherwise fallback to chunk → meta-summary → article flow.
    """
    total_tokens = count_tokens(transcript_text, model)
    # determine safe limit for this model
    context_limit = CONTEXT_WINDOW.get(model, 0)
    safe_limit = max(context_limit - OVERHEAD, 0)

    if total_tokens <= safe_limit:
        print(f"[INFO] Transcript tokens={total_tokens} <= safe limit={safe_limit}; one-shot with {model}")
        return _call_article_api(transcript_text, language, model)

    # fallback: chunk-and-summarize
    print(f"[WARN] Transcript tokens={total_tokens} exceeds safe limit={safe_limit}; using fallback summarization flow")
    chunks = chunk_text(transcript_text, max_tokens=20000)
    sub_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"[INFO] Fallback: summarizing chunk {i}/{len(chunks)} with gpt-3.5-turbo...")
        sub_summaries.append(
            _call_article_api(chunk, language, "gpt-3.5-turbo")
        )

    meta_summary = "\n\n".join(sub_summaries)
    meta_tokens = count_tokens(meta_summary, model)
    print(f"[INFO] Meta-summary tokens={meta_tokens}; now calling {model} for final article...")
    return _call_article_api(meta_summary, language, model)
