from openai import OpenAI
import os

def summarize_text(text, style="detailed", model="gpt-4"):
    """
    Summarize a transcript using OpenAI's GPT API.

    Parameters:
        text (str): The transcript to summarize.
        style (str): Summary style — 'detailed', 'bullet', or 'short'.
        model (str): OpenAI model to use.

    Returns:
        str: The summary.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    client = OpenAI(api_key=api_key)

    if style == "bullet":
        prompt = (
            "Summarize the following video transcript into clear bullet points "
            "that capture the main ideas and topics discussed:\n\n"
        )
    elif style == "short":
        prompt = "Summarize this video transcript in 3-4 concise sentences:\n\n"
    else:  # detailed
        prompt = (
            "Create a detailed, structured summary of this video transcript, divided into logical sections "
            "with clear headers. For each section, include an estimated timestamp based on where the topic appears. "
            "Use format like [00:45] at the beginning of each section title.\n\n"
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes YouTube transcripts."},
            {"role": "user", "content": prompt + text}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
