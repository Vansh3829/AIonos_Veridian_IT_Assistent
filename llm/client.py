"""
Thin wrapper around the Groq-hosted Llama 3.1 8B Instant model via langchain-groq.

Kept deliberately simple: one function to get the LLM client, one helper to
call it and parse a JSON object out of the response (with a safe fallback).
"""

import os
import json
import re
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

_llm = None


class LLMConfigError(Exception):
    """Raised when the Groq API key is missing or invalid."""
    pass


def get_llm(temperature: float = None):
    """Return a cached ChatGroq client. Raises LLMConfigError if no API key is set.

    Model name and default temperature come from .env (GROQ_MODEL, LLM_TEMPERATURE)
    so they can be changed without touching code.
    """
    global _llm
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    if _llm is None:
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        temp = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.0"))
        _llm = ChatGroq(
            model=model_name,
            temperature=temp,
            api_key=api_key,
        )
    return _llm


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send a system + user prompt to the LLM and return the plain text response."""
    llm = get_llm()
    messages = [
        ("system", system_prompt),
        ("human", user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def call_llm_json(system_prompt: str, user_prompt: str, fallback: dict) -> dict:
    """
    Call the LLM expecting a JSON object back, and parse it.
    If parsing fails for any reason, return `fallback` so the app never crashes
    on a malformed model response.
    """
    raw = call_llm(system_prompt, user_prompt)
    text = raw.strip()

    # Strip markdown code fences if the model wrapped the JSON in them
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    # Extract the first {...} block in case there's extra chatter around it
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return fallback
