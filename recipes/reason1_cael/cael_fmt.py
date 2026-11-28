"""Shared formatting for the Seed reasoning-1 run (CAEL Verified Reasoning CC BY 4.0, GSM8K MIT)."""
import re, hashlib
LEADS = ["Quick question: ", "Can you help me with this? ", "I'm trying to work something out. ", "Here is a puzzle. ", "Help me check this: "]
CLOSERS = ["Just give me the answer.", "Explain briefly.", "I want to be sure I have this right.", "Please show your working."]
EOT = "<|endoftext|>"

def user_text(messages):
    return "\n".join(m["content"] for m in messages)

def core_key(messages):
    """Question text with the known template lead-in / closing wrappers stripped (for leakage dedupe)."""
    parts = []
    for m in messages:
        s = m["content"].strip()
        for l in LEADS:
            if s.startswith(l): s = s[len(l):]
        for c in CLOSERS:
            if s.endswith(c): s = s[: -len(c)]
        parts.append(s.strip())
    s = re.sub(r"\s+", " ", " || ".join(parts)).strip().lower()
    return hashlib.blake2b(s.encode("utf8"), digest_size=8).digest()

def cael_doc(r):
    q = "\n".join("User: " + m["content"].strip() for m in r["messages"])
    return f"{q}\nAssistant: {str(r['solution']).strip()}\nAnswer: {str(r['answer']).strip()}"

def prompt_of(messages):
    return "\n".join("User: " + m["content"].strip() for m in messages) + "\nAssistant:"

def gsm_doc(q, a):
    body, final = a.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()      # drop calculator annotations
    return f"User: {q.strip()}\nAssistant: {body}\nAnswer: {final.strip().replace(',', '')}"