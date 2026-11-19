# G1: removes personal data from reviews so it never reaches the model or the logs (REQ-3, GDPR Art. 5(1)(c)).

import re

# Kinds of personal data to look for: (label, pattern). Specific ones go first.
PATTERNS = [
    ("EMAIL",   re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)+")),  # eg name@example.com
    ("URL",     re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)),             # eg www.example.com
    ("IP",      re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),                         # eg 192.0.2.44
    ("PPSN",    re.compile(r"\b\d{7}[A-W][A-IW]?\b", re.IGNORECASE)),                # eg 1234567TA
    ("EIRCODE", re.compile(r"\b[AC-FHKNPRTV-Y]\d{2}\s?[0-9AC-FHKNPRTV-Y]{4}\b", re.IGNORECASE)),  # eg D02 X285
    ("PHONE",   re.compile(r"(?:\+353|\b0)[\s\-]?\d{1,2}[\s\-]?\d{3}[\s\-]?\d{3,4}\b")),  # eg 087 123 4567
]


def redact(text):
    """Swap personal data for [LABEL] and count how many of each were found."""
    counts = {}
    for label, pattern in PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)  # replace and count the matches
        if n:
            counts[label] = n  # only keep the labels that were found
    return text, counts