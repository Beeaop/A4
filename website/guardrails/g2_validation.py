# G2: checks the review before the app uses it, and refuses bad input (REQ-4, GDPR Art. 32, OWASP).

import unicodedata

ALLOWED_MODELS = ("naive_bayes", "decision_tree")
MAX_CHARS = 5000  # placeholder, we'll pick the real number from the A2 data later


def clean_text(text):
    # Removing invisible characters and flag any banned control characters
    text = unicodedata.normalize("NFKC", text)  # turn look-alike letters into normal ones
    kept = []
    has_control = False
    for ch in text:
        kind = unicodedata.category(ch)
        if kind == "Cf":  # invisible, e.g. zero-width space: remove it
            continue
        if kind == "Cc" and ch not in "\n\r\t":  # control character, e.g. NUL is not allowed
            has_control = True
            continue
        kept.append(ch)
    return "".join(kept).strip(), has_control


#Checking the form. Returns (review, model, error). error is None if it's fine

def validate(form):
    review = form.get("review")
    model = form.get("model")

    if review is None or model is None:
        return None, None, "Please enter a review and choose a model."
    if model not in ALLOWED_MODELS:
        return None, None, "Please choose one of the listed models."

    review, has_control = clean_text(review)
    if has_control:
        return None, None, "Your review contains characters that aren't allowed."
    if not review:
        return None, None, "Please enter some text."
    if len(review) > MAX_CHARS:
        return None, None, f"Reviews can be at most {MAX_CHARS} characters."

    return review, model, None