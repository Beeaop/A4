# G3: says "uncertain" instead of guessing when the model has too little evidence (REQ-5, Trustworthy AI).

def decide_with_abstention(probs: dict, known_words: int, threshold=0.70, min_known_words=2):

    label = max(probs, key=probs.get)
    conf = float(probs[label])

    if known_words < min_known_words:
        return "uncertain", conf

    if conf < threshold:
        return "uncertain", conf
    return label, conf