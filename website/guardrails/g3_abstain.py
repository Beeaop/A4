# G3: Abstains from deciding whether a review is positive or negative if the confidence does not pass a given threshold

def decide_with_abstention(probs: dict, threshold=0.70):
    label = max(probs, key=probs.get)
    conf = float(probs[label])
    if conf < threshold:
        return "uncertain", conf
    return label, conf