# Tests for G3 (Abstain when unsure)
from guardrails.g3_abstain import decide_with_abstention

# Normal Cases

def test_high_positive():
    label, conf = decide_with_abstention({"Positive": 0.89, "Negative": 0.11}, known_words=3)
    assert label == "Positive"

def test_high_negative():
    label, conf = decide_with_abstention({"Positive": 0.04, "Negative": 0.96}, known_words=3)
    assert label == "Negative"

def test_unsure():
    label, conf = decide_with_abstention({"Positive": 0.54, "Negative": 0.46}, known_words=3)
    assert label == "uncertain"

# Trying to break G3

def test_adverse_gibberish():
    label, conf = decide_with_abstention({"Positive": 1.0, "Negative": 0.0}, known_words=0)
    assert label == "uncertain"

def test_adverse_spaced_letters():
    label, conf = decide_with_abstention({"Positive": 1.0, "Negative": 0.0}, known_words=1)
    assert label == "uncertain"

def test_adverse_sarcasm():
    label, conf = decide_with_abstention({"Positive": 0.80, "Negative": 0.20}, known_words=8)
    assert label == "Positive"