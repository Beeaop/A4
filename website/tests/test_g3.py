# Tests for G3 (Abstain when unsure)
import app as app_module
from guardrails.g3_abstain import decide_with_abstention

# Normal Cases

def test_high_positive():
    label, conf = decide_with_abstention("Amazing amazing great great film.")
    assert label != "uncertain"

def test_high_negative():
    label, conf = decide_with_abstention("Awful awful bad bad film.")
    assert label != "uncertain"

def test_unsure():
    label, conf = decide_with_abstention("Amazing awful great bad film.")
    assert label == "uncertain"

# Trying to break G3

def test_adverse_positive():
    label, conf = decide_with_abstention("g o o d g r e a t film.")
    assert label != "uncertain"

def test_adverse_negative():
    label, conf = decide_with_abstention("a w f u l b a d film.")
    assert label != "uncertain"

def test_adverse_unsure():
    label, conf = decide_with_abstention("g o o d b a d film.")
    assert label == "uncertain"