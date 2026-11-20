# Tests for G2 (input validation). All personal data here is fake.
import app as app_module
from guardrails.g1_privacy import redact
from guardrails.g2_validation import MAX_CHARS, validate

#Normal cases

def test_good_review_is_accepted():
    review, model, error = validate({"review": "  Great film  ", "model": "naive_bayes"})
    assert error is None and review == "Great film"


def test_spaces_only_is_refused():
    review, model, error = validate({"review": "    ", "model": "naive_bayes"})
    assert error == "Please enter some text."


def test_too_long_is_refused():
    ok = validate({"review": "a" * MAX_CHARS, "model": "naive_bayes"})
    too_long = validate({"review": "a" * (MAX_CHARS + 1), "model": "naive_bayes"})
    assert ok[2] is None and too_long[2] is not None


def test_fake_model_name_gets_400():
    client = app_module.app.test_client()
    page = client.post("/predict", data={"review": "Great film", "model": "hacker"})
    assert page.status_code == 400


def test_fails_closed_if_validate_crashes(monkeypatch):
    def crash(form):
        raise RuntimeError("test crash")
    monkeypatch.setattr(app_module, "validate", crash)
    client = app_module.app.test_client()
    page = client.post("/predict", data={"review": "call 087 123 4567", "model": "naive_bayes"})
    assert page.status_code == 400
    assert "087 123 4567" not in page.get_data(as_text=True)


#Trying to break G2

def test_break_control_characters_are_refused():
    for bad in ["good\x00film", "good \x1b[31m film", "good\x7ffilm"]:
        review, model, error = validate({"review": bad, "model": "naive_bayes"})
        assert error is not None


def test_break_invisible_padding_cannot_beat_length_limit():
    # 5000 zero-width spaces are stripped first, so only the real text counts
    padded = "Great film" + "\u200b" * 5000
    review, model, error = validate({"review": padded, "model": "naive_bayes"})
    assert error is None and review == "Great film"


def test_break_hidden_email_is_caught_by_g2_then_g1():
    # G1 alone misses this, but G2 removes the hidden character first
    client = app_module.app.test_client()
    page = client.post("/predict", data={"review": "mail fake\u200b@example.com",
                                         "model": "naive_bayes"})
    assert "fake@example.com" not in page.get_data(as_text=True)


def test_break_lookalike_letters_are_missed():
    # LIMITATION: a Cyrillic "а" looks like a normal "a", but G2 doesn't change it,
    # so the email no longer matches G1's pattern and slips through.
    review, model, error = validate({"review": "mail fake@ex\u0430mple.com", "model": "naive_bayes"})
    text, counts = redact(review)
    assert "EMAIL" not in counts