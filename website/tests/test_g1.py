# Tests for G1 (personal-data redaction). All personal data here is fake.
import app as app_module
from guardrails.g1_privacy import redact

# The Normal cases

def test_email_is_removed():
    text, counts = redact("great film, email me at fake@example.com")
    assert "[EMAIL]" in text and counts == {"EMAIL": 1}


def test_irish_phone_is_removed():
    text, counts = redact("ring me on 087 123 4567")
    assert "[PHONE]" in text


def test_ppsn_and_eircode_are_removed():
    text, counts = redact("my ppsn is 1234567TA and I live at D02 X285")
    assert counts == {"PPSN": 1, "EIRCODE": 1}


def test_normal_review_is_left_alone():
    text, counts = redact("Released in 2004, I give it 10/10")
    assert counts == {}


def test_result_page_does_not_show_the_email():
    client = app_module.app.test_client()
    page = client.post("/predict", data={"review": "loved it, fake@example.com",
                                         "model": "naive_bayes"})
    body = page.get_data(as_text=True)
    assert "fake@example.com" not in body and "[EMAIL]" in body


def test_fails_closed_if_redact_crashes(monkeypatch):
    def crash(text):
        raise RuntimeError("test crash")
    monkeypatch.setattr(app_module, "redact", crash)
    client = app_module.app.test_client()
    page = client.post("/predict", data={"review": "call 087 123 4567",
                                         "model": "naive_bayes"})
    assert page.status_code == 500
    assert "087 123 4567" not in page.get_data(as_text=True)


# TRYING TO BREAK G1

def test_break_phone_with_dashes_and_country_code():
    # G1 survives this: different formats are still caught
    assert "[PHONE]" in redact("call 087-123-4567")[0]
    assert "[PHONE]" in redact("call +353 87 123 4567")[0]


def test_break_spelled_out_phone_is_missed():
    # LIMITATION: numbers written as words are not caught
    text, counts = redact("call oh eight seven one two three four five six seven")
    assert counts == {}


def test_break_hidden_character_in_email_is_missed():
    # LIMITATION: an invisible zero-width space stops the email pattern matching.
    # G2 will fix this by stripping invisible characters before G1 runs.
    text, counts = redact("mail fake\u200b@example.com")
    assert "EMAIL" not in counts