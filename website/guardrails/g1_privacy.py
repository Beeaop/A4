## G1 

# replaces the email addresses with [EMAIL] 
# so personal data never reaches the model 
# the or logs (REQ-3, GDPR Art. 5(1)(c)).

import re

EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)+")


def redact(text):
    return EMAIL.sub("[EMAIL]", text)