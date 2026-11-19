import os
import secrets

def _env_bool(name, default):
    value = os.environ.get(name) ## Gauardrails can be run when = false if the code isnt written
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class Config:
    # The production baselines
    DEBUG = False
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 64 * 1024))  # rejects hude requires with a 413 error
    LOG_HMAC_KEY = os.environ.get("LOG_HMAC_KEY") or secrets.token_hex(32)     # secret, never in code
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

    # The Guardrail on and off switches
    G1_ENABLED = _env_bool("G1_ENABLED", True)
    G2_ENABLED = _env_bool("G2_ENABLED", True)
    G3_ENABLED = _env_bool("G3_ENABLED", True)


