import json
import os
from pathlib import Path


DEFAULT_LLM_CONFIG = {
    'api_key': os.environ.get('AZURE_OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY', '')),
    'endpoint': os.environ.get('AZURE_OPENAI_ENDPOINT', ''),
    'api_version': os.environ.get('AZURE_OPENAI_API_VERSION', '2024-10-21'),
    'model': os.environ.get('AZURE_OPENAI_MODEL', 'gpt-4o'),
}


class LLMConfigError(RuntimeError):
    pass


def llm_config_path():
    configured_path = os.environ.get('CELLQUANT_LLM_CONFIG')
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / '.cellquant' / 'llm_config.json'


def load_llm_config():
    config = dict(DEFAULT_LLM_CONFIG)
    path = llm_config_path()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
            if isinstance(saved_config, dict):
                config.update({k: v for k, v in saved_config.items() if v is not None})
        except Exception:
            pass
    env_api_key = os.environ.get('AZURE_OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if env_api_key:
        config['api_key'] = env_api_key
    return config


def save_llm_config(config):
    path = llm_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = dict(DEFAULT_LLM_CONFIG)
    cleaned.update({k: (v.strip() if isinstance(v, str) else v) for k, v in config.items() if v is not None})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def validate_llm_config(config):
    missing = [key for key in ('api_key', 'endpoint', 'api_version', 'model') if not config.get(key)]
    if missing:
        raise LLMConfigError(
            f"Missing LLM config field(s): {', '.join(missing)}. "
            "Please fill the LLM settings in the Analysis panel and click Save."
        )
    return config


def require_llm_config():
    return validate_llm_config(load_llm_config())


def format_llm_exception(error, context='LLM request'):
    return (
        f"{context} failed: {error}. "
        "Please check the LLM API key, endpoint, API version, model name, network connection, "
        "and whether the openai Python package is installed in the current environment."
    )
