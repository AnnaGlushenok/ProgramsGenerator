import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    PATTERN_SENTENCE = r'[.!?]\s+'
    TRANSFORMER_MODEL = "intfloat/multilingual-e5-base"
    URL = "https://openrouter.ai/api/v1"
    load_dotenv()
    OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

    LLM_MODELS = {
        "GPT": "openai/gpt-oss-20b:free",  # 5 mini, gpt-oss
        "deepseek": "deepseek/deepseek-chat-v3.1:free",  #
        "gemini": "google/gemini-2.0-flash-exp:free",  # от gemini
        "polaris": "openrouter/polaris-alpha",
        "sherlock": "openrouter/sherlock-think-alpha"
    }
