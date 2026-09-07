
"""
agent/config.py

Configuration settings for the Self-Learning Coding Agent.
Loads settings from environment variables or .env file.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class AgentConfig:
    """Agent runtime configuration."""
    
    # Kimi LLM Provider settings
    api_key: str = os.getenv("KIMI_API_KEY", "")
    base_url: str = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    model: str = os.getenv("KIMI_MODEL", "kimi-k2.7-code")
    
    # Generation parameters (kimi-k2.7-code requires temperature=1.0)
    temperature: float = float(os.getenv("TEMPERATURE", "1.0"))
    max_tokens: Optional[int] = int(os.getenv("MAX_TOKENS", "16384"))
    
    # Execution & Self-Correction
    max_retries: int = int(os.getenv("MAX_RETRIES", "4"))
    sandbox_timeout: float = float(os.getenv("SANDBOX_TIMEOUT", "15.0"))
    
    # Memory storage path
    memory_file: str = os.getenv("MEMORY_FILE", "agent_memory.json")
    
    def validate(self) -> bool:
        """Checks if minimum required credentials are present."""
        if not self.api_key or self.api_key == "your_kimi_api_key_here":
            return False
        return True


def get_config() -> AgentConfig:
    """Returns a new AgentConfig instance populated from environment."""
    return AgentConfig()
