"""
agents/base.py
Base agent class with common functionality.
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class BaseAgent(ABC):
    """
    Base class for all agents in the Agentic RAG system.

    Why this is needed:
    - Provides unified LLM access
    - Standardizes agent interface
    - Handles API key management
    """

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.1):
        self.model_name = model_name or Config.LLM_MODEL
        self.temperature = temperature
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initialize LLM based on available API keys."""
        if Config.GROQ_API_KEY:
            return ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=Config.MAX_TOKENS,
                api_key=Config.GROQ_API_KEY
            )
        elif Config.OPENROUTER_API_KEY:
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=Config.MAX_TOKENS,
                base_url="https://openrouter.ai/api/v1",
                api_key=Config.OPENROUTER_API_KEY
            )
        else:
            raise ValueError("No LLM API key configured")

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's task. Must be implemented by subclasses."""
        pass

    def _invoke_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Helper to invoke LLM with optional system prompt."""
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = self.llm.invoke(messages)
        return response.content
