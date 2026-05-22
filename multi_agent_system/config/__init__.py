"""Configuration package for multi-agent system"""
from .agent_config import (
    AgentConfig,
    TextProcessingConfig,
    KnowledgeAgentConfig,
    DecisionAgentConfig,
    ExplanationAgentConfig,
    ConfigManager
)

__all__ = [
    'AgentConfig',
    'TextProcessingConfig',
    'KnowledgeAgentConfig',
    'DecisionAgentConfig',
    'ExplanationAgentConfig',
    'ConfigManager'
]
