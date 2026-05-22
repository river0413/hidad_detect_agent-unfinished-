"""Multi-Agent System using LangGraph architecture"""
from .orchestrator import LangGraphOrchestrator
from .agents import (
    BaseAgent,
    AgentResponse,
    AgentRegistry,
    TextProcessingAgent,
    KnowledgeAgent,
    DecisionAgent,
    ExplanationAgent
)
from .config import (
    ConfigManager,
    AgentConfig,
    TextProcessingConfig,
    KnowledgeAgentConfig,
    DecisionAgentConfig,
    ExplanationAgentConfig
)

__version__ = "1.0.0"

__all__ = [
    'LangGraphOrchestrator',
    'BaseAgent',
    'AgentResponse',
    'AgentRegistry',
    'TextProcessingAgent',
    'KnowledgeAgent',
    'DecisionAgent',
    'ExplanationAgent',
    'ConfigManager',
    'AgentConfig',
    'TextProcessingConfig',
    'KnowledgeAgentConfig',
    'DecisionAgentConfig',
    'ExplanationAgentConfig'
]
