"""智能体注册器，用于管理和发现智能体"""
from typing import Dict, Type, List, Optional
from .base_agent import BaseAgent


class AgentRegistry:
    """管理智能体实例和类的注册器"""

    _instance = None
    _agents: Dict[str, BaseAgent] = {}
    _agent_classes: Dict[str, Type[BaseAgent]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}
            cls._instance._agent_classes = {}
        return cls._instance

    def register_class(self, agent_type: str, agent_class: Type[BaseAgent]):
        """注册一个智能体类"""
        self._agent_classes[agent_type] = agent_class

    def register_instance(self, agent_type: str, agent_instance: BaseAgent):
        """注册一个智能体实例"""
        self._agents[agent_type] = agent_instance

    def get_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """根据类型获取智能体实例"""
        return self._agents.get(agent_type)

    def get_agent_class(self, agent_type: str) -> Optional[Type[BaseAgent]]:
        """根据类型获取智能体类"""
        return self._agent_classes.get(agent_type)

    def list_agents(self) -> List[str]:
        """列出所有已注册的智能体类型"""
        return list(self._agents.keys())

    def list_agent_classes(self) -> List[str]:
        """列出所有已注册的智能体类"""
        return list(self._agent_classes.keys())

    def create_agent(self, agent_type: str, config: any) -> Optional[BaseAgent]:
        """从已注册的类创建新的智能体实例"""
        agent_class = self._agent_classes.get(agent_type)
        if agent_class:
            agent_instance = agent_class(config)
            self.register_instance(agent_type, agent_instance)
            return agent_instance
        return None

    def remove_agent(self, agent_type: str):
        """移除一个智能体实例"""
        if agent_type in self._agents:
            agent = self._agents[agent_type]
            agent.close()
            del self._agents[agent_type]

    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """获取所有已注册的智能体实例"""
        return self._agents.copy()

    def clear(self):
        """清除所有已注册的智能体"""
        for agent in self._agents.values():
            agent.close()
        self._agents.clear()
