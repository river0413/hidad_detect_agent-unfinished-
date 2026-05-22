"""智能体配置管理模块"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os
import json
from pathlib import Path


@dataclass
class AgentConfig:
    """智能体的基础配置"""
    name: str
    model: str = "deepseek-reasoner"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None
    api_base: str = "https://api.deepseek.com"
    system_prompt: str = ""
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'api_key': self.api_key,
            'api_base': self.api_base,
            'system_prompt': self.system_prompt,
            'enabled': self.enabled
        }


@dataclass
class TextProcessingConfig(AgentConfig):
    """文本处理智能体配置"""
    classifier_path: str = "text/inference_service.py"
    use_ai_fallback: bool = True

    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = """你是一个专业的文本分析专家，专注于判断文本是否为正常文本、广告或隐性广告。

**分类定义：**
1. **正常文本**：纯粹的表达个人情感、分享经历、交流观点的文字
2. **显性广告**：明确、直接的推广内容，包含明显的购买引导、价格信息、促销信息等
3. **隐性广告**：以看似自然的方式包装的推广内容，通过分享、推荐、测评等形式的广告

请对给定的文本进行分析并返回JSON格式的判断结果。"""


@dataclass
class KnowledgeAgentConfig(AgentConfig):
    """知识管理智能体配置"""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    sqlite_path: str = "data/knowledge.db"
    graph_name: str = "text_analysis_graph"

    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = """你是一个知识图谱专家，负责管理和查询知识图谱数据。

你的职责包括：
1. 将文本分析结果存储到知识图谱中
2. 查询历史分析记录和统计信息
3. 识别文本之间的关联关系
4. 提供知识推理和查询服务"""


@dataclass
class DecisionAgentConfig(AgentConfig):
    """决策智能体配置"""
    updateable_agents: list = field(default_factory=lambda: ["text_processing", "knowledge", "explanation"])

    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = """你是一个决策智能体，负责协调其他智能体的工作并更新它们的提示词。

你的职责包括：
1. 分析任务需求，决定调用哪些智能体
2. 根据上下文更新其他智能体的提示词
3. 优化工作流程，提高效率
4. 处理冲突和异常情况

你可以更新以下智能体的提示词：
- text_processing: 文本处理智能体
- knowledge: 知识智能体
- explanation: 解释智能体"""


@dataclass
class ExplanationAgentConfig(AgentConfig):
    """解释智能体配置"""
    output_format: str = "human_readable"

    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = """你是一个解释专家，负责将分析结果转换为人类容易理解的内容。

你的职责包括：
1. 将JSON格式的结构化数据转换为自然语言描述
2. 提供详细的分析和解释
3. 用通俗易懂的方式解释复杂的技术结果
4. 生成可视化的建议和结论"""


class ConfigManager:
    """所有智能体的配置管理器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "agent_configs.json"
        self.configs: Dict[str, AgentConfig] = {}

    def load_configs(self) -> Dict[str, AgentConfig]:
        """从文件加载配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for agent_type, config_data in data.items():
                    self.configs[agent_type] = self._create_config(agent_type, config_data)
        else:
            self.configs = self._get_default_configs()
            self.save_configs()
        return self.configs

    def _create_config(self, agent_type: str, data: Dict[str, Any]) -> AgentConfig:
        """根据类型创建配置对象"""
        config_classes = {
            'text_processing': TextProcessingConfig,
            'knowledge': KnowledgeAgentConfig,
            'decision': DecisionAgentConfig,
            'explanation': ExplanationAgentConfig
        }
        config_class = config_classes.get(agent_type, AgentConfig)
        return config_class(**data)

    def _get_default_configs(self) -> Dict[str, AgentConfig]:
        """获取默认配置"""
        return {
            'text_processing': TextProcessingConfig(
                name="text_processing",
                model="deepseek-reasoner"
            ),
            'knowledge': KnowledgeAgentConfig(
                name="knowledge",
                model="deepseek-reasoner"
            ),
            'decision': DecisionAgentConfig(
                name="decision",
                model="deepseek-reasoner"
            ),
            'explanation': ExplanationAgentConfig(
                name="explanation",
                model="deepseek-reasoner"
            )
        }

    def save_configs(self):
        """保存配置到文件"""
        data = {k: v.to_dict() for k, v in self.configs.items()}
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_config(self, agent_type: str) -> Optional[AgentConfig]:
        """获取指定智能体类型的配置"""
        return self.configs.get(agent_type)

    def update_config(self, agent_type: str, updates: Dict[str, Any]):
        """更新指定智能体类型的配置"""
        if agent_type in self.configs:
            config = self.configs[agent_type]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            self.save_configs()

    def get_prompt(self, agent_type: str) -> str:
        """获取智能体的当前提示词"""
        config = self.get_config(agent_type)
        return config.system_prompt if config else ""

    def update_prompt(self, agent_type: str, new_prompt: str):
        """更新智能体的提示词"""
        self.update_config(agent_type, {'system_prompt': new_prompt})
