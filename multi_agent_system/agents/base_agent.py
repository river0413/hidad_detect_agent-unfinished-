"""多智能体系统的基础智能体类"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import requests
import json
import time
from datetime import datetime


@dataclass
class AgentResponse:
    """智能体的标准响应格式"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    agent_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'agent_name': self.agent_name,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class BaseAgent(ABC):
    """多智能体系统中所有智能体的基类"""

    def __init__(self, config: Any):
        import os
        self.config = config
        self.name = config.name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.api_key = config.api_key or self._load_api_key()
        self.model = os.getenv('DEEPSEEK_MODEL') or config.model
        self.api_base = os.getenv('DEEPSEEK_API_BASE') or config.api_base
        self.system_prompt = config.system_prompt
        self.enabled = config.enabled
        self.session = None
        self._init_session()
        self._prompt_history: List[Dict[str, str]] = []

    def _init_session(self):
        """初始化HTTP会话，仅在存在有效API密钥时设置认证头"""
        if self.api_key:
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })

    def _load_env(self):
        """从multi_agent_system/.env文件加载所有环境变量到os.environ"""
        import os
        try:
            env_file = Path(__file__).resolve().parent.parent / ".env"
            if env_file.exists():
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            if key and value and key not in os.environ:
                                os.environ[key] = value
        except Exception:
            pass

    def _load_api_key(self) -> Optional[str]:
        """从环境变量加载API密钥"""
        import os
        self._load_env()
        return os.getenv('DEEPSEEK_API_KEY')

    @abstractmethod
    def process(self, input_data: Any) -> AgentResponse:
        """处理输入并返回响应，必须由子类实现"""
        pass

    def _call_deepseek(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> Dict[str, Any]:
        """调用DeepSeek API"""
        if not self.session:
            raise RuntimeError(f"[{self.name}] 未配置API密钥，无法调用DeepSeek API")

        url = f"{self.api_base}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_tokens
        }

        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"[{self.name}] 请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"API请求失败: {e}")

        return {}

    def update_prompt(self, new_prompt: str):
        """更新智能体的系统提示词"""
        self.system_prompt = new_prompt
        self._prompt_history.append({
            'timestamp': datetime.now().isoformat(),
            'old_prompt': self._get_previous_prompt(),
            'new_prompt': new_prompt
        })

    def _get_previous_prompt(self) -> str:
        """从历史记录中获取上一个提示词"""
        if self._prompt_history:
            return self._prompt_history[-1]['new_prompt']
        return self.system_prompt

    def get_prompt_history(self) -> List[Dict[str, str]]:
        """获取提示词变更历史"""
        return self._prompt_history

    def validate_input(self, input_data: Any) -> tuple[bool, Optional[str]]:
        """验证输入数据，可由子类覆盖"""
        return True, None

    def close(self):
        """关闭会话"""
        if self.session:
            self.session.close()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', model='{self.model}', enabled={self.enabled})>"
