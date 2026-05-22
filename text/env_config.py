"""
环境配置加载模块
从.env文件加载配置
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any


class EnvConfig:
    """环境配置管理类"""
    
    _instance: Optional['EnvConfig'] = None
    _config: Dict[str, Any] = {}
    _loaded: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._loaded:
            self.load_env()
    
    def load_env(self, env_file: str = None):
        """从.env文件加载配置"""
        if env_file is None:
            env_file = Path(__file__).parent / ".env"
        
        if not os.path.exists(env_file):
            print(f"警告: 配置文件 {env_file} 不存在")
            return
        
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                if '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                self._config[key] = value
                os.environ[key] = value
        
        self._loaded = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config.get(key, os.environ.get(key, default))
    
    def get_required(self, key: str) -> str:
        """获取必需的配置值"""
        value = self.get(key)
        if value is None or value == '':
            raise ValueError(f"必需的配置项 '{key}' 未设置或为空")
        return value
    
    def is_loaded(self) -> bool:
        """检查配置是否已加载"""
        return self._loaded
    
    def get_api_key(self) -> Optional[str]:
        """获取DeepSeek API密钥"""
        return self.get('DEEPSEEK_API_KEY', 'sk-bb13768525af4da3bcba5bc6481ded6e')
    
    def get_api_base(self) -> str:
        """获取API基础URL"""
        return self.get('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.get('DEEPSEEK_MODEL', 'deepseek-v4-pro')
    
    def get_corpus_path(self) -> str:
        """获取语料库路径"""
        return self.get('CORPUS_BASE_PATH', r'G:\projects\agent\text\ChineseNlpCorpus-master\datasets')
    
    def get_output_dir(self) -> str:
        """获取输出目录"""
        return self.get('OUTPUT_DIR', r'G:\projects\agent\text\processed_data')
    
    def get_total_samples(self) -> int:
        """获取样本总数"""
        return int(self.get('TOTAL_SAMPLES', '10000'))
    
    def get_test_ratio(self) -> float:
        """获取测试集比例"""
        return float(self.get('TEST_RATIO', '0.2'))
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'model_name': self.get('MODEL_NAME', 'ernie-3.0-medium-zh'),
            'max_seq_length': int(self.get('MAX_SEQ_LENGTH', '256')),
            'batch_size': int(self.get('BATCH_SIZE', '16')),
            'epochs': int(self.get('EPOCHS', '10')),
            'learning_rate': float(self.get('LEARNING_RATE', '5e-5'))
        }
    
    def get_model_path(self) -> str:
        """获取模型路径"""
        return self.get('MODEL_PATH', r'G:\projects\agent\text\hidden_ad_model')


def load_env_config(env_file: str = None) -> EnvConfig:
    """加载环境配置"""
    config = EnvConfig()
    config.load_env(env_file)
    return config


def get_config() -> EnvConfig:
    """获取配置实例"""
    return EnvConfig()
