"""
模型推理服务封装模块
提供统一的推理接口，可被Flask等后端框架直接调用
"""
import os
import sys
import json
import warnings
import logging
import io
from typing import Optional, Dict, List, Any

os.environ['FLAGS_disable_dyelibrary_warnings'] = 'true'
os.environ['GLOG_v'] = '0'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_log_dir'] = 'NUL'
os.environ['BACKEND'] = 'op'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.ERROR)
logging.getLogger('paddlenlp').setLevel(logging.ERROR)
logging.getLogger('paddle').setLevel(logging.ERROR)
logging.getLogger('paddle.base').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
from env_config import get_config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
config = get_config()


class HiddenAdClassifier(nn.Layer):
    def __init__(self, pretrained_name: str, num_classes: int = 3, dropout_prob: float = 0.1):
        super().__init__()
        self.ernie = ErnieModel.from_pretrained(pretrained_name)
        self.dropout = nn.Dropout(dropout_prob)
        hidden_size = 768
        self.label_classifier = nn.Linear(hidden_size, num_classes)
        self.confidence_regressor = nn.Linear(hidden_size, 1)
        self.weight_predictor = nn.Linear(hidden_size, 6)
        self.activation = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids):
        outputs = self.ernie(input_ids=input_ids, token_type_ids=token_type_ids)
        pooled = outputs[1]
        pooled = self.dropout(pooled)
        label_logits = self.label_classifier(pooled)
        confidence = self.activation(self.confidence_regressor(pooled))
        keyword_weights = self.activation(self.weight_predictor(pooled))
        return label_logits, confidence, keyword_weights


class ErnieTextClassifier:
    """
    ERNIE文本分类器 - 将ERNIE推理模型封装到类中
    """
    
    def __init__(self, model_path: str = None):
        """
        初始化分类器
        
        Args:
            model_path: 模型路径，默认为配置中的路径
        """
        self._model = None
        self._tokenizer = None
        self._config_data = None
        self._model_path = model_path if model_path else os.path.join(PROJECT_ROOT, 'hidden_ad_model')
        self._max_seq_length = config.get_model_config().get('max_seq_length', 256)
        
        self._label_map = {0: 'normal', 1: 'ad', 2: 'hidden_ad'}
        self._label_names = {'normal': '正常文本', 'ad': '广告', 'hidden_ad': '隐性广告'}
        self._weight_names = [
            'promotion_words', 'price_mentions', 'urgency_expressions',
            'brand_mentions', 'action_words', 'natural_expression'
        ]
    
    def _suppress_logs(self):
        """临时抑制paddle/paddlenlp日志输出"""
        devnull = io.StringIO()
        old_stderr = sys.stderr
        old_stdout = sys.stdout

        def suppress():
            sys.stderr = devnull
            sys.stdout = devnull

        def restore():
            sys.stderr = old_stderr
            sys.stdout = old_stdout

        return suppress, restore
    
    def load(self) -> bool:
        """
        加载模型
        
        Returns:
            是否加载成功
        """
        if self._model is not None:
            return True
        
        if not os.path.exists(self._model_path):
            print(f"警告: 模型目录不存在 {self._model_path}")
            return False
        
        suppress_fn, restore_fn = self._suppress_logs()
        suppress_fn()
        
        try:
            config_path = os.path.join(self._model_path, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                self._config_data = json.load(f)

            self._tokenizer = ErnieTokenizer.from_pretrained(self._model_path)
            self._model = HiddenAdClassifier(self._config_data["pretrained_model"], num_classes=3)
            model_state = paddle.load(os.path.join(self._model_path, "model_state.pdparams"))
            self._model.set_state_dict(model_state)
            self._model.eval()
            
            print(f"[OK] ERNIE模型已加载: {self._config_data.get('pretrained_model', 'unknown')}")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
        finally:
            restore_fn()
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model is not None
    
    def predict(self, text: str) -> Optional[Dict[str, Any]]:
        """
        预测单条文本
        
        Args:
            text: 待预测的文本
        
        Returns:
            预测结果字典，如果失败返回None
        """
        if not text or not isinstance(text, str):
            return None
        
        if self._model is None:
            if not self.load():
                return None
        
        device = paddle.get_device()
        if paddle.is_compiled_with_cuda():
            device = 'gpu:0'
        else:
            device = 'cpu'
        paddle.set_device(device)

        encoded = self._tokenizer(
            text,
            max_length=self._max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pd'
        )

        input_ids = encoded['input_ids'].to(device)
        token_type_ids = encoded['token_type_ids'].to(device)

        with paddle.no_grad():
            label_logits, confidence, keyword_weights = self._model(input_ids, token_type_ids)

        predicted_label = int(paddle.argmax(label_logits, axis=1).item())
        confidence_score = float(confidence.item())
        weights_list = keyword_weights.numpy()[0].tolist()

        weights_dict = {name: round(weight, 4) for name, weight in zip(self._weight_names, weights_list)}

        probs = paddle.nn.functional.softmax(label_logits, axis=1)[0]

        return {
            'text': text,
            'text_type': self._label_map[predicted_label],
            'text_type_name': self._label_names[self._label_map[predicted_label]],
            'confidence': round(confidence_score, 4),
            'keyword_weights': weights_dict,
            'raw_logits': label_logits.numpy()[0].tolist(),
            'all_probabilities': {
                'normal': round(float(probs[0].item()), 4),
                'ad': round(float(probs[1].item()), 4),
                'hidden_ad': round(float(probs[2].item()), 4)
            }
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        批量预测文本
        
        Args:
            texts: 待预测的文本列表
        
        Returns:
            预测结果列表
        """
        results = []
        for text in texts:
            result = self.predict(text)
            if result:
                results.append(result)
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """获取模型状态"""
        return {
            'model_loaded': self.is_loaded(),
            'model_name': self._config_data.get('pretrained_model', 'unknown') if self._config_data else '未加载',
            'model_path': self._model_path,
            'max_seq_length': self._max_seq_length,
            'detector_ready': True
        }
    
    def unload(self):
        """卸载模型，释放资源"""
        if self._model is not None:
            try:
                paddle.device.cuda.empty_cache()
            except Exception:
                pass
            
            self._model = None
            self._tokenizer = None
            self._config_data = None
            
            print("[OK] ERNIE模型资源已释放")


# 全局单例实例
_ernie_classifier = None


def get_ernie_classifier() -> ErnieTextClassifier:
    """获取ERNIE分类器单例"""
    global _ernie_classifier
    if _ernie_classifier is None:
        _ernie_classifier = ErnieTextClassifier()
    return _ernie_classifier


def init_model(model_path: str = None) -> bool:
    """
    初始化模型（延迟加载）
    
    Args:
        model_path: 模型路径
    
    Returns:
        是否初始化成功
    """
    classifier = get_ernie_classifier()
    if model_path:
        classifier._model_path = model_path
    return classifier.load()


def detect_text(text: str, use_ai: bool = True) -> Dict[str, Any]:
    """
    检测单条文本是否为隐性广告
    
    Args:
        text: 待检测的文本
        use_ai: 是否使用AI模型（本地模型），False时使用规则方法
    
    Returns:
        检测结果字典
    """
    if not text or not isinstance(text, str):
        return {
            'text': text,
            'text_type': 'normal',
            'text_type_name': '正常文本',
            'confidence': 0.0,
            'keyword_weights': {},
            'all_probabilities': {'normal': 1.0, 'ad': 0.0, 'hidden_ad': 0.0},
            'source': 'invalid_input',
            'error': '输入文本为空或格式错误'
        }

    result = {}

    if use_ai:
        classifier = get_ernie_classifier()
        if not classifier.is_loaded():
            load_success = classifier.load()
            if not load_success:
                use_ai = False
        else:
            try:
                result = classifier.predict(text)
                if result:
                    result['source'] = 'model'
                    return result
                else:
                    use_ai = False
            except Exception as e:
                print(f"AI模型推理失败: {e}")
                use_ai = False

    if not use_ai:
        from deepseek_classifier import HiddenAdDetector

        detector = HiddenAdDetector()
        analysis = detector.analyze_text(text, use_ai=False)

        label_names = {'normal': '正常文本', 'ad': '广告', 'hidden_ad': '隐性广告'}

        result = {
            'text': text,
            'text_type': analysis.get('category', 'normal'),
            'text_type_name': label_names.get(analysis.get('category', 'normal'), '未知'),
            'confidence': analysis.get('confidence', 0.5),
            'keyword_weights': analysis.get('keyword_weights', {}),
            'all_probabilities': {
                'normal': analysis.get('confidence', 0.5) if analysis.get('category') == 'normal' else 0.0,
                'ad': analysis.get('confidence', 0.5) if analysis.get('category') == 'ad' else 0.0,
                'hidden_ad': analysis.get('confidence', 0.5) if analysis.get('category') == 'hidden_ad' else 0.0
            },
            'source': analysis.get('source', 'rule_based'),
            'reasoning': analysis.get('reasoning', '')
        }

    return result


def batch_detect_texts(texts: List[str], use_ai: bool = True) -> List[Dict[str, Any]]:
    """
    批量检测文本
    
    Args:
        texts: 待检测的文本列表
        use_ai: 是否使用AI模型
    
    Returns:
        检测结果列表
    """
    results = []
    
    if use_ai:
        classifier = get_ernie_classifier()
        if not classifier.is_loaded():
            classifier.load()
    
    for text in texts:
        result = detect_text(text, use_ai=use_ai)
        results.append(result)
    
    return results


def get_model_status() -> Dict[str, Any]:
    """
    获取模型状态信息
    
    Returns:
        模型状态字典
    """
    classifier = get_ernie_classifier()
    return classifier.get_status()


def shutdown():
    """
    关闭推理服务，清理资源
    """
    global _ernie_classifier
    
    if _ernie_classifier is not None:
        _ernie_classifier.unload()
        _ernie_classifier = None


if __name__ == "__main__":
    print("测试推理服务封装模块")
    print("=" * 60)
    
    test_texts = [
        "今天天气真好，和朋友一起去公园散步，心情特别愉快！",
        "【限时特价】Nike运动鞋仅售299元！立即点击购买>>",
        "这款面膜真的太好用了！朋友推荐给我的，用完之后皮肤又白又嫩，效果非常明显！"
    ]
    
    print("\n1. 获取模型状态:")
    status = get_model_status()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    
    print("\n2. 检测单条文本:")
    result = detect_text(test_texts[0], use_ai=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n3. 批量检测文本:")
    results = batch_detect_texts(test_texts, use_ai=True)
    for i, res in enumerate(results):
        print(f"\n文本 {i+1}:")
        print(json.dumps(res, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 60)
    print("测试完成！")