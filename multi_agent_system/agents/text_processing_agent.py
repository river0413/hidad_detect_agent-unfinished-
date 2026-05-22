"""Text Processing Agent - Uses ERNIE model from text folder"""
import sys
from pathlib import Path
from typing import Any, Dict, List

_text_dir = Path(__file__).resolve().parent.parent.parent / "text"
if str(_text_dir) not in sys.path:
    sys.path.insert(0, str(_text_dir))

from .base_agent import BaseAgent, AgentResponse
from ..config.agent_config import TextProcessingConfig


class TextProcessingAgent(BaseAgent):
    """Agent for processing and classifying text using ERNIE model"""

    def __init__(self, config: TextProcessingConfig):
        super().__init__(config)
        self._classifier = None

    def _get_classifier(self):
        """Lazy load the ERNIE classifier"""
        if self._classifier is None:
            try:
                from inference_service import get_ernie_classifier, init_model
                self._classifier = get_ernie_classifier()
                self._classifier.load()
            except Exception as e:
                print(f"Warning: Could not load ERNIE model: {e}")
                self._classifier = None
        return self._classifier

    def process(self, input_data: Any) -> AgentResponse:
        """Process text input and classify it"""
        try:
            if isinstance(input_data, dict):
                text = input_data.get('text', '')
                use_ai = input_data.get('use_ai', True)
            elif isinstance(input_data, str):
                text = input_data
                use_ai = True
            else:
                return AgentResponse(
                    success=False,
                    error=f"Invalid input type: {type(input_data)}",
                    agent_name=self.name
                )

            if not text:
                return AgentResponse(
                    success=False,
                    error="Empty text provided",
                    agent_name=self.name
                )

            classifier = self._get_classifier()

            if classifier and classifier.is_loaded():
                # Use ERNIE model
                from inference_service import detect_text
                result = detect_text(text, use_ai=use_ai)
                
                # Convert to expected format
                result['category'] = result.get('text_type', 'normal')
                result['reasoning'] = f"ERNIE模型预测结果，置信度: {result.get('confidence', 0):.2f}"
                result['source'] = result.get('source', 'model')
                
            else:
                # Fallback to rule-based
                result = self._rule_based_classification(text)

            result['text'] = text
            result['agent'] = self.name

            return AgentResponse(
                success=True,
                data=result,
                agent_name=self.name,
                metadata={'model': 'ERNIE' if (classifier and classifier.is_loaded()) else 'rule_based'}
            )

        except Exception as e:
            import traceback
            print(f"Error in text processing: {e}")
            traceback.print_exc()
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name
            )

    def _rule_based_classification(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based classification"""
        promotion_keywords = [
            '购买', '下单', '立即', '抢购', '限时', '折扣', '优惠', '特价',
            '爆款', '热卖', '推荐', '种草', '安利', '必买', '超赞', '强烈推荐',
            '价格', '多少钱', '性价比', '划算', '值得', '超值', '便宜', '实惠',
            '赶紧', '快来', '马上', '立刻', '不要错过', '仅剩', '名额有限',
            '点击', '扫码', '链接', '私信', '联系', '购买', '下单',
            '淘宝', '天猫', '京东', '拼多多', '官网', '小程序', 'APP'
        ]

        brand_keywords = [
            '品牌', '官方', '正品', '旗舰店', '专营', '授权', '代理',
            '招商', '加盟', '合作', '货源', '批发', '一件代发'
        ]

        normal_keywords = [
            '个人', '我觉得', '我认为', '感受', '体验', '心情', '日记',
            '分享', '记录', '吐槽', '生活', '工作', '学习', '朋友',
            '家人', '今天', '昨天', '明天', '周末', '假期'
        ]

        ad_score = sum(1 for kw in promotion_keywords if kw in text)
        brand_score = sum(2 for kw in brand_keywords if kw in text)
        normal_score = sum(1 for kw in normal_keywords if kw in text)

        keyword_weights = {
            'promotion_words': min(ad_score / 10, 1.0),
            'price_mentions': min(sum(1 for kw in ['价格', '多少钱', '性价比', '划算', '值得', '超值', '便宜', '实惠'] if kw in text) / 5, 1.0),
            'urgency_expressions': min(sum(1 for kw in ['赶紧', '快来', '马上', '立刻', '不要错过', '仅剩'] if kw in text) / 3, 1.0),
            'brand_mentions': min(brand_score / 5, 1.0),
            'action_words': min(sum(1 for kw in ['点击', '扫码', '链接', '私信', '联系', '购买'] if kw in text) / 4, 1.0),
            'natural_expression': min(normal_score / 8, 1.0)
        }

        ad_ratio = ad_score / (len(text) / 50 + 1)
        normal_ratio = normal_score / (len(text) / 30 + 1)

        if ad_ratio > 3.0 and any(kw in text for kw in ['购买', '价格', '优惠', '促销']):
            category = 'ad'
            confidence = min(ad_ratio / (ad_ratio + normal_ratio + 1), 0.95)
        elif ad_ratio > 1.5 or any(kw in text for kw in ['推荐', '种草', '安利', '超赞']):
            if any(kw in text for kw in ['我觉得', '个人', '朋友']):
                category = 'hidden_ad'
                confidence = min(ad_ratio / (ad_ratio + normal_ratio + 1), 0.90)
            else:
                category = 'ad'
                confidence = min(ad_ratio / (ad_ratio + normal_ratio + 1), 0.85)
        else:
            category = 'normal'
            confidence = min(normal_ratio / (ad_ratio + normal_ratio + 1), 0.90)

        if len(text) < 15 and ad_score > 0:
            category = 'ad'
            confidence = 0.85

        return {
            'category': category,
            'confidence': round(confidence, 4),
            'reasoning': '规则方法判断',
            'keyword_weights': {k: round(v, 4) for k, v in keyword_weights.items()},
            'source': 'rule_based'
        }

    def batch_process(self, texts: List[str]) -> List[AgentResponse]:
        """Process multiple texts"""
        from inference_service import batch_detect_texts
        try:
            results = batch_detect_texts(texts, use_ai=True)
            responses = []
            for result in results:
                result['category'] = result.get('text_type', 'normal')
                result['reasoning'] = f"ERNIE模型预测结果，置信度: {result.get('confidence', 0):.2f}"
                result['agent'] = self.name
                responses.append(
                    AgentResponse(
                        success=True,
                        data=result,
                        agent_name=self.name,
                        metadata={'model': 'ERNIE'}
                    )
                )
            return responses
        except:
            return [self.process(text) for text in texts]

    def validate_input(self, input_data: Any) -> tuple[bool, str | None]:
        """Validate input data"""
        if isinstance(input_data, dict):
            if 'text' not in input_data:
                return False, "Missing 'text' field in input"
            if not input_data['text']:
                return False, "Text field is empty"
        elif isinstance(input_data, str):
            if not input_data.strip():
                return False, "Empty text provided"
        else:
            return False, f"Invalid input type: {type(input_data)}"
        return True, None
