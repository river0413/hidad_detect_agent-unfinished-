"""
DeepSeek API 集成模块
用于AI判断文本是否为正常文本、广告或隐性广告
"""
import os
import json
import time
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from env_config import get_config


@dataclass
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: int = 30


class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None):
        config = get_config()
        
        if api_key is None:
            api_key = config.get_api_key()
        
        if not api_key:
            raise ValueError(
                "DeepSeek API key is required. "
                "Please set DEEPSEEK_API_KEY in .env file or pass api_key parameter."
            )
        
        self.config = DeepSeekConfig(
            api_key=api_key,
            base_url=config.get_api_base(),
            model=config.get_model_name()
        )
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        })
    
    def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000) -> Dict:
        """
        发送聊天请求到DeepSeek API
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数，控制随机性
            max_tokens: 最大token数
        
        Returns:
            API响应字典
        """
        url = f"{self.config.base_url}/chat/completions"
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    url, 
                    json=payload, 
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                if attempt < self.config.max_retries - 1:
                    print(f"请求失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                    time.sleep(self.config.retry_delay)
                else:
                    raise Exception(f"API请求失败: {e}")
        
        return None
    
    def close(self):
        """关闭会话"""
        self.session.close()


class HiddenAdDetector:
    """
    隐性广告检测器 - 使用DeepSeek API进行AI判断
    """
    
    SYSTEM_PROMPT = """你是一个专业的文本分析专家，专门判断文本内容是否为广告或隐性广告。

请对给定的文本进行分析，并输出JSON格式的判断结果。

**分类定义：**
1. **正常文本**: 纯粹的表达个人情感、分享经历、交流观点的文字，不包含任何推广、销售或引导消费的内容
2. **显性广告**: 明确、直接的推广内容，包含明显的购买引导、价格信息、促销信息等
3. **隐性广告**: 以看似自然的方式包装的推广内容，通过分享、推荐、测评等形式的广告，让人难以察觉其推广意图

**输出格式（JSON）：**
{
    "category": "normal/ad/hidden_ad",
    "confidence": 0.85,
    "reasoning": "判断理由简要说明",
    "keyword_weights": {
        "promotion_words": 0.0-1.0,
        "price_mentions": 0.0-1.0,
        "urgency_expressions": 0.0-1.0,
        "brand_mentions": 0.0-1.0,
        "action_words": 0.0-1.0,
        "natural_expression": 0.0-1.0
    }
}

**判断标准：**
- 正常文本: 不包含任何商业推广元素
- 显性广告: 包含明确的价格、促销、购买引导
- 隐性广告: 通过分享、测评、推荐等形式包装，但实际有推广目的

请严格按JSON格式输出，不要输出其他内容。"""
    
    def __init__(self, api_key: Optional[str] = None):
        config = get_config()
        
        if api_key is None:
            api_key = config.get_api_key()
        
        if not api_key:
            print("警告: 未设置DEEPSEEK_API_KEY环境变量，将使用规则方法进行判断")
            self.client = None
        else:
            self.client = DeepSeekClient(api_key)
    
    def analyze_text_ai(self, text: str) -> Optional[Dict]:
        """
        使用AI分析文本
        
        Args:
            text: 待分析的文本
        
        Returns:
            分析结果字典，如果失败返回None
        """
        if self.client is None:
            return None
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"请分析以下文本:\n\n{text}"}
        ]
        
        try:
            response = self.client.chat(messages, temperature=0.3, max_tokens=500)
            
            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                
                result = json.loads(content.strip())
                
                return {
                    'category': result.get('category', 'normal'),
                    'confidence': float(result.get('confidence', 0.5)),
                    'reasoning': result.get('reasoning', ''),
                    'keyword_weights': result.get('keyword_weights', {}),
                    'source': 'deepseek_ai'
                }
        
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始内容: {content[:200] if 'content' in locals() else 'N/A'}")
        except Exception as e:
            print(f"AI分析失败: {e}")
        
        return None
    
    def analyze_text_rule(self, text: str) -> Dict:
        """
        使用规则方法分析文本（备用方案）
        
        Returns:
            分析结果字典
        """
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
        
        ad_score = 0
        for kw in promotion_keywords:
            if kw in text:
                ad_score += 1
        
        for kw in brand_keywords:
            if kw in text:
                ad_score += 2
        
        normal_score = 0
        for kw in normal_keywords:
            if kw in text:
                normal_score += 1
        
        ad_ratio = ad_score / (len(text) / 50 + 1)
        normal_ratio = normal_score / (len(text) / 30 + 1)
        
        keyword_weights = {
            'promotion_words': min(ad_score / 10, 1.0),
            'price_mentions': min(sum(1 for kw in ['价格', '多少钱', '性价比', '划算', '值得', '超值', '便宜', '实惠'] if kw in text) / 5, 1.0),
            'urgency_expressions': min(sum(1 for kw in ['赶紧', '快来', '马上', '立刻', '不要错过', '仅剩'] if kw in text) / 3, 1.0),
            'brand_mentions': min(sum(1 for kw in brand_keywords if kw in text) / 5, 1.0),
            'action_words': min(sum(1 for kw in ['点击', '扫码', '链接', '私信', '联系', '购买'] if kw in text) / 4, 1.0),
            'natural_expression': min(normal_score / 8, 1.0)
        }
        
        if ad_ratio > 3.0 and any(kw in text for kw in ['购买', '价格', '优惠', '促销']):
            category = 'ad'
            confidence = min(ad_ratio / (ad_ratio + normal_ratio + 1), 0.95)
        elif ad_ratio > 1.5 or any(kw in text for kw in ['推荐', '种草', '安利', '超赞']):
            if '我觉得' in text or '个人' in text or '朋友' in text:
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
    
    def analyze_text(self, text: str, use_ai: bool = True) -> Dict:
        """
        分析文本，首先尝试使用AI，失败时使用规则方法
        
        Args:
            text: 待分析的文本
            use_ai: 是否优先使用AI分析
        
        Returns:
            分析结果字典
        """
        if use_ai and self.client is not None:
            result = self.analyze_text_ai(text)
            if result is not None:
                return result
        
        return self.analyze_text_rule(text)
    
    def _analyze_single_text(self, text: str, index: int, use_ai: bool) -> Dict:
        """
        分析单条文本（用于并发处理）
        
        Args:
            text: 待分析的文本
            index: 原始索引
            use_ai: 是否使用AI分析
        
        Returns:
            分析结果字典
        """
        result = self.analyze_text(text, use_ai=use_ai)
        result['text'] = text
        result['index'] = index
        return result
    
    def batch_analyze_texts_concurrent(self, texts: List[str], batch_size: int = 50, 
                                      delay_between_batches: float = 2.0,
                                      max_workers: int = 4) -> List[Dict]:
        """
        并发批量分析文本
        
        Args:
            texts: 文本列表
            batch_size: 每批处理的文本数
            delay_between_batches: 批次之间的延迟（秒）
            max_workers: 最大并发线程数
        
        Returns:
            分析结果列表（保持原始顺序）
        """
        results = []
        total_texts = len(texts)
        total_batches = (total_texts + batch_size - 1) // batch_size
        use_ai = self.client is not None
        
        print(f"\n并发分析模式 - 最大线程数: {max_workers}")
        print(f"总文本数: {total_texts}, 总批次数: {total_batches}")
        
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            current_batch = i // batch_size + 1
            
            print(f"\n处理批次 {current_batch}/{total_batches} ({len(batch_texts)} 条文本)")
            
            batch_results = [None] * len(batch_texts)
            
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_texts))) as executor:
                future_to_index = {}
                
                for j, text in enumerate(batch_texts):
                    future = executor.submit(
                        self._analyze_single_text,
                        text,
                        i + j,
                        use_ai
                    )
                    future_to_index[future] = j
                
                completed_count = 0
                for future in as_completed(future_to_index):
                    j = future_to_index[future]
                    try:
                        result = future.result()
                        batch_results[j] = result
                    except Exception as e:
                        print(f"  文本 {j + 1} 处理失败: {e}")
                        batch_results[j] = {
                            'text': batch_texts[j],
                            'index': i + j,
                            'category': 'normal',
                            'confidence': 0.5,
                            'reasoning': '分析失败',
                            'keyword_weights': {},
                            'source': 'error'
                        }
                    
                    completed_count += 1
                    if completed_count % max(1, len(batch_texts) // 5) == 0:
                        print(f"  已完成: {completed_count}/{len(batch_texts)}")
            
            results.extend(batch_results)
            
            if i + batch_size < total_texts:
                print(f"  等待 {delay_between_batches} 秒...")
                time.sleep(delay_between_batches)
        
        return results
    
    def batch_analyze_texts(self, texts: List[str], batch_size: int = 50, 
                          delay_between_batches: float = 2.0,
                          use_concurrent: bool = True,
                          max_workers: int = 4) -> List[Dict]:
        """
        批量分析文本
        
        Args:
            texts: 文本列表
            batch_size: 每批处理的文本数
            delay_between_batches: 批次之间的延迟（秒）
            use_concurrent: 是否使用并发处理
            max_workers: 最大并发线程数（仅在并发模式下生效）
        
        Returns:
            分析结果列表
        """
        if use_concurrent:
            return self.batch_analyze_texts_concurrent(
                texts, batch_size, delay_between_batches, max_workers
            )
        
        results = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            current_batch = i // batch_size + 1
            
            print(f"处理批次 {current_batch}/{total_batches} ({len(batch_texts)} 条文本)")
            
            for j, text in enumerate(batch_texts):
                result = self.analyze_text(text, use_ai=(self.client is not None))
                result['text'] = text
                result['index'] = i + j
                results.append(result)
                
                if (j + 1) % 10 == 0:
                    print(f"  已处理 {j + 1}/{len(batch_texts)} 条")
            
            if i + batch_size < len(texts):
                print(f"  等待 {delay_between_batches} 秒...")
                time.sleep(delay_between_batches)
        
        return results


def batch_analyze_with_deepseek(
    texts: List[str],
    api_key: Optional[str] = None,
    batch_size: int = 50,
    delay_between_batches: float = 2.0,
    use_concurrent: bool = True,
    max_workers: int = 4
) -> List[Dict]:
    """
    批量使用DeepSeek API分析文本
    
    Args:
        texts: 文本列表
        api_key: DeepSeek API密钥（可选，将从.env加载）
        batch_size: 每批处理的文本数
        delay_between_batches: 批次之间的延迟（秒）
        use_concurrent: 是否使用并发处理
        max_workers: 最大并发线程数
    
    Returns:
        分析结果列表
    """
    detector = HiddenAdDetector(api_key)
    
    return detector.batch_analyze_texts(
        texts, 
        batch_size, 
        delay_between_batches,
        use_concurrent=use_concurrent,
        max_workers=max_workers
    )


if __name__ == "__main__":
    print("DeepSeek API 隐性广告检测模块")
    print("=" * 60)
    
    config = get_config()
    api_key = config.get_api_key()
    
    if not api_key:
        print("未设置DEEPSEEK_API_KEY")
        print("请在 .env 文件中设置您的API密钥")
        print("\n将使用规则方法进行演示")
    
    detector = HiddenAdDetector(api_key if api_key else None)
    
    test_texts = [
        "今天天气真好，和朋友一起去逛街，心情特别棒！",
        "【限时优惠】Nike运动鞋特价399元，错过再等一年！点击购买>>",
        "这款面膜真的太好用了！敷完之后皮肤又白又嫩，朋友推荐给我的，效果真的很明显！"
    ]
    
    print("\n测试文本分析:")
    for text in test_texts:
        print(f"\n文本: {text[:50]}...")
        result = detector.analyze_text(text, use_ai=False)
        print(f"分类: {result['category']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"词语权重: {result['keyword_weights']}")
        print("-" * 60)
