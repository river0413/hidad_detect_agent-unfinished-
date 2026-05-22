"""
NLP语料库数据处理脚本
使用DeepSeek API + 规则方法对文本进行分类
分类为：正常文本、广告、隐性广告
"""
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from deepseek_classifier import HiddenAdDetector
from env_config import get_config


class TextDataProcessor:
    def __init__(self, corpus_base_path: str = None, use_ai: bool = True):
        config = get_config()
        
        if corpus_base_path is None:
            corpus_base_path = config.get_corpus_path()
        
        self.corpus_base_path = Path(corpus_base_path)
        self.use_ai = use_ai
        
        self.classifier = HiddenAdDetector()
        
        self.category_names = {
            'normal': '正常文本',
            'ad': '广告',
            'hidden_ad': '隐性广告'
        }

    def load_weibo_data(self, sample_size: int = 4000) -> List[Dict]:
        """加载微博情感数据"""
        file_path = self.corpus_base_path / "weibo_senti_100k" / "weibo_senti_100k" / "weibo_senti_100k.csv"
        if not file_path.exists():
            print(f"警告: 文件不存在 {file_path}")
            return []
        
        try:
            df = pd.read_csv(file_path)
            df = df[df['review'].notna() & (df['review'].str.len() > 10)]
            df = df.sample(min(sample_size, len(df)), random_state=42)
            
            return [
                {'text': str(row['review']), 'source': 'weibo', 'original_label': int(row['label'])}
                for _, row in df.iterrows()
            ]
        except Exception as e:
            print(f"加载微博数据失败: {e}")
            return []

    def load_shopping_data(self, sample_size: int = 4000) -> List[Dict]:
        """加载购物评论数据"""
        file_path = self.corpus_base_path / "online_shopping_10_cats" / "online_shopping_10_cats.csv"
        if not file_path.exists():
            print(f"警告: 文件不存在 {file_path}")
            return []
        
        try:
            df = pd.read_csv(file_path)
            df = df[df['review'].notna() & (df['review'].str.len() > 10)]
            df = df.sample(min(sample_size, len(df)), random_state=42)
            
            return [
                {'text': str(row['review']), 'source': 'shopping', 
                 'category': row['cat'], 'original_label': int(row['label'])}
                for _, row in df.iterrows()
            ]
        except Exception as e:
            print(f"加载购物数据失败: {e}")
            return []

    def load_waimai_data(self, sample_size: int = 2000) -> List[Dict]:
        """加载外卖评论数据"""
        file_path = self.corpus_base_path / "waimai_10k" / "waimai_10k.csv"
        if not file_path.exists():
            print(f"警告: 文件不存在 {file_path}")
            return []
        
        try:
            df = pd.read_csv(file_path)
            df = df[df['review'].notna() & (df['review'].str.len() > 10)]
            df = df.sample(min(sample_size, len(df)), random_state=42)
            
            return [
                {'text': str(row['review']), 'source': 'waimai', 'original_label': int(row['label'])}
                for _, row in df.iterrows()
            ]
        except Exception as e:
            print(f"加载外卖数据失败: {e}")
            return []

    def load_all_data(self, total_size: int = 10000) -> List[Dict]:
        """加载并整合所有数据"""
        print("开始加载数据集...")
        
        all_data = []
        
        weibo_size = int(total_size * 0.4)
        shopping_size = int(total_size * 0.4)
        waimai_size = total_size - weibo_size - shopping_size
        
        weibo_data = self.load_weibo_data(weibo_size)
        all_data.extend(weibo_data)
        print(f"微博数据: {len(weibo_data)} 条")
        
        shopping_data = self.load_shopping_data(shopping_size)
        all_data.extend(shopping_data)
        print(f"购物数据: {len(shopping_data)} 条")
        
        waimai_data = self.load_waimai_data(waimai_size)
        all_data.extend(waimai_data)
        print(f"外卖数据: {len(waimai_data)} 条")
        
        random.shuffle(all_data)
        print(f"总共加载: {len(all_data)} 条数据")
        
        return all_data[:total_size]

    def classify_text(self, text: str) -> Dict:
        """
        对文本进行分类
        
        Returns:
            包含分类结果的字典
        """
        result = self.classifier.analyze_text(text, use_ai=self.use_ai)
        
        return {
            'text': text,
            'category': result['category'],
            'category_name': self.category_names.get(result['category'], '未知'),
            'confidence': result['confidence'],
            'reasoning': result.get('reasoning', ''),
            'keyword_weights': result['keyword_weights'],
            'source_type': result.get('source', 'unknown')
        }

    def process_dataset(self, data: List[Dict], output_path: str = None, 
                       use_ai: bool = False, ai_batch_size: int = 50) -> List[Dict]:
        """
        处理整个数据集
        
        Args:
            data: 原始数据列表
            output_path: 输出文件路径
            use_ai: 是否使用AI分类
            ai_batch_size: AI批量处理的批次大小
        """
        print(f"开始处理数据集 (使用{'AI' if use_ai else '规则'}方法)...")
        
        processed_data = []
        category_stats = {'normal': 0, 'ad': 0, 'hidden_ad': 0}
        
        if use_ai and self.classifier.client is not None:
            texts = [item['text'] for item in data]
            ai_results = self.classifier.batch_analyze_texts(texts, batch_size=ai_batch_size)
            
            for i, item in enumerate(data):
                ai_result = ai_results[i]
                
                processed_item = {
                    'text': item['text'],
                    'category': ai_result['category'],
                    'category_name': self.category_names.get(ai_result['category'], '未知'),
                    'confidence': ai_result['confidence'],
                    'reasoning': ai_result.get('reasoning', ''),
                    'keyword_weights': ai_result['keyword_weights'],
                    'source': item.get('source', 'unknown'),
                    'original_label': item.get('original_label'),
                    'category_field': item.get('category')
                }
                
                processed_data.append(processed_item)
                category_stats[ai_result['category']] += 1
        else:
            for i, item in enumerate(data):
                text = item['text']
                result = self.classify_text(text)
                
                processed_item = {
                    'text': item['text'],
                    'category': result['category'],
                    'category_name': result['category_name'],
                    'confidence': result['confidence'],
                    'reasoning': result['reasoning'],
                    'keyword_weights': result['keyword_weights'],
                    'source': item.get('source', 'unknown'),
                    'original_label': item.get('original_label'),
                    'category_field': item.get('category')
                }
                
                processed_data.append(processed_item)
                category_stats[result['category']] += 1
                
                if (i + 1) % 1000 == 0:
                    print(f"已处理 {i + 1}/{len(data)} 条数据")
                    print(f"  正常文本: {category_stats['normal']} 条")
                    print(f"  广告: {category_stats['ad']} 条")
                    print(f"  隐性广告: {category_stats['hidden_ad']} 条")
        
        print(f"\n处理完成! 总计 {len(processed_data)} 条")
        
        if len(processed_data) > 0:
            print(f"正常文本: {category_stats['normal']} 条 ({category_stats['normal']/len(processed_data)*100:.1f}%)")
            print(f"广告: {category_stats['ad']} 条 ({category_stats['ad']/len(processed_data)*100:.1f}%)")
            print(f"隐性广告: {category_stats['hidden_ad']} 条 ({category_stats['hidden_ad']/len(processed_data)*100:.1f}%)")
        else:
            print("警告: 处理后数据为空")
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            print(f"数据已保存到: {output_path}")
        
        return processed_data

    def split_dataset(self, data: List[Dict], test_ratio: float = 0.2, 
                     output_dir: str = None) -> Tuple[List[Dict], List[Dict]]:
        """
        划分训练集和测试集 (默认2:8比例)
        """
        random.shuffle(data)
        
        split_idx = int(len(data) * test_ratio)
        test_data = data[:split_idx]
        train_data = data[split_idx:]
        
        print(f"\n数据集划分完成:")
        print(f"测试集: {len(test_data)} 条 ({test_ratio*100:.0f}%)")
        print(f"训练集: {len(train_data)} 条 ({(1-test_ratio)*100:.0f}%)")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, 'test_data.json'), 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            with open(os.path.join(output_dir, 'train_data.json'), 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            print(f"数据已保存到: {output_dir}")
        
        return train_data, test_data

    def prepare_for_ernie_training(self, data: List[Dict], output_path: str):
        """
        准备适用于ERNIE3.0训练的数据格式
        输出: 文本 + 类别 + 置信度 + 词语权重
        """
        print(f"准备ERNIE训练数据...")
        
        category_to_label = {
            'normal': 0,
            'ad': 1,
            'hidden_ad': 2
        }
        
        ernie_data = []
        for item in data:
            training_item = {
                'text': item['text'],
                'category': item['category'],
                'category_name': item['category_name'],
                'label': category_to_label.get(item['category'], 0),
                'confidence': item['confidence'],
                'keyword_weights': item['keyword_weights'],
                'reasoning': item.get('reasoning', ''),
                'source': item.get('source', 'unknown')
            }
            ernie_data.append(training_item)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ernie_data, f, ensure_ascii=False, indent=2)
        
        print(f"ERNIE训练数据已保存到: {output_path}")
        return ernie_data

    def generate_statistics(self, data: List[Dict], output_path: str = None) -> Dict:
        """生成数据统计信息"""
        stats = {
            'total_samples': len(data),
            'category_distribution': {
                'normal': {'count': 0, 'percentage': 0},
                'ad': {'count': 0, 'percentage': 0},
                'hidden_ad': {'count': 0, 'percentage': 0}
            },
            'confidence_stats': {
                'normal': {'avg': 0, 'min': 0, 'max': 0},
                'ad': {'avg': 0, 'min': 0, 'max': 0},
                'hidden_ad': {'avg': 0, 'min': 0, 'max': 0}
            },
            'keyword_weight_stats': {},
            'source_distribution': {}
        }
        
        category_data = {cat: [] for cat in ['normal', 'ad', 'hidden_ad']}
        
        for item in data:
            cat = item['category']
            category_data[cat].append(item)
            stats['category_distribution'][cat]['count'] += 1
            
            source = item.get('source', 'unknown')
            if source not in stats['source_distribution']:
                stats['source_distribution'][source] = 0
            stats['source_distribution'][source] += 1
        
        if len(data) > 0:
            for cat in ['normal', 'ad', 'hidden_ad']:
                count = stats['category_distribution'][cat]['count']
                stats['category_distribution'][cat]['percentage'] = round(count / len(data) * 100, 2)
                
                if category_data[cat]:
                    confidences = [item['confidence'] for item in category_data[cat]]
                    stats['confidence_stats'][cat] = {
                        'avg': round(sum(confidences) / len(confidences), 4),
                        'min': round(min(confidences), 4),
                        'max': round(max(confidences), 4)
                    }
        else:
            print("警告: generate_statistics 收到空数据")
        
        weight_keys = ['promotion_words', 'price_mentions', 'urgency_expressions', 
                       'brand_mentions', 'action_words', 'natural_expression']
        
        for weight_key in weight_keys:
            stats['keyword_weight_stats'][weight_key] = {
                cat: 0 for cat in ['normal', 'ad', 'hidden_ad']
            }
            
            for cat in ['normal', 'ad', 'hidden_ad']:
                weights = [item['keyword_weights'].get(weight_key, 0) 
                          for item in category_data[cat]]
                if weights:
                    stats['keyword_weight_stats'][weight_key][cat] = round(
                        sum(weights) / len(weights), 4
                    )
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"统计信息已保存到: {output_path}")
        
        return stats


def main():
    print("="*70)
    print("NLP语料库数据处理 - 三大类别分类系统")
    print("="*70)
    print("分类类别: 正常文本 | 广告 | 隐性广告")
    print("="*70)
    
    config = get_config()
    api_key = config.get_api_key()
    use_ai = api_key is not None and api_key != ''
    
    if use_ai:
        print("\n✓ 检测到DeepSeek API密钥，将使用AI进行分类")
        use_ai = True
    else:
        print("\n✗ 未设置DeepSeek API密钥，将使用规则方法进行分类")
        print("  如需使用AI分类，请在.env文件中设置DEEPSEEK_API_KEY")
        use_ai = False
    
    processor = TextDataProcessor(use_ai=use_ai)
    
    total_samples = config.get_total_samples()
    output_dir = config.get_output_dir()
    test_ratio = config.get_test_ratio()
    
    all_data = processor.load_all_data(total_size=total_samples)
    
    processed_data = processor.process_dataset(
        all_data,
        output_path=os.path.join(output_dir, 'classified_dataset.json'),
        use_ai=use_ai
    )
    
    train_data, test_data = processor.split_dataset(
        processed_data,
        test_ratio=test_ratio,
        output_dir=output_dir
    )
    
    processor.prepare_for_ernie_training(
        train_data,
        os.path.join(output_dir, 'ernie_train_data.json')
    )
    
    processor.prepare_for_ernie_training(
        test_data,
        os.path.join(output_dir, 'ernie_test_data.json')
    )
    
    stats = processor.generate_statistics(
        processed_data,
        os.path.join(output_dir, 'statistics.json')
    )
    
    print("\n" + "="*70)
    print("处理完成!")
    print("="*70)
    print(f"\n数据统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    print(f"\n类别分布:")
    for cat in ['normal', 'ad', 'hidden_ad']:
        print(f"  {processor.category_names[cat]}: {stats['category_distribution'][cat]['count']} 条 "
              f"({stats['category_distribution'][cat]['percentage']}%)")
    
    print(f"\n置信度统计:")
    for cat in ['normal', 'ad', 'hidden_ad']:
        s = stats['confidence_stats'][cat]
        print(f"  {processor.category_names[cat]}: 平均={s['avg']:.4f}, "
              f"最小={s['min']:.4f}, 最大={s['max']:.4f}")
    
    print(f"\n关键词权重统计:")
    for weight_key, cat_stats in stats['keyword_weight_stats'].items():
        print(f"  {weight_key}:")
        for cat in ['normal', 'ad', 'hidden_ad']:
            print(f"    {processor.category_names[cat]}: {cat_stats[cat]:.4f}")
    
    print("\n" + "="*70)
    print("生成的文件:")
    print(f"  - {output_dir}/classified_dataset.json (完整分类数据集)")
    print(f"  - {output_dir}/train_data.json (训练集)")
    print(f"  - {output_dir}/test_data.json (测试集)")
    print(f"  - {output_dir}/ernie_train_data.json (ERNIE训练数据)")
    print(f"  - {output_dir}/ernie_test_data.json (ERNIE测试数据)")
    print(f"  - {output_dir}/statistics.json (统计信息)")
    print("="*70)
    
    return processed_data


if __name__ == "__main__":
    processed_data = main()
