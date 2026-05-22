"""
平衡数据集构建脚本
按比例构建：正常文本:隐性广告:广告 = 5:4:1，共10000条
优先从现有数据集筛选，不足部分调用AI生成
"""
import json
import os
import random
import time
from typing import List, Dict, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from deepseek_classifier import HiddenAdDetector
from env_config import get_config


class BalancedDatasetBuilder:
    def __init__(self):
        config = get_config()
        self.output_dir = Path(config.get_output_dir())
        self.classifier = HiddenAdDetector()
        self.use_ai_for_generation = self.classifier.client is not None
        
        self.category_names = {
            'normal': '正常文本',
            'ad': '广告',
            'hidden_ad': '隐性广告'
        }
        
        # 目标分布：正常文本:隐性广告:广告 = 5:4:1
        self.target_ratio = {
            'normal': 0.5,
            'hidden_ad': 0.4,
            'ad': 0.1
        }
        
    def load_existing_classified_data(self) -> List[Dict]:
        """加载已分类的数据集"""
        classified_file = self.output_dir / 'classified_dataset.json'
        if not classified_file.exists():
            print(f"警告: 未找到已分类数据集 {classified_file}")
            return []
        
        with open(classified_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"加载了 {len(data)} 条已分类数据")
        return data
    
    def filter_by_category(self, data: List[Dict], category: str, count: int) -> List[Dict]:
        """按类别筛选数据"""
        filtered = [item for item in data if item['category'] == category]
        random.shuffle(filtered)
        return filtered[:count]
    
    def generate_normal_texts(self, count: int) -> List[str]:
        """生成正常文本（使用规则方法生成）"""
        print(f"\n生成 {count} 条正常文本...")
        
        normal_templates = [
            "今天天气{weather}，{activity}，心情{feeling}。",
            "最近在{doing}，感觉{feeling}。",
            "{time}和{who}一起{activity}，很开心。",
            "刚看完{what}，觉得{feeling}，推荐给大家。",
            "{feeling}的一天，{activity}让我很放松。",
            "周末打算{plan}，希望能{wish}。",
            "最近学习{subject}，{feeling}但很有收获。",
            "和朋友聊了聊{topic}，觉得{feeling}。",
            "今天吃了{food}，味道{feeling}。",
            "看了{movie}，{feeling}，值得推荐。"
        ]
        
        weather_options = ["很好", "不错", "晴朗", "有点阴", "下雨了", "很舒服"]
        activity_options = ["出去散步", "在家看书", "和朋友聚餐", "看电影", "运动", "听音乐"]
        feeling_options = ["很愉快", "不错", "挺好的", "一般", "有点无聊", "很充实"]
        doing_options = ["学习Python", "看论文", "做项目", "锻炼身体", "学做菜"]
        time_options = ["昨天", "今天", "周末", "早上", "晚上"]
        who_options = ["家人", "朋友", "同事", "同学"]
        what_options = ["一本书", "一部电影", "一篇文章", "一个展览"]
        plan_options = ["去公园", "爬山", "看展览", "购物", "休息"]
        wish_options = ["顺利", "玩得开心", "有收获"]
        subject_options = ["机器学习", "深度学习", "数据分析", "编程"]
        topic_options = ["工作", "生活", "学习", "未来规划"]
        food_options = ["火锅", "烧烤", "川菜", "日料", "家常菜"]
        movie_options = ["一部电影", "一部纪录片", "一部动画"]
        
        generated = []
        for _ in range(count):
            template = random.choice(normal_templates)
            text = template.format(
                weather=random.choice(weather_options),
                activity=random.choice(activity_options),
                feeling=random.choice(feeling_options),
                doing=random.choice(doing_options),
                time=random.choice(time_options),
                who=random.choice(who_options),
                what=random.choice(what_options),
                plan=random.choice(plan_options),
                wish=random.choice(wish_options),
                subject=random.choice(subject_options),
                topic=random.choice(topic_options),
                food=random.choice(food_options),
                movie=random.choice(movie_options)
            )
            generated.append(text)
        
        return generated
    
    def generate_ad_texts(self, count: int) -> List[str]:
        """生成显性广告文本"""
        print(f"\n生成 {count} 条显性广告文本...")
        
        ad_templates = [
            "【限时特惠】{product}仅售{price}元，{urgency}，点击{action}！",
            "{product}大促销，{price}元起，{urgency}，{action}抢购！",
            "{brand}{product}特价{price}，{urgency}，{action}购买！",
            "爆款{product}限时{discount}，{price}元，{urgency}！",
            "{product}新品上市，{price}元，{urgency}，{action}！"
        ]
        
        product_options = [
            "运动鞋", "T恤", "手机壳", "面膜", "护肤品", "充电宝",
            "蓝牙耳机", "保温杯", "笔记本", "背包", "雨伞", "手表"
        ]
        price_options = ["99", "199", "299", "399", "499", "59"]
        urgency_options = ["限时抢购", "仅剩{num}件", "错过再等一年", "先到先得"]
        action_options = ["立即", "马上", "扫码", "点击链接"]
        brand_options = ["Nike", "Apple", "华为", "小米", "OPPO", "vivo"]
        discount_options = ["5折", "买一送一", "满减优惠", "买二送一"]
        
        generated = []
        for _ in range(count):
            template = random.choice(ad_templates)
            text = template.format(
                product=random.choice(product_options),
                price=random.choice(price_options),
                urgency=random.choice(urgency_options).format(num=random.randint(10, 100)),
                action=random.choice(action_options),
                brand=random.choice(brand_options),
                discount=random.choice(discount_options)
            )
            generated.append(text)
        
        return generated
    
    def generate_hidden_ad_texts(self, count: int) -> List[str]:
        """生成隐性广告文本（以分享/推荐形式包装）"""
        print(f"\n生成 {count} 条隐性广告文本...")
        
        hidden_ad_templates = [
            "最近发现一款{product}，用了之后{effect}，推荐给大家！",
            "{time}入手了{product}，{feeling}，朋友们都说好！",
            "被{who}种草了{product}，{effect}，强烈安利！",
            "试了很多{category}，这款{product}真的{effect}，爱了！",
            "{feeling}！{product}太好用了，{effect}，推荐！"
        ]
        
        product_options = [
            "护肤品", "面膜", "洗面奶", "精华液", "面霜",
            "笔记本电脑", "键盘", "鼠标", "显示器",
            "运动鞋", "休闲鞋", "背包", "保温杯"
        ]
        effect_options = [
            "皮肤变好了", "效果很明显", "颜值超高", "性价比很高",
            "用着很舒服", "质量很好", "续航超久", "设计很用心"
        ]
        time_options = ["上周", "最近", "前几天", "昨天"]
        feeling_options = ["太惊喜了", "真的不错", "超出预期", "很满意"]
        who_options = ["朋友", "闺蜜", "同事", "博主"]
        category_options = ["品牌", "产品", "款式"]
        
        generated = []
        for _ in range(count):
            template = random.choice(hidden_ad_templates)
            text = template.format(
                product=random.choice(product_options),
                effect=random.choice(effect_options),
                time=random.choice(time_options),
                feeling=random.choice(feeling_options),
                who=random.choice(who_options),
                category=random.choice(category_options)
            )
            generated.append(text)
        
        return generated
    
    def _generate_single_ai_text(self, category: str, prompt: str, max_retries: int = 3) -> Dict:
        """生成单条AI文本（用于并发处理）"""
        messages = [
            {"role": "system", "content": "你是一个专业的文案生成助手。请根据用户需求生成自然、真实的中文文本。"},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(max_retries):
            try:
                response = self.classifier.client.chat(messages, temperature=0.8, max_tokens=100)
                
                if 'choices' in response and len(response['choices']) > 0:
                    text = response['choices'][0]['message']['content'].strip()
                    # 验证生成的文本
                    validation = self.classifier.analyze_text(text, use_ai=True)
                    if validation['category'] == category:
                        return {
                            'text': text,
                            'category': category,
                            'category_name': self.category_names[category],
                            'confidence': validation['confidence'],
                            'reasoning': f"AI生成并验证为{self.category_names[category]}",
                            'keyword_weights': validation['keyword_weights'],
                            'source': 'ai_generated'
                        }
                    else:
                        return None  # 分类不匹配，返回None
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    print(f"    AI生成失败: {e}")
        
        return None
    
    def generate_by_ai(self, count: int, category: str) -> List[Dict]:
        """使用AI并发生成指定类别的文本"""
        if not self.use_ai_for_generation:
            print(f"未配置DeepSeek API，无法使用AI生成{self.category_names[category]}")
            return []
        
        print(f"\n使用AI并发生成 {count} 条{self.category_names[category]}...")
        
        prompts = {
            'normal': "请生成一段正常的中文文本，内容是日常生活中的感想、心情或分享，不包含任何商业推广内容。",
            'ad': "请生成一段中文广告文本，内容包含产品名称、价格、促销信息和购买引导。",
            'hidden_ad': "请生成一段中文隐性广告文本，以个人分享、推荐或使用体验的形式呈现，但实际有推广目的。"
        }
        
        prompt = prompts[category]
        results = []
        batch_size = 5  # 每批并发请求数
        max_workers = 5  # 最大并发数
        total_batches = (count + batch_size - 1) // batch_size
        
        with tqdm(total=count, desc=f"AI生成{self.category_names[category]}", unit="条") as pbar:
            for batch_idx in range(total_batches):
                needed = min(batch_size, count - len(results))
                if needed <= 0:
                    break
                    
                batch_results = []
                
                # 使用线程池并发处理
                with ThreadPoolExecutor(max_workers=min(max_workers, needed)) as executor:
                    futures = []
                    for _ in range(needed):
                        future = executor.submit(self._generate_single_ai_text, category, prompt)
                        futures.append(future)
                    
                    # 收集结果
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            batch_results.append(result)
                        pbar.update(1)
                
                results.extend(batch_results)
                
                if batch_idx + 1 < total_batches:
                    time.sleep(0.5)
        
        print(f"  成功生成 {len(results)} 条有效文本")
        return results
    
    def build_balanced_dataset(self, total_size: int = 10000) -> List[Dict]:
        """构建平衡数据集"""
        print("=" * 70)
        print("平衡数据集构建开始")
        print("=" * 70)
        print(f"目标分布: 正常文本:隐性广告:广告 = 5:4:1")
        print(f"总样本数: {total_size}")
        print("=" * 70)
        
        # 计算各类别目标数量
        target_counts = {
            'normal': int(total_size * self.target_ratio['normal']),
            'hidden_ad': int(total_size * self.target_ratio['hidden_ad']),
            'ad': int(total_size * self.target_ratio['ad'])
        }
        
        # 调整确保总数为10000
        target_counts['normal'] += total_size - sum(target_counts.values())
        
        print(f"\n目标数量:")
        for cat, count in target_counts.items():
            print(f"  {self.category_names[cat]}: {count} 条 ({self.target_ratio[cat] * 100:.0f}%)")
        
        # 加载现有数据
        existing_data = self.load_existing_classified_data()
        
        # 按类别分组现有数据
        available_by_category = {
            'normal': [],
            'ad': [],
            'hidden_ad': []
        }
        
        for item in existing_data:
            cat = item['category']
            if cat in available_by_category:
                available_by_category[cat].append(item)
        
        print(f"\n现有数据按类别分布:")
        for cat, items in available_by_category.items():
            print(f"  {self.category_names[cat]}: {len(items)} 条")
        
        # 收集各类别的数据
        collected = {
            'normal': [],
            'ad': [],
            'hidden_ad': []
        }
        
        # 先从现有数据中选取
        for cat in ['normal', 'ad', 'hidden_ad']:
            take_count = min(target_counts[cat], len(available_by_category[cat]))
            collected[cat] = available_by_category[cat][:take_count]
            print(f"\n从现有数据中选取 {self.category_names[cat]}: {len(collected[cat])} 条")
        
        # 补充生成的数据
        for cat in ['normal', 'ad', 'hidden_ad']:
            needed = target_counts[cat] - len(collected[cat])
            if needed <= 0:
                continue
            
            print(f"\n需要补充 {self.category_names[cat]}: {needed} 条")
            
            generated = []
            
            # 优先使用AI生成（如果配置了API）
            if self.use_ai_for_generation:
                print(f"  优先使用AI生成...")
                ai_generated = self.generate_by_ai(needed, cat)
                generated.extend(ai_generated)
            
            # AI生成不足时，使用规则方法补充
            still_needed = needed - len(generated)
            if still_needed > 0:
                print(f"  AI生成不足 ({len(generated)}条)，使用规则方法补充 {still_needed} 条...")
                
                attempts = 0
                max_attempts = still_needed * 2  # 最多尝试2倍数量
                
                # 第一阶段：尝试生成并验证
                while len(generated) < needed and attempts < max_attempts:
                    attempts += 1
                    
                    # 生成文本
                    if cat == 'normal':
                        text = self.generate_normal_texts(1)[0]
                    elif cat == 'ad':
                        text = self.generate_ad_texts(1)[0]
                    else:
                        text = self.generate_hidden_ad_texts(1)[0]
                    
                    # 验证分类
                    result = self.classifier.analyze_text(text, use_ai=False)
                    if result['category'] == cat:
                        generated.append({
                            'text': text,
                            'category': cat,
                            'category_name': self.category_names[cat],
                            'confidence': result['confidence'],
                            'reasoning': result.get('reasoning', ''),
                            'keyword_weights': result.get('keyword_weights', {}),
                            'source': 'rule_generated_validated'
                        })
                
                # 第二阶段：如果验证生成不够，直接生成不验证（确保达到目标数量）
                still_needed_after = needed - len(generated)
                if still_needed_after > 0:
                    print(f"  验证生成不足，直接生成 {still_needed_after} 条...")
                    
                    if cat == 'normal':
                        texts = self.generate_normal_texts(still_needed_after)
                    elif cat == 'ad':
                        texts = self.generate_ad_texts(still_needed_after)
                    else:
                        texts = self.generate_hidden_ad_texts(still_needed_after)
                    
                    for text in texts:
                        generated.append({
                            'text': text,
                            'category': cat,
                            'category_name': self.category_names[cat],
                            'confidence': 0.85,
                            'reasoning': f'生成的{self.category_names[cat]}',
                            'keyword_weights': {},
                            'source': 'rule_generated_direct'
                        })
            
            collected[cat].extend(generated)
            print(f"  成功补充: {len(generated)} 条")
        
        # 合并并打乱
        balanced_data = []
        for cat in ['normal', 'ad', 'hidden_ad']:
            balanced_data.extend(collected[cat][:target_counts[cat]])
        
        random.shuffle(balanced_data)
        
        print("\n" + "=" * 70)
        print("数据集构建完成")
        print("=" * 70)
        
        # 统计最终分布
        final_counts = {'normal': 0, 'ad': 0, 'hidden_ad': 0}
        for item in balanced_data:
            final_counts[item['category']] += 1
        
        print("\n最终数据分布:")
        for cat, count in final_counts.items():
            percentage = count / len(balanced_data) * 100
            print(f"  {self.category_names[cat]}: {count} 条 ({percentage:.1f}%)")
        
        return balanced_data
    
    def save_dataset(self, data: List[Dict], filename: str = 'balanced_dataset.json'):
        """保存数据集到文件"""
        output_path = self.output_dir / filename
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据集已保存到: {output_path}")
        return output_path


def main():
    builder = BalancedDatasetBuilder()
    
    # 构建平衡数据集
    balanced_data = builder.build_balanced_dataset(total_size=10000)
    
    # 保存数据集
    builder.save_dataset(balanced_data, 'balanced_dataset.json')
    
    # 也保存为ERNIE训练格式
    from process_classified_data import TextDataProcessor
    processor = TextDataProcessor()
    processor.prepare_for_ernie_training(balanced_data, str(builder.output_dir / 'ernie_balanced_data.json'))
    
    print("\n" + "=" * 70)
    print("所有操作完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
