"""解释智能体 - 将结构化数据转换为人类可读的解释"""
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentResponse
from ..config.agent_config import ExplanationAgentConfig


class ExplanationAgent(BaseAgent):
    """负责以人类可读格式解释结果的智能体"""

    def __init__(self, config: ExplanationAgentConfig):
        super().__init__(config)
        self.output_format = config.output_format
        self._explanation_templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """加载解释模板"""
        return {
            'classification': """【文本分类结果】

📝 原文摘要：{text_summary}

📊 分类类别：{category}
🎯 置信度：{confidence}%
💡 判断理由：{reasoning}

📈 特征分析：
{keyword_analysis}

🕐 分析时间：{timestamp}
🔧 分析来源：{source}""",

            'summary': """【分析摘要报告】

总计分析文本数：{total_count}
识别为正常文本：{normal_count} ({normal_percent}%)
识别为显性广告：{ad_count} ({ad_percent}%)
识别为隐性广告：{hidden_ad_count} ({hidden_ad_percent}%)

📊 整体置信度：{avg_confidence}%

⏱️ 生成时间：{timestamp}""",

            'history': """【历史记录查询结果】

共找到 {count} 条相关记录：

{records}

📋 显示第 {start}-{end} 条记录""",

            'error': """❌ 发生错误

错误类型：{error_type}
错误描述：{error_message}
发生时间：{timestamp}

建议：请检查输入数据或联系管理员。""",

            'recommendation': """【智能建议】

基于当前分析结果，建议如下：

{recommendations}

💡 以上建议仅供参考，实际决策需结合更多因素。"""
        }

    def process(self, input_data: Any) -> AgentResponse:
        """处理解释请求"""
        try:
            if isinstance(input_data, dict):
                format_type = input_data.get('format', 'classification')
                data = input_data.get('data', input_data)
            else:
                format_type = 'classification'
                data = input_data

            if format_type == 'classification':
                return self._explain_classification(data)
            elif format_type == 'summary':
                return self._explain_summary(data)
            elif format_type == 'history':
                return self._explain_history(data)
            elif format_type == 'ai':
                return self._explain_with_ai(data)
            elif format_type == 'recommendation':
                return self._explain_recommendation(data)
            else:
                return self._explain_general(data)

        except Exception as e:
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name
            )

    def _explain_classification(self, data: Dict[str, Any]) -> AgentResponse:
        """解释分类结果"""
        text = data.get('text', '')
        text_summary = text[:100] + '...' if len(text) > 100 else text

        category = data.get('category', 'unknown')
        category_display = {
            'normal': '✅ 正常文本',
            'ad': '🚨 显性广告',
            'hidden_ad': '⚠️ 隐性广告',
            'unknown': '❓ 未知类别'
        }.get(category, category)

        confidence = data.get('confidence', 0) * 100
        reasoning = data.get('reasoning', '未提供')
        source = data.get('source', 'unknown')
        timestamp = data.get('timestamp', datetime.now().isoformat())

        keyword_weights = data.get('keyword_weights', {})
        keyword_analysis = self._format_keyword_analysis(keyword_weights)

        explanation = self._explanation_templates['classification'].format(
            text_summary=text_summary,
            category=category_display,
            confidence=round(confidence, 2),
            reasoning=reasoning,
            keyword_analysis=keyword_analysis,
            timestamp=timestamp,
            source=source
        )

        return AgentResponse(
            success=True,
            data={
                'explanation': explanation,
                'category': category,
                'confidence': confidence,
                'format': 'human_readable'
            },
            agent_name=self.name,
            metadata={'format': 'classification'}
        )

    def _format_keyword_analysis(self, keyword_weights: Dict[str, float]) -> str:
        """将关键词权重格式化为可读分析"""
        if not keyword_weights:
            return "未检测到明显特征"

        analysis_items = []
        for key, value in keyword_weights.items():
            key_display = {
                'promotion_words': '推广词汇',
                'price_mentions': '价格提及',
                'urgency_expressions': '紧迫性表达',
                'brand_mentions': '品牌提及',
                'action_words': '行动词汇',
                'natural_expression': '自然表达'
            }.get(key, key)

            bar_length = int(value * 10)
            bar = '█' * bar_length + '░' * (10 - bar_length)
            analysis_items.append(f"  {key_display}: {bar} ({value:.2f})")

        return '\n'.join(analysis_items)

    def _explain_summary(self, data: Dict[str, Any]) -> AgentResponse:
        """解释摘要统计"""
        total_count = data.get('total_analyses', 0)
        category_stats = data.get('category_stats', [])

        normal_count = 0
        ad_count = 0
        hidden_ad_count = 0

        for stat in category_stats:
            cat = stat.get('category', '')
            count = stat.get('count', 0)
            if cat == 'normal':
                normal_count = count
            elif cat == 'ad':
                ad_count = count
            elif cat == 'hidden_ad':
                hidden_ad_count = count

        avg_confidence = data.get('average_confidence', 0) * 100
        timestamp = datetime.now().isoformat()

        normal_percent = (normal_count / total_count * 100) if total_count > 0 else 0
        ad_percent = (ad_count / total_count * 100) if total_count > 0 else 0
        hidden_ad_percent = (hidden_ad_count / total_count * 100) if total_count > 0 else 0

        explanation = self._explanation_templates['summary'].format(
            total_count=total_count,
            normal_count=normal_count,
            normal_percent=round(normal_percent, 2),
            ad_count=ad_count,
            ad_percent=round(ad_percent, 2),
            hidden_ad_count=hidden_ad_count,
            hidden_ad_percent=round(hidden_ad_percent, 2),
            avg_confidence=round(avg_confidence, 2),
            timestamp=timestamp
        )

        return AgentResponse(
            success=True,
            data={
                'explanation': explanation,
                'summary': {
                    'total': total_count,
                    'normal': normal_count,
                    'ad': ad_count,
                    'hidden_ad': hidden_ad_count
                },
                'format': 'human_readable'
            },
            agent_name=self.name,
            metadata={'format': 'summary'}
        )

    def _explain_history(self, data: Dict[str, Any]) -> AgentResponse:
        """解释历史记录"""
        history = data.get('history', [])
        count = data.get('count', len(history))

        if not history:
            return AgentResponse(
                success=True,
                data={'explanation': '未找到历史记录。', 'count': 0},
                agent_name=self.name
            )

        records = []
        for i, record in enumerate(history[:10], 1):
            text = record.get('text', '')[:50]
            category = record.get('category', 'unknown')
            confidence = record.get('confidence', 0) * 100
            timestamp = record.get('timestamp', '')[:19]

            category_icon = {'normal': '✅', 'ad': '🚨', 'hidden_ad': '⚠️'}.get(category, '❓')

            records.append(f"{i}. {category_icon} {text}... (置信度: {confidence:.1f}%, 时间: {timestamp})")

        explanation = self._explanation_templates['history'].format(
            count=count,
            records='\n'.join(records),
            start=1,
            end=min(10, count)
        )

        return AgentResponse(
            success=True,
            data={
                'explanation': explanation,
                'count': count,
                'format': 'human_readable'
            },
            agent_name=self.name,
            metadata={'format': 'history', 'displayed': len(records)}
        )

    def _explain_recommendation(self, data: Dict[str, Any]) -> AgentResponse:
        """基于分析生成建议"""
        recommendations = data.get('recommendations', [])

        if not recommendations:
            recommendations = [
                "建议持续监控文本内容，及时发现潜在广告",
                "可结合多个维度的特征进行综合判断",
                "置信度较低时建议人工复核"
            ]

        rec_text = '\n'.join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])

        explanation = self._explanation_templates['recommendation'].format(
            recommendations=rec_text
        )

        return AgentResponse(
            success=True,
            data={
                'explanation': explanation,
                'recommendations': recommendations,
                'format': 'human_readable'
            },
            agent_name=self.name,
            metadata={'format': 'recommendation'}
        )

    def _explain_with_ai(self, data: Dict[str, Any]) -> AgentResponse:
        """使用AI生成自然语言解释"""
        prompt = f"""请将以下分析结果转换为通俗易懂的解释：

{json.dumps(data, ensure_ascii=False, indent=2)}

请用简洁明了的语言解释这些结果，让普通用户也能理解。
直接输出解释内容，不要使用markdown格式。"""

        try:
            messages = [
                {"role": "system", "content": "你是一个专业的文本分析解释专家。"},
                {"role": "user", "content": prompt}
            ]
            response = self._call_deepseek(messages, temperature=0.7)

            if 'choices' in response and len(response['choices']) > 0:
                explanation = response['choices'][0]['message']['content'].strip()

                return AgentResponse(
                    success=True,
                    data={
                        'explanation': explanation,
                        'format': 'ai_generated'
                    },
                    agent_name=self.name,
                    metadata={'format': 'ai'}
                )

        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"AI解释生成失败: {str(e)}",
                agent_name=self.name
            )

    def _explain_general(self, data: Any) -> AgentResponse:
        """对任何数据的通用解释"""
        if isinstance(data, dict):
            explanation_parts = []
            for key, value in data.items():
                key_display = {
                    'text': '文本内容',
                    'category': '分类结果',
                    'confidence': '置信度',
                    'reasoning': '判断理由',
                    'keyword_weights': '关键词权重',
                    'source': '数据来源',
                    'timestamp': '时间戳'
                }.get(key, key)

                if key == 'confidence' and isinstance(value, float):
                    explanation_parts.append(f"{key_display}: {value * 100:.2f}%")
                elif isinstance(value, dict):
                    explanation_parts.append(f"{key_display}: [详细信息]")
                else:
                    explanation_parts.append(f"{key_display}: {value}")

            explanation = '\n'.join([f"• {part}" for part in explanation_parts])
        else:
            explanation = str(data)

        return AgentResponse(
            success=True,
            data={
                'explanation': explanation,
                'format': 'general'
            },
            agent_name=self.name,
            metadata={'format': 'general'}
        )

    def format_batch_explanation(self, results: List[Dict[str, Any]]) -> AgentResponse:
        """格式化批量结果的解释"""
        if not results:
            return AgentResponse(
                success=True,
                data={'explanation': '没有结果需要解释。'},
                agent_name=self.name
            )

        summary = {
            'total': len(results),
            'categories': {}
        }

        for result in results:
            cat = result.get('category', 'unknown')
            summary['categories'][cat] = summary['categories'].get(cat, 0) + 1

        explanation_parts = [f"【批量分析结果】\n\n总计: {summary['total']} 条文本\n\n分类统计："]

        for cat, count in summary['categories'].items():
            percent = count / summary['total'] * 100
            cat_name = {'normal': '正常', 'ad': '显性广告', 'hidden_ad': '隐性广告'}.get(cat, cat)
            explanation_parts.append(f"• {cat_name}: {count} 条 ({percent:.1f}%)")

        return AgentResponse(
            success=True,
            data={
                'explanation': '\n'.join(explanation_parts),
                'summary': summary,
                'format': 'batch_summary'
            },
            agent_name=self.name,
            metadata={'result_count': len(results)}
        )


import json
