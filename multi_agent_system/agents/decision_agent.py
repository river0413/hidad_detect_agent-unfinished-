"""决策智能体 - 协调其他智能体并更新它们的提示词"""
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentResponse
from ..config.agent_config import DecisionAgentConfig


class DecisionAgent(BaseAgent):
    """负责决策制定和协调其他智能体的智能体"""

    def __init__(self, config: DecisionAgentConfig):
        super().__init__(config)
        self.updateable_agents = config.updateable_agents
        self._agent_registry = None
        self._config_manager = None
        self._decision_history: List[Dict[str, Any]] = []

    def set_registry(self, registry):
        """设置智能体注册表"""
        self._agent_registry = registry

    def set_config_manager(self, config_manager):
        """设置配置管理器"""
        self._config_manager = config_manager

    def process(self, input_data: Any) -> AgentResponse:
        """处理决策请求"""
        try:
            if isinstance(input_data, dict):
                action = input_data.get('action', 'decide')
            else:
                action = 'decide'

            if action == 'decide':
                return self._make_decision(input_data)
            elif action == 'update_prompt':
                return self._update_agent_prompt(input_data)
            elif action == 'get_recommendation':
                return self._get_agent_recommendation(input_data)
            elif action == 'coordinate':
                return self._coordinate_agents(input_data)
            elif action == 'analyze_task':
                return self._analyze_task(input_data)
            elif action == 'final_decision':
                return self._make_final_decision(input_data)
            else:
                return AgentResponse(
                    success=False,
                    error=f"未知操作: {action}",
                    agent_name=self.name
                )

        except Exception as e:
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name
            )

    def _make_decision(self, task: Dict[str, Any]) -> AgentResponse:
        """做出如何处理任务的决策"""
        task_description = task.get('description', str(task))
        task_type = self._infer_task_type(task_description)

        agents_to_use = self._determine_agents_for_task(task_type)

        decision = {
            'task_type': task_type,
            'recommended_agents': agents_to_use,
            'reasoning': self._generate_reasoning(task_type, agents_to_use),
            'timestamp': datetime.now().isoformat()
        }

        self._decision_history.append(decision)

        return AgentResponse(
            success=True,
            data=decision,
            agent_name=self.name,
            metadata={'task_type': task_type}
        )

    def _infer_task_type(self, task_description: str) -> str:
        """从描述中推断任务类型"""
        task_lower = task_description.lower()

        if any(word in task_lower for word in ['分类', '分析', '文本', '检测', '判断', '广告']):
            return 'text_classification'
        elif any(word in task_lower for word in ['查询', '搜索', '知识', '历史', '统计']):
            return 'knowledge_query'
        elif any(word in task_lower for word in ['解释', '说明', '输出', '展示', '报告']):
            return 'explanation'
        elif any(word in task_lower for word in ['更新', '修改', '改变', '调整']):
            return 'prompt_update'
        else:
            return 'general'

    def _determine_agents_for_task(self, task_type: str) -> List[str]:
        """确定针对任务类型要使用的智能体"""
        task_agent_map = {
            'text_classification': ['text_processing', 'knowledge'],
            'knowledge_query': ['knowledge'],
            'explanation': ['explanation', 'knowledge'],
            'prompt_update': ['decision'],
            'general': ['text_processing', 'knowledge', 'explanation']
        }
        return task_agent_map.get(task_type, ['text_processing'])

    def _generate_reasoning(self, task_type: str, agents: List[str]) -> str:
        """生成决策理由"""
        reasoning_templates = {
            'text_classification': f"任务涉及文本分类，建议使用文本处理智能体进行分类，并将结果存储到知识智能体中。需要时由解释智能体生成报告。",
            'knowledge_query': f"任务涉及知识查询，直接使用知识智能体进行数据检索和分析。",
            'explanation': f"任务需要生成解释和说明，使用解释智能体将结构化数据转换为人类可读的内容。",
            'prompt_update': f"任务涉及更新提示词，由决策智能体分析并更新相应智能体的提示词。",
            'general': f"任务类型为通用任务，建议调用多个智能体协作完成。"
        }
        return reasoning_templates.get(task_type, "根据任务特征做出决策。")

    def _update_agent_prompt(self, update_request: Dict[str, Any]) -> AgentResponse:
        """更新智能体的提示词"""
        target_agent = update_request.get('agent')
        new_prompt = update_request.get('new_prompt')
        context = update_request.get('context', '')

        if target_agent not in self.updateable_agents:
            return AgentResponse(
                success=False,
                error=f"智能体 '{target_agent}' 不可更新。可更新的智能体: {self.updateable_agents}",
                agent_name=self.name
            )

        if not new_prompt:
            return AgentResponse(
                success=False,
                error="需要提供 new_prompt",
                agent_name=self.name
            )

        enhanced_prompt = self._enhance_prompt_with_context(new_prompt, context)

        if self._config_manager:
            self._config_manager.update_prompt(target_agent, enhanced_prompt)

        if self._agent_registry:
            agent = self._agent_registry.get_agent(target_agent)
            if agent:
                agent.update_prompt(enhanced_prompt)

        self._decision_history.append({
            'action': 'update_prompt',
            'target_agent': target_agent,
            'timestamp': datetime.now().isoformat(),
            'context': context
        })

        return AgentResponse(
            success=True,
            data={
                'agent': target_agent,
                'prompt_updated': True,
                'enhanced': bool(context)
            },
            agent_name=self.name,
            metadata={'target_agent': target_agent}
        )

    def _enhance_prompt_with_context(self, base_prompt: str, context: str) -> str:
        """用额外的上下文增强提示词"""
        if not context:
            return base_prompt

        enhancement_prompt = f"""原始提示词：
{base_prompt}

附加上下文：
{context}

请结合上述信息，优化原始提示词，使其更适应特定的上下文需求。直接输出优化后的提示词，不要解释。"""

        try:
            messages = [
                {"role": "system", "content": "你是一个提示词优化专家。"},
                {"role": "user", "content": enhancement_prompt}
            ]
            response = self._call_deepseek(messages, temperature=0.7)
            if 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"提示词增强失败: {e}")

        return base_prompt

    def _get_agent_recommendation(self, query: Dict[str, Any]) -> AgentResponse:
        """获取智能体配置的推荐"""
        task = query.get('task', '')
        task_type = self._infer_task_type(task)

        recommendations = {
            'suggested_agents': self._determine_agents_for_task(task_type),
            'suggested_temperature': self._get_temperature_for_task(task_type),
            'task_type': task_type,
            'reasoning': self._generate_reasoning(task_type, self._determine_agents_for_task(task_type))
        }

        return AgentResponse(
            success=True,
            data=recommendations,
            agent_name=self.name
        )

    def _get_temperature_for_task(self, task_type: str) -> float:
        """获取任务类型的推荐温度"""
        temperature_map = {
            'text_classification': 0.3,
            'knowledge_query': 0.5,
            'explanation': 0.7,
            'prompt_update': 0.7,
            'general': 0.7
        }
        return temperature_map.get(task_type, 0.7)

    def _coordinate_agents(self, coordination_request: Dict[str, Any]) -> AgentResponse:
        """协调多个智能体共同工作"""
        task = coordination_request.get('task')
        agent_sequence = coordination_request.get('sequence', [])

        if not agent_sequence:
            decision_response = self._make_decision({'description': str(task)})
            if decision_response.success:
                agent_sequence = decision_response.data.get('recommended_agents', [])

        workflow = {
            'task': task,
            'agent_sequence': agent_sequence,
            'coordination_type': 'sequential',
            'timestamp': datetime.now().isoformat()
        }

        self._decision_history.append(workflow)

        return AgentResponse(
            success=True,
            data=workflow,
            agent_name=self.name,
            metadata={'agent_count': len(agent_sequence)}
        )

    def _analyze_task(self, task: Dict[str, Any]) -> AgentResponse:
        """分析任务并提供详细分析"""
        description = task.get('description', str(task))
        task_type = self._infer_task_type(description)

        analysis = {
            'inferred_type': task_type,
            'complexity': self._assess_complexity(description),
            'suggested_agents': self._determine_agents_for_task(task_type),
            'estimated_steps': self._estimate_steps(task_type),
            'recommendations': self._generate_recommendations(task_type)
        }

        return AgentResponse(
            success=True,
            data=analysis,
            agent_name=self.name,
            metadata={'task_type': task_type}
        )

    def _assess_complexity(self, task_description: str) -> str:
        """评估任务复杂度"""
        length = len(task_description)
        if length > 500:
            return "high"
        elif length > 200:
            return "medium"
        else:
            return "low"

    def _estimate_steps(self, task_type: str) -> int:
        """估计所需步骤数"""
        steps_map = {
            'text_classification': 3,
            'knowledge_query': 2,
            'explanation': 2,
            'prompt_update': 2,
            'general': 4
        }
        return steps_map.get(task_type, 3)

    def _generate_recommendations(self, task_type: str) -> List[str]:
        """生成任务建议"""
        recommendations_map = {
            'text_classification': [
                '建议使用文本处理智能体进行分类',
                '结果应存储到知识智能体以供后续查询',
                '可配置解释智能体生成用户友好的报告'
            ],
            'knowledge_query': [
                '直接查询知识智能体获取历史数据',
                '可结合统计分析了解分类趋势'
            ],
            'explanation': [
                '使用解释智能体将JSON转换为自然语言',
                '建议提供详细的上下文以获得更好的解释'
            ],
            'prompt_update': [
                '谨慎更新提示词，建议提供充分的上下文',
                '更新后建议测试提示词效果'
            ],
            'general': [
                '建议先分析任务类型',
                '根据任务类型选择合适的智能体组合'
            ]
        }
        return recommendations_map.get(task_type, ['根据具体情况做出决策'])

    def get_decision_history(self) -> List[Dict[str, Any]]:
        """获取决策历史"""
        return self._decision_history.copy()

    def _make_final_decision(self, input_data: Dict[str, Any]) -> AgentResponse:
        """综合分类结果及自身判断，决定最终文本类别"""
        text = input_data.get('text', '')
        classification_result = input_data.get('classification_result', {})
        
        classifier_category = classification_result.get('category', 'unknown')
        classifier_confidence = classification_result.get('confidence', 0)
        
        final_category, final_confidence, reasoning = self._analyze_and_decide(
            text, classifier_category, classifier_confidence
        )
        
        decision = {
            'text': text,
            'classifier_category': classifier_category,
            'classifier_confidence': classifier_confidence,
            'final_category': final_category,
            'final_confidence': final_confidence,
            'category_changed': final_category != classifier_category,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        }
        
        self._decision_history.append(decision)
        
        return AgentResponse(
            success=True,
            data=decision,
            agent_name=self.name,
            metadata={'category_changed': final_category != classifier_category}
        )
    
    def _analyze_and_decide(self, text: str, classifier_category: str, classifier_confidence: float) -> tuple:
        """分析文本并做出最终决策"""
        text_lower = text.lower()
        
        keywords_normal = ['今天', '天气', '朋友', '心情', '生活', '日记', '记录', '分享', '感受']
        keywords_ad = ['购买', '下单', '立即', '抢购', '限时', '折扣', '优惠', '特价', '爆款', '热卖', '推荐', '种草', '安利', '必买', '超赞', '价格', '多少钱', '性价比', '划算', '值得', '超值', '便宜', '实惠', '赶紧', '快来', '马上', '立刻', '不要错过', '仅剩', '名额有限', '点击', '扫码', '链接', '私信', '联系', '淘宝', '天猫', '京东', '拼多多', '官网', '小程序', 'app']
        keywords_hidden = ['这款', '真的', '太好用', '真的很明显', '效果', '皮肤', '朋友推荐', '我觉得', '大家都在用', '真心推荐', '绝绝子', 'yyds', '无限回购', '空瓶', '必备']
        
        normal_count = sum(1 for word in keywords_normal if word in text_lower)
        ad_count = sum(1 for word in keywords_ad if word in text_lower)
        hidden_count = sum(1 for word in keywords_hidden if word in text_lower)
        
        final_category = classifier_category
        final_confidence = classifier_confidence
        reasoning = f"使用分类器结果，置信度: {classifier_confidence:.2f}"
        
        if ad_count >= 3:
            if classifier_category != 'ad':
                final_category = 'ad'
                final_confidence = 0.85 + (ad_count * 0.02)
                reasoning = f"检测到{ad_count}个明显广告特征，修正为显性广告"
        elif hidden_count >= 2:
            if classifier_category != 'hidden_ad':
                final_category = 'hidden_ad'
                final_confidence = 0.75 + (hidden_count * 0.03)
                reasoning = f"检测到{hidden_count}个隐性广告特征，修正为隐性广告"
        elif normal_count >= 3 and classifier_confidence < 0.6:
            if classifier_category != 'normal':
                final_category = 'normal'
                final_confidence = 0.70 + (normal_count * 0.02)
                reasoning = f"检测到{normal_count}个正常文本特征，修正为正常文本"
        
        return final_category, final_confidence, reasoning
    
    def clear_history(self):
        """清空决策历史"""
        self._decision_history.clear()
