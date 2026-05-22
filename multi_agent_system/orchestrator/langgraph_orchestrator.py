"""基于LangGraph的多智能体系统编排器"""
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from datetime import datetime
import json

from ..agents import (
    AgentRegistry,
    TextProcessingAgent,
    KnowledgeAgent,
    DecisionAgent,
    ExplanationAgent,
    AgentResponse
)
from ..config.agent_config import ConfigManager


class AgentState(TypedDict):
    """LangGraph编排器管理的状态"""
    task: str
    task_type: str
    current_step: str
    text_input: Optional[str]
    classification_result: Optional[Dict[str, Any]]
    final_decision_result: Optional[Dict[str, Any]]
    knowledge_result: Optional[Dict[str, Any]]
    explanation_result: Optional[Dict[str, Any]]
    decision_result: Optional[Dict[str, Any]]
    error: Optional[str]
    agent_outputs: Dict[str, Any]
    workflow_history: Annotated[List[Dict[str, Any]], add]
    metadata: Dict[str, Any]


class LangGraphOrchestrator:
    """使用LangGraph进行多智能体工作流管理的编排器"""

    def __init__(self, config_dir: str = "config"):
        self.registry = AgentRegistry()
        self.config_manager = ConfigManager(config_dir)
        self.config_manager.load_configs()
        self._initialize_agents()
        self.graph = self._build_graph()

    def _initialize_agents(self):
        """初始化所有智能体并注册"""
        configs = self.config_manager.configs

        text_config = configs.get('text_processing')
        if text_config:
            text_agent = TextProcessingAgent(text_config)
            self.registry.register_instance('text_processing', text_agent)

        knowledge_config = configs.get('knowledge')
        if knowledge_config:
            knowledge_agent = KnowledgeAgent(knowledge_config)
            self.registry.register_instance('knowledge', knowledge_agent)

        decision_config = configs.get('decision')
        if decision_config:
            decision_agent = DecisionAgent(decision_config)
            decision_agent.set_registry(self.registry)
            decision_agent.set_config_manager(self.config_manager)
            self.registry.register_instance('decision', decision_agent)

        explanation_config = configs.get('explanation')
        if explanation_config:
            explanation_agent = ExplanationAgent(explanation_config)
            self.registry.register_instance('explanation', explanation_agent)

    def _build_graph(self) -> StateGraph:
        """构建LangGraph状态图"""
        workflow = StateGraph(AgentState)

        workflow.add_node("analyze_task", self._analyze_task_node)
        workflow.add_node("classify_text", self._classify_text_node)
        workflow.add_node("final_decision", self._final_decision_node)
        workflow.add_node("store_knowledge", self._store_knowledge_node)
        workflow.add_node("generate_explanation", self._generate_explanation_node)
        workflow.add_node("handle_error", self._handle_error_node)

        workflow.set_entry_point("analyze_task")

        workflow.add_edge("analyze_task", "classify_text")
        workflow.add_edge("classify_text", "final_decision")
        workflow.add_edge("final_decision", "store_knowledge")
        workflow.add_edge("store_knowledge", "generate_explanation")
        workflow.add_edge("generate_explanation", END)

        def should_handle_error(state: AgentState) -> bool:
            return bool(state.get('error'))

        workflow.add_conditional_edges(
            "analyze_task",
            should_handle_error,
            {True: "handle_error", False: "classify_text"}
        )
        workflow.add_conditional_edges(
            "classify_text",
            should_handle_error,
            {True: "handle_error", False: "final_decision"}
        )
        workflow.add_conditional_edges(
            "final_decision",
            should_handle_error,
            {True: "handle_error", False: "store_knowledge"}
        )
        workflow.add_conditional_edges(
            "store_knowledge",
            should_handle_error,
            {True: "handle_error", False: "generate_explanation"}
        )
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    def _analyze_task_node(self, state: AgentState) -> Dict[str, Any]:
        """分析任务并确定工作流"""
        result = {}
        task = state.get('task', '')

        decision_agent = self.registry.get_agent('decision')
        if decision_agent:
            response = decision_agent.process({
                'action': 'analyze_task',
                'task': {'description': task}
            })

            if response.success:
                result['task_type'] = response.data.get('inferred_type', 'general')
                result['decision_result'] = response.data
                result['agent_outputs'] = {**state.get('agent_outputs', {}), 'decision': response.to_dict()}
                result['workflow_history'] = state.get('workflow_history', []) + [{
                    'step': 'analyze_task',
                    'timestamp': datetime.now().isoformat(),
                    'result': 'success'
                }]

        result['current_step'] = 'analyze_task'
        return result

    def _classify_text_node(self, state: AgentState) -> Dict[str, Any]:
        """使用TextProcessingAgent分类文本"""
        result = {}
        text_agent = self.registry.get_agent('text_processing')

        if not text_agent:
            result['error'] = 'TextProcessingAgent not available'
            return result

        text_input = state.get('text_input')
        if not text_input:
            text_input = state.get('task', '')

        response = text_agent.process({'text': text_input, 'use_ai': True})

        if response.success:
            result['classification_result'] = response.data
            result['agent_outputs'] = {**state.get('agent_outputs', {}), 'text_processing': response.to_dict()}
            result['workflow_history'] = state.get('workflow_history', []) + [{
                'step': 'classify_text',
                'timestamp': datetime.now().isoformat(),
                'result': 'success',
                'category': response.data.get('category')
            }]
        else:
            result['error'] = response.error

        result['current_step'] = 'classify_text'
        return result

    def _final_decision_node(self, state: AgentState) -> Dict[str, Any]:
        """使用决策智能体做出最终分类决策"""
        result = {}
        decision_agent = self.registry.get_agent('decision')
        
        if not decision_agent:
            result['error'] = 'DecisionAgent not available'
            result['final_decision_result'] = state.get('classification_result')
            return result
        
        text_input = state.get('text_input', '')
        if not text_input:
            text_input = state.get('task', '')
        
        classification_result = state.get('classification_result', {})
        
        response = decision_agent.process({
            'action': 'final_decision',
            'text': text_input,
            'classification_result': classification_result
        })
        
        if response.success:
            final_data = response.data
            result['final_decision_result'] = final_data
            
            # 创建兼容的分类结果数据结构供解释和存储使用
            compatible_result = {
                'text': text_input,
                'category': final_data.get('final_category'),
                'confidence': final_data.get('final_confidence'),
                'reasoning': final_data.get('reasoning'),
                'source': 'decision_agent'
            }
            
            result['classification_result'] = compatible_result
            result['agent_outputs'] = {**state.get('agent_outputs', {}), 'final_decision': response.to_dict()}
            result['workflow_history'] = state.get('workflow_history', []) + [{
                'step': 'final_decision',
                'timestamp': datetime.now().isoformat(),
                'result': 'success',
                'category_changed': response.data.get('category_changed', False),
                'final_category': response.data.get('final_category')
            }]
        else:
            result['error'] = response.error
            result['final_decision_result'] = classification_result
        
        result['current_step'] = 'final_decision'
        return result

    def _store_knowledge_node(self, state: AgentState) -> Dict[str, Any]:
        """将最终决策结果存储到知识库"""
        result = {}
        knowledge_agent = self.registry.get_agent('knowledge')

        if not knowledge_agent:
            result['knowledge_result'] = {'stored': False, 'reason': 'agent_not_available'}
            return result

        classification = state.get('final_decision_result')
        if not classification:
            classification = state.get('classification_result')
        if not classification:
            classification = state.get('task', {})

        if isinstance(classification, dict):
            response = knowledge_agent.process({
                'action': 'store',
                'data': classification
            })

            if response.success:
                result['knowledge_result'] = response.data
                result['agent_outputs'] = {**state.get('agent_outputs', {}), 'knowledge': response.to_dict()}
                result['workflow_history'] = state.get('workflow_history', []) + [{
                    'step': 'store_knowledge',
                    'timestamp': datetime.now().isoformat(),
                    'result': 'success',
                    'text_hash': response.data.get('text_hash')
                }]
            else:
                result['error'] = response.error

        result['current_step'] = 'store_knowledge'
        return result

    def _generate_explanation_node(self, state: AgentState) -> Dict[str, Any]:
        """生成人类可读的解释"""
        result = {}
        explanation_agent = self.registry.get_agent('explanation')

        if not explanation_agent:
            result['explanation_result'] = {'explained': False, 'reason': 'agent_not_available'}
            return result

        # 获取最终决策结果
        final_decision = state.get('final_decision_result', {})
        
        # 构建解释用的数据结构
        explanation_data = {}
        if final_decision:
            explanation_data = {
                'text': final_decision.get('text', state.get('text_input', '')),
                'category': final_decision.get('final_category', 'unknown'),
                'confidence': final_decision.get('final_confidence', 0),
                'reasoning': final_decision.get('reasoning', ''),
                'source': 'decision_agent',
                'timestamp': final_decision.get('timestamp', datetime.now().isoformat())
            }
        else:
            # 如果没有最终决策，使用原始分类结果
            classification = state.get('classification_result', {})
            explanation_data = classification

        response = explanation_agent.process({
            'format': 'classification',
            'data': explanation_data
        })

        if response.success:
            result['explanation_result'] = response.data
            result['agent_outputs'] = {**state.get('agent_outputs', {}), 'explanation': response.to_dict()}
            result['workflow_history'] = state.get('workflow_history', []) + [{
                'step': 'generate_explanation',
                'timestamp': datetime.now().isoformat(),
                'result': 'success'
            }]
        else:
            result['error'] = response.error

        result['current_step'] = 'generate_explanation'
        return result

    def _handle_error_node(self, state: AgentState) -> Dict[str, Any]:
        """处理工作流中的错误"""
        error = state.get('error', 'Unknown error')
        return {
            'workflow_history': state.get('workflow_history', []) + [{
                'step': state.get('current_step', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'result': 'error',
                'error': error
            }]
        }

    def run(self, task: str, text_input: Optional[str] = None) -> Dict[str, Any]:
        """运行编排器工作流"""
        initial_state = AgentState(
            task=task,
            task_type='',
            current_step='',
            text_input=text_input or task,
            classification_result=None,
            final_decision_result=None,
            knowledge_result=None,
            explanation_result=None,
            decision_result=None,
            error=None,
            agent_outputs={},
            workflow_history=[],
            metadata={}
        )

        try:
            result = self.graph.invoke(initial_state)
            return {
                'success': result.get('error') is None,
                'result': {
                    'classification_result': result.get('classification_result'),
                    'final_decision_result': result.get('final_decision_result'),
                    'knowledge': result.get('knowledge_result'),
                    'explanation': result.get('explanation_result'),
                    'decision': result.get('decision_result')
                },
                'workflow': result.get('workflow_history'),
                'error': result.get('error')
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'result': None,
                'workflow': initial_state.get('workflow_history', [])
            }

    def run_simple(self, text: str) -> AgentResponse:
        """简单的单步执行文本分类"""
        text_agent = self.registry.get_agent('text_processing')
        if text_agent:
            return text_agent.process(text)
        return AgentResponse(
            success=False,
            error='TextProcessingAgent not available',
            agent_name='orchestrator'
        )

    def run_with_decision(self, task: str) -> Dict[str, Any]:
        """在执行前进行决策"""
        decision_agent = self.registry.get_agent('decision')
        if not decision_agent:
            return {'success': False, 'error': 'DecisionAgent not available'}

        decision_response = decision_agent.process({
            'action': 'decide',
            'description': task
        })

        if not decision_response.success:
            return {'success': False, 'error': decision_response.error}

        recommended_agents = decision_response.data.get('recommended_agents', [])
        results = {}

        for agent_type in recommended_agents:
            agent = self.registry.get_agent(agent_type)
            if agent:
                response = agent.process(task if agent_type == 'text_processing' else {'data': task})
                results[agent_type] = response.to_dict()

        return {
            'success': True,
            'decision': decision_response.data,
            'results': results
        }

    def add_agent(self, agent_type: str, agent_instance):
        """向注册表添加新智能体"""
        self.registry.register_instance(agent_type, agent_instance)

    def remove_agent(self, agent_type: str):
        """从注册表移除智能体"""
        self.registry.remove_agent(agent_type)

    def list_agents(self) -> List[str]:
        """列出所有可用智能体"""
        return self.registry.list_agents()

    def get_agent(self, agent_type: str):
        """获取特定智能体"""
        return self.registry.get_agent(agent_type)

    def update_agent_prompt(self, agent_type: str, new_prompt: str, context: str = ''):
        """更新智能体的提示词"""
        decision_agent = self.registry.get_agent('decision')
        if decision_agent:
            return decision_agent.process({
                'action': 'update_prompt',
                'agent': agent_type,
                'new_prompt': new_prompt,
                'context': context
            })
        return AgentResponse(
            success=False,
            error='DecisionAgent not available',
            agent_name='orchestrator'
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        knowledge_agent = self.registry.get_agent('knowledge')
        if knowledge_agent:
            response = knowledge_agent.process({'action': 'stats'})
            if response.success:
                return response.data
        return {}

    def close(self):
        """关闭所有智能体并清理"""
        for agent in self.registry.get_all_agents().values():
            agent.close()
