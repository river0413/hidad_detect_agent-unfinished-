"""多智能体系统主入口"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from multi_agent_system.orchestrator import LangGraphOrchestrator
from multi_agent_system.agents import AgentResponse


class MultiAgentSystem:
    """多智能体系统主接口"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = Path(__file__).parent / "config"
        self.orchestrator = LangGraphOrchestrator(str(config_dir))

    def analyze_text(self, text: str, use_explanation: bool = True) -> dict:
        """分析文本并返回分类结果（可选包含解释）"""
        result = self.orchestrator.run(text, text_input=text)

        if use_explanation and result.get('success'):
            explanation = result['result'].get('explanation', {})
            if isinstance(explanation, dict):
                result['explanation_text'] = explanation.get('explanation', '')

        return result

    def query_knowledge(self, query: dict) -> AgentResponse:
        """查询知识库"""
        knowledge_agent = self.orchestrator.get_agent('knowledge')
        if knowledge_agent:
            return knowledge_agent.process({'action': 'query', 'query': query})
        return AgentResponse(
            success=False,
            error='Knowledge agent not available',
            agent_name='system'
        )

    def get_statistics(self) -> dict:
        """获取分析统计信息"""
        return self.orchestrator.get_statistics()

    def update_agent_prompt(self, agent_type: str, new_prompt: str, context: str = '') -> AgentResponse:
        """更新智能体的提示词"""
        return self.orchestrator.update_agent_prompt(agent_type, new_prompt, context)

    def list_available_agents(self) -> list:
        """列出所有可用智能体"""
        return self.orchestrator.list_agents()

    def add_custom_agent(self, agent_type: str, agent_instance):
        """向系统添加自定义智能体"""
        self.orchestrator.add_agent(agent_type, agent_instance)

    def close(self):
        """关闭系统并清理"""
        self.orchestrator.close()


def demo():
    """演示函数，展示系统功能"""
    import sys
    print("=" * 70, flush=True)
    print("多智能体系统演示", flush=True)
    print("=" * 70, flush=True)

    try:
        system = MultiAgentSystem()

        print("\n[1] 初始化成功 - 可用智能体:", flush=True)
        for agent in system.list_available_agents():
            print(f"  - {agent}", flush=True)

        print("\n[2] 测试文本分类:", flush=True)
        test_texts = [
            "非常快,超级方便,辛苦啦"
        ]

        for i, text in enumerate(test_texts, 1):
            print(f"\n  测试文本 {i}: {text[:50]}...", flush=True)
            try:
                result = system.analyze_text(text)
            except Exception as e:
                print(f"  analyze_text异常: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue

            if result.get('success'):
                final_decision = result['result'].get('final_decision_result', {})
                classification = result['result'].get('classification_result', {})
                
                if final_decision:
                    print(f"  原始分类结果: {classification.get('category', 'unknown')}", flush=True)
                    print(f"  原始置信度: {classification.get('confidence', 0) * 100:.2f}%", flush=True)
                    print(f"  最终分类结果: {final_decision.get('final_category', 'unknown')}", flush=True)
                    print(f"  最终置信度: {final_decision.get('final_confidence', 0) * 100:.2f}%", flush=True)
                    
                    if final_decision.get('category_changed'):
                        print(f"  类别已变更: {final_decision.get('reasoning', '')}", flush=True)
                else:
                    print(f"  分类结果: {classification.get('category', 'unknown')}", flush=True)
                    print(f"  置信度: {classification.get('confidence', 0) * 100:.2f}%", flush=True)

                if 'explanation_text' in result:
                    explanation = result['explanation_text']
                    try:
                        explanation = explanation.encode('gbk', errors='replace').decode('gbk')
                        print(f"\n  解释:\n{explanation}", flush=True)
                    except UnicodeEncodeError:
                        explanation = explanation.encode('utf-8', errors='ignore').decode('utf-8')
                        print(f"\n  解释:\n{explanation}", flush=True)
            else:
                print(f"  错误: {result.get('error')}", flush=True)

        print("\n\n[3] 查询知识库统计:", flush=True)
        stats = system.get_statistics()
        if stats:
            print(f"  总分析数: {stats.get('total_analyses', 0)}", flush=True)
            avg_conf = stats.get('average_confidence') or 0
            print(f"  平均置信度: {avg_conf * 100:.2f}%", flush=True)
        else:
            print("  暂无统计数据", flush=True)

        print("\n\n[4] 更新智能体提示词演示:", flush=True)
        update_result = system.update_agent_prompt(
            'text_processing',
            '你是一个严格的文本审核员，需要仔细检查每个文本中的潜在广告特征。',
            '需要更严格地检测隐性广告'
        )
        print(f"  更新结果: {'成功' if update_result.success else '失败'}", flush=True)

        print("\n" + "=" * 70, flush=True)
        print("演示完成!", flush=True)
        print("=" * 70, flush=True)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            system.close()
        except:
            pass


if __name__ == "__main__":
    demo()
