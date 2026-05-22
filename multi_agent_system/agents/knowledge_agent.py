"""使用SQLite存储的知识智能体"""
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from .base_agent import BaseAgent, AgentResponse
from ..config.agent_config import KnowledgeAgentConfig


class KnowledgeAgent(BaseAgent):
    """管理知识存储的智能体，使用SQLite存储"""

    def __init__(self, config: KnowledgeAgentConfig):
        super().__init__(config)
        self.sqlite_path = config.sqlite_path
        self._sqlite_conn = None

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        """获取SQLite连接"""
        if self._sqlite_conn is None:
            db_path = Path(self.sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._sqlite_conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._init_sqlite_schema()
        return self._sqlite_conn

    def _init_sqlite_schema(self):
        """初始化SQLite数据库表结构"""
        cursor = self._sqlite_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS text_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT NOT NULL,
                text_content TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL,
                reasoning TEXT,
                keyword_weights TEXT,
                source TEXT,
                agent_name TEXT,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS category_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                count INTEGER DEFAULT 0,
                avg_confidence REAL,
                last_updated TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                result TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_text_hash ON text_analysis(text_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON text_analysis(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON text_analysis(timestamp)")
        self._sqlite_conn.commit()

    def process(self, input_data: Any) -> AgentResponse:
        """处理知识存储和查询请求"""
        try:
            if isinstance(input_data, dict):
                action = input_data.get('action', 'store')
                data = input_data.get('data')
            else:
                action = 'store'
                data = input_data

            if action == 'store':
                return self._store_analysis(data)
            elif action == 'query':
                return self._query_knowledge(data)
            elif action == 'stats':
                return self._get_statistics()
            elif action == 'history':
                return self._get_history(data)
            elif action == 'update_graph':
                return self._update_graph_relationships(data)
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

    def _store_analysis(self, data: Dict[str, Any]) -> AgentResponse:
        """存储文本分析结果"""
        import hashlib

        text = data.get('text', '')
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        category = data.get('category', 'unknown')
        confidence = data.get('confidence', 0.0)
        reasoning = data.get('reasoning', '')
        keyword_weights = json.dumps(data.get('keyword_weights', {}))
        source = data.get('source', 'unknown')
        agent_name = data.get('agent', self.name)
        timestamp = datetime.now().isoformat()
        metadata = json.dumps(data.get('metadata', {}))

        conn = self._get_sqlite_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO text_analysis
            (text_hash, text_content, category, confidence, reasoning, keyword_weights, source, agent_name, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (text_hash, text, category, confidence, reasoning, keyword_weights, source, agent_name, timestamp, metadata))

        cursor.execute("""
            INSERT INTO analysis_history (session_id, text_hash, result, timestamp)
            VALUES (?, ?, ?, ?)
        """, (agent_name, text_hash, json.dumps(data), timestamp))

        self._update_category_stats(category, confidence)

        conn.commit()

        return AgentResponse(
            success=True,
            data={
                'text_hash': text_hash,
                'category': category,
                'stored': True,
                'storage': 'sqlite'
            },
            agent_name=self.name,
            metadata={'text_hash': text_hash}
        )

    def _update_category_stats(self, category: str, confidence: float):
        """更新分类统计信息"""
        cursor = self._sqlite_conn.cursor()
        cursor.execute("SELECT count, avg_confidence FROM category_stats WHERE category = ?", (category,))
        row = cursor.fetchone()

        if row:
            new_count = row[0] + 1
            new_avg = (row[1] * row[0] + confidence) / new_count
            cursor.execute("""
                UPDATE category_stats
                SET count = ?, avg_confidence = ?, last_updated = ?
                WHERE category = ?
            """, (new_count, new_avg, datetime.now().isoformat(), category))
        else:
            cursor.execute("""
                INSERT INTO category_stats (category, count, avg_confidence, last_updated)
                VALUES (?, 1, ?, ?)
            """, (category, confidence, datetime.now().isoformat()))

    def _query_knowledge(self, query: Dict[str, Any]) -> AgentResponse:
        """查询已存储的知识"""
        query_type = query.get('type', 'category')
        query_value = query.get('value', '')

        conn = self._get_sqlite_conn()
        cursor = conn.cursor()

        if query_type == 'category':
            cursor.execute("""
                SELECT text_content, category, confidence, reasoning, timestamp
                FROM text_analysis
                WHERE category = ?
                ORDER BY timestamp DESC
                LIMIT 100
            """, (query_value,))
        elif query_type == 'hash':
            cursor.execute("""
                SELECT text_content, category, confidence, reasoning, timestamp, keyword_weights
                FROM text_analysis
                WHERE text_hash = ?
            """, (query_value,))
        elif query_type == 'text_search':
            cursor.execute("""
                SELECT text_content, category, confidence, reasoning, timestamp
                FROM text_analysis
                WHERE text_content LIKE ?
                ORDER BY timestamp DESC
                LIMIT 50
            """, (f'%{query_value}%',))
        elif query_type == 'recent':
            limit = query.get('limit', 50)
            cursor.execute("""
                SELECT text_content, category, confidence, reasoning, timestamp
                FROM text_analysis
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        else:
            return AgentResponse(
                success=False,
                error=f"未知查询类型: {query_type}",
                agent_name=self.name
            )

        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                'text': row[0],
                'category': row[1],
                'confidence': row[2],
                'reasoning': row[3],
                'timestamp': row[4]
            })

        return AgentResponse(
            success=True,
            data={'results': results, 'count': len(results), 'query_type': query_type},
            agent_name=self.name
        )

    def _get_statistics(self) -> AgentResponse:
        """获取分析统计信息"""
        conn = self._get_sqlite_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT category, count, avg_confidence FROM category_stats ORDER BY count DESC")
        stats = [{'category': row[0], 'count': row[1], 'avg_confidence': row[2]} for row in cursor.fetchall()]

        cursor.execute("SELECT COUNT(*) as total, AVG(confidence) as avg FROM text_analysis")
        total_row = cursor.fetchall()[0]

        cursor.execute("SELECT COUNT(DISTINCT text_hash) as unique_texts FROM text_analysis")
        unique_row = cursor.fetchall()[0]

        return AgentResponse(
            success=True,
            data={
                'total_analyses': total_row[0],
                'unique_texts': unique_row[0],
                'average_confidence': total_row[1],
                'category_stats': stats
            },
            agent_name=self.name
        )

    def _get_history(self, params: Dict[str, Any]) -> AgentResponse:
        """获取分析历史"""
        session_id = params.get('session_id', '')
        limit = params.get('limit', 100)

        conn = self._get_sqlite_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ah.session_id, ah.text_hash, ah.result, ah.timestamp, ta.text_content
            FROM analysis_history ah
            LEFT JOIN text_analysis ta ON ah.text_hash = ta.text_hash
            WHERE ah.session_id = ?
            ORDER BY ah.timestamp DESC
            LIMIT ?
        """, (session_id, limit))

        rows = cursor.fetchall()
        history = []
        for row in rows:
            history.append({
                'session_id': row[0],
                'text_hash': row[1],
                'result': json.loads(row[2]),
                'timestamp': row[3],
                'text': row[4]
            })

        return AgentResponse(
            success=True,
            data={'history': history, 'count': len(history)},
            agent_name=self.name
        )

    def close(self):
        """关闭数据库连接"""
        if self._sqlite_conn:
            self._sqlite_conn.close()
        super().close()
