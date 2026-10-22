"""集中注册 Agent 可以调用的工具。"""
import json
from tools.evidence_search import DEFAULT_TOP_K, evidence_tool_schema, search_evidence

# 工具列表
TOOL_SCHEMAS = [evidence_tool_schema]

def execute_tool(tool_name, arguments):
    """执行工具的统一入口"""
    if tool_name == "search_evidence":
        return search_evidence(
            query=arguments["query"],
            top_k=DEFAULT_TOP_K,
            source=arguments.get("source"),
        )

    return json.dumps(
        {"error": f"未知工具：{tool_name}"},
        ensure_ascii=False,
    )
