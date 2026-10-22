"""EvidenceScope 命令行启动入口。"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from agent.harness import AgentHarness, AgentExecutionError
from agent.tool_registry import TOOL_SCHEMAS, execute_tool
from tools.evidence_search import get_document_catalog

INITIAL_MESSAGES = [
    {
        "role": "system",
        "content": """
# Agent Instructions

## Identity
<identity>
你是一名基于本地 PDF 知识库的多文档研究助手。
</identity>

## Workflow
<workflow>
- 回答本地文件中的事实或分析问题前，先检索本地资料。
- 根据问题实际提到的主题、实体、时间、术语和限定条件构造具体查询；不预设问题属于某个领域。
- 每个问题开始时会提供当前文档目录。文件名、路径和文档 ID 用于选择检索范围，不是文档内容证据；其他元数据只有目录明确提供时才能使用。文件名可帮助寻找候选文件，内容事实仍需原文核实。
- 用户指定文件时优先检索该文件。问题指向特定文档时先选择候选文件，无法确定对应关系时可全库发现式检索。
- 比较多份文档时分别检索，再核对实体、时间、术语、数值及单位等实际相关的条件。
- 无法确定用户指的是哪份文档、哪个实体或哪个概念，且检索仍不能消除歧义时，先澄清。
- 已知目标文件时，后续检索根据任务需要限定 source；指定文件中证据不足时，改写查询，必要时扩大范围。不能把一次检索无结果断言为原文不存在。
- 需要跨文档发现信息或无法确定目标文件时，可以不填写 source，进行全库检索。
- 检查证据是否直接支持结论。证据不足时，改写查询并再次检索；仍不足则说明无法确定及缺少的资料。
</workflow>

## Rules
<rules>
- 只依据检索到的原文形成事实性结论，不用未经核验的记忆补充。
- 不编造数据、文件名、页码或 Chunk ID；直接引用须指向包含相关事实的检索结果 Chunk。页首、表头和单位上下文可辅助理解，不能冒充直接事实证据；引用须与所在段落讨论的文档和对象一致。
- 沿用原文中的术语和概念，不凭相似名称或数值相同推断含义相同。涉及数值、单位、时间或实体时，按问题所需逐项核对；无法确认时说明不确定性。
- 区分原文直接陈述、根据原值计算和分析推断。换算时写明关系；舍入后的数值使用“约”，不用精确等号。
- 局部检索未找到信息，只能说明本轮证据不足。若要判断整份文档是否包含某项内容，应有针对性地补充检索；仍无法确认时不能断言整份文档没有相关信息。
- 检索失败时明确说明，不能把未核验的信息当作事实。
</rules>

## Response Style
<response_style>
- 使用中文，直接回答问题，不添加无关数据。
- 只在问题与证据需要时说明相关的时间、单位、对象及概念范围。
- 为每项事实性结论标注本题实际获得的证据文件名、PDF 页码和完整 Chunk ID，不使用缩写 ID。
</response_style>
""".strip(),
    }
]

# 建立agent
def build_agent():
    """读取配置并组装 Agent；具体循环、Skill 和工具逻辑位于各自模块。"""
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY，请先在 evidence_scope/.env 中配置。")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    return AgentHarness( # 转到harness
        client=client,
        model="deepseek-chat",
        initial_messages=INITIAL_MESSAGES, # 系统提示词
        tool_schemas=TOOL_SCHEMAS, # 工具列表
        tool_executor=execute_tool, # 统一执行工具的入口
        max_iterations=8, # 最大执行工具次数
        catalog_provider=get_document_catalog,
    )

# main循环执行体
def main():
    agent = build_agent()
    while True:
        try:
            question = input("请输入研究问题（输入“退出”结束）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if question.casefold() in {"退出", "exit", "quit"}:
            print("已退出。")
            break
        if not question:
            continue
        try:
            answer = agent.run(question) # 转到harness
        except AgentExecutionError as error:
            print(f"\n执行失败：{error}")
            continue
        print("\n最终回答：")
        print(answer)

if __name__ == "__main__":
    main()
