"""通用 Agent Harness：负责模型调用、循环控制和工具执行。"""

import json
from openai import OpenAIError


class AgentExecutionError(RuntimeError):
    """Agent 无法完成当前任务。"""

class ModelServiceError(AgentExecutionError):
    """上游模型服务调用失败。"""

class AgentHarness:
    def __init__(self,client, model,initial_messages,tool_schemas,tool_executor,max_iterations=6,catalog_provider=None):
        self.client = client
        self.model = model
        self.initial_messages = initial_messages
        self.tool_schemas = tool_schemas
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations
        self.catalog_provider = catalog_provider

    def run(self, question):
        """处理一个用户问题，返回 Agent 的最终文本答案。"""
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空。")
        messages = list(self.initial_messages)
        if self.catalog_provider is not None:
            try:
                catalog = self.catalog_provider()
            except (ValueError, RuntimeError, OSError) as error:
                raise AgentExecutionError("文档目录读取失败，请检查 PDF 与索引。") from error
            messages.append({
                "role": "system",
                "content": "当前本地 PDF 文档目录（仅用于选择文件，不是文档内容证据）："
                           + json.dumps(catalog, ensure_ascii=False),
            })
        messages.append({"role": "user", "content": question})
        tool_failed = False
        tool_succeeded = False
        for _ in range(self.max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
                )
            except OpenAIError as error:
                raise ModelServiceError("模型服务暂时不可用，请稍后重试。") from error
            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump(exclude_none=True))
            # 此处没有工具调用表示模型已经形成最终答案。
            if not assistant_message.tool_calls:
                if tool_failed and not tool_succeeded:
                    raise AgentExecutionError("检索工具执行失败，未能完成回答。")
                if not assistant_message.content:
                    raise AgentExecutionError("模型没有返回文本答案。")
                return assistant_message.content
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                print("\n模型请求调用：", tool_name)
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    result = self.tool_executor(tool_name, arguments)
                except (json.JSONDecodeError,ValueError,KeyError,TypeError,RuntimeError,OSError,) as error:
                    # 把错误作为工具结果交还给模型，让模型可以解释或改用其他参数。
                    result = json.dumps(
                        {"error": str(error)},
                        ensure_ascii=False,
                    )
                try:
                    tool_result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    tool_result = {"error": "工具返回数据格式错误"}
                if isinstance(tool_result, dict) and "error" in tool_result:
                    tool_failed = True
                else:
                    tool_succeeded = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        raise AgentExecutionError("Agent 达到最大迭代次数，未能完成回答。")


