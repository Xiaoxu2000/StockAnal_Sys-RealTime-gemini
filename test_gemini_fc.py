import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# 1. 加载环境变量
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 2. 模拟的本地工具函数 (保持不变)
def search_stock_news(query: str):
    print(f"\n[本地代码] 正在执行 search_stock_news(query='{query}')...")
    import time
    time.sleep(1)
    mock_news = {
        "summary": f"关于“{query}”的搜索结果：",
        "results": [{"title": f"新闻标题1: {query}", "source": "财经日报"}]
    }
    print(f"[本地代码] search_stock_news 执行完毕。\n")
    return mock_news

# 3. 主测试函数 (使用旧版字典语法)
def run_test():
    print("--- 开始 Gemini Function Calling 测试 (旧版字典声明模式) ---")

    # ================================================================
    #  【【【 核心修改：使用纯字典来声明工具 】】】
    # ================================================================
    # 这是一种更通用的、符合OpenAPI v3 Schema的字典格式
    search_tool_declaration_dict = {
        "function_declarations": [
            {
                "name": "search_stock_news",
                "description": "根据给定的查询词，搜索最新的股票或财经新闻。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "用于搜索新闻的关键词，例如 \"新能源汽车补贴政策\""
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    }
    
    # a. 初始化模型，并传入字典格式的工具
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
        tools=[search_tool_declaration_dict] # <--- 传入字典
    )

    # b. 创建聊天会话
    chat = model.start_chat()

    # c. 构造用户问题
    user_prompt = "请帮我搜索一下关于'新能源汽车补贴政策'的最新新闻"
    print(f"用户提问: {user_prompt}\n")

    # d. 发送消息
    first_response = chat.send_message(user_prompt)
    
    response_part = first_response.candidates[0].content.parts[0]

    # f. 处理函数调用 (这部分逻辑和之前一样，但我们使用protos来构建响应，兼容性最好)
    if response_part.function_call:
        print("--- 模型决定调用工具 ---")
        tool_call = response_part.function_call
        function_name = tool_call.name
        
        if function_name == 'search_stock_news':
            function_args = dict(tool_call.args)
            function_response_content = search_stock_news(query=function_args['query'])
            
            # 使用 protos 构建响应，这是最兼容的方式
            from google.generativeai import protos
            tool_response_part = protos.Part(
                function_response=protos.FunctionResponse(
                    name=function_name,
                    response={'content': json.dumps(function_response_content, ensure_ascii=False)}
                )
            )
            
            second_response = chat.send_message(tool_response_part)
            
            print("\n--- 最终结果：模型根据工具返回结果生成的回答 ---")
            print(second_response.text)
        else:
            print(f"错误：模型请求调用未知的函数 '{function_name}'")
    else:
        print("--- 模型决定不调用工具，直接回答 ---")
        print("\n--- 最终结果：模型的直接回答 ---")
        print(first_response.text)

# 4. 运行测试
if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"\n--- 测试过程中发生错误 ---\n{type(e).__name__}: {e}")