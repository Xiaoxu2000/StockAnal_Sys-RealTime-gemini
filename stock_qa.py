# -*- coding: utf-8 -*-
"""
智能分析系统（股票） - 股票市场数据分析系统
开发者：熊猫大侠
修改：XiaoXu
版本：v2.2.0 (Refactored for modern Gemini SDK)
许可证：MIT License

stock_qa.py - 提供股票相关问题的智能问答功能，支持联网搜索实时信息和多轮对话
"""
import os
import json
import uuid
import traceback
import logging
from urllib.parse import urlparse
from datetime import datetime

# (*** 关键修改：使用标准的、推荐的导入方式 ***)
import google.generativeai as genai
import google.generativeai.types as glm
from google.generativeai import protos  # 直接导入 protos 模块

class StockQA:
    def __init__(self, analyzer, gemini_api_key=None, gemini_model=None):
        self.analyzer = analyzer
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', gemini_api_key)
        # (*** 简化：模型名称统一管理，不再需要 function_call_model ***)
        self.gemini_model_name = os.getenv('GEMINI_API_MODEL', gemini_model or 'gemini-1.5-flash-latest')
        self.serp_api_key = os.getenv('SERP_API_KEY')
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.max_qa_rounds = int(os.getenv('MAX_QA_ROUNDS', '10'))

        self.conversation_history = {}
        self.logger = logging.getLogger(__name__)

        if not self.gemini_api_key:
            self.logger.error("未配置GEMINI_API_KEY，智能问答功能将不可用。")
        else:
            genai.configure(api_key=self.gemini_api_key)

    def answer_question(self, stock_code, question, market_type='A', conversation_id=None, clear_history=False):
        """
        回答关于股票的问题，支持联网搜索实时信息和多轮对话 (使用现代 Gemini SDK)
        """

        print("testing answer_question method...")  # 测试代码
        try:
            if not self.gemini_api_key:
                return {"error": "未配置Gemini API密钥，无法使用智能问答功能"}

            # --- 1. 对话和历史管理 ---
            if conversation_id is None:
                conversation_id = f"{stock_code}_{uuid.uuid4().hex[:8]}"
            
            if clear_history or conversation_id not in self.conversation_history:
                # (*** 优化：初始化时直接注入上下文，避免重复发送 ***)
                self.logger.info(f"为新对话 {conversation_id} 准备上下文...")
                stock_context = self._get_stock_context(stock_code, market_type)
                
                # 创建一个初始的历史记录，包含背景资料
                initial_history = [
                    protos.Content(role="user", parts=[
                        protos.Part(text=f"我们来分析股票 {stock_context.get('stock_name', stock_code)}。这是它的背景资料，请基于此进行回答：\n\n{stock_context['context']}")
                    ]),
                    protos.Content(role="model", parts=[
                        protos.Part(text="好的，我已经理解了这支股票的背景资料。请提出你的问题。")
                    ])
                ]
                self.conversation_history[conversation_id] = {
                    "history": initial_history,
                    "context": stock_context # 存储上下文供后续使用
                }
            
            # 从存储中获取当前对话的历史和上下文
            chat_session = self.conversation_history[conversation_id]
            current_history = chat_session["history"]
            stock_context = chat_session["context"]
            stock_name = stock_context.get("stock_name", "未知")

            # --- 2. 定义工具和模型 ---
            # (*** 关键修改：使用 SDK 对象定义工具，而不是字典 ***)
            search_tool = protos.Tool(
                function_declarations=[
                    protos.FunctionDeclaration(
                        name="search_stock_news",
                        description="当需要获取实时、最新的市场信息时，使用此工具搜索股票相关的最新新闻、公告和行业动态。",
                        parameters=protos.Schema(
                            type=protos.Type.OBJECT,
                            properties={
                                "query": protos.Schema(
                                    type=protos.Type.STRING, 
                                    description="一个精确的搜索查询词，用于查找最相关的新闻。例如：'贵州茅台最新财报' 或 '半导体行业政策'"
                                )
                            },
                            required=["query"]
                        )
                    )
                ]
            )

            system_instruction = """你是摩根大通的高级宏观策略师和首席投资顾问，拥有哈佛经济学博士学位和20年华尔街顶级投行经验。你同时也是国家发改委、央行和证监会的政策研究顾问团专家，了解中国宏观经济和产业政策走向。

你的特点是：
1. 思维深度 - 从表面现象洞察深层次的经济周期、产业迁移和资本流向规律，预见市场忽视的长期趋势
2. 全局视角 - 将个股分析放在全球经济格局、国内政策环境、产业转型、供应链重构和流动性周期的大背景下
3. 结构化思考 - 运用专业框架如PEST分析、波特五力模型、杜邦分析、价值链分析和SWOT分析等系统评估
4. 多层次透视 - 能同时从资本市场定价、产业发展阶段、公司竞争地位和治理结构等维度剖析股票价值
5. 前瞻预判 - 善于前瞻性分析科技创新、产业政策和地缘政治变化对中长期市场格局的影响

沟通时，你会：
- 将复杂的金融概念转化为简洁明了的比喻和案例，使普通投资者理解专业分析
- 强调投资思维和方法论，而非简单的买卖建议
- 提供层次分明的分析：1)微观公司基本面 2)中观产业格局 3)宏观经济环境
- 引用相关研究、历史案例或数据支持你的观点
- 在必要时搜索最新资讯，确保观点基于最新市场情况
- 兼顾短中长期视角，帮助投资者建立自己的投资决策框架

作为金融专家，你始终：
- 谨慎评估不同情景下的概率分布，而非做出确定性预测
- 坦承市场的不确定性和你认知的边界
- 同时提供乐观和保守的观点，帮助用户全面权衡
- 强调风险管理和长期投资价值
- 避免传播市场谣言或未经证实的信息

请记住，你的价值在于提供深度思考框架和专业视角，帮助投资者做出明智决策，而非简单的投资指令。在需要时，使用search_stock_news工具获取最新市场信息。
"""

            model = genai.GenerativeModel(
                model_name=self.gemini_model_name,
                tools=[search_tool],
                system_instruction=system_instruction
            )
            
            # (*** 优化：使用已包含上下文的历史记录启动聊天 ***)
            chat = model.start_chat(history=current_history)

            # --- 3. 发送请求并处理工具调用 ---
            self.logger.info(f"向Gemini发送问题: '{question}'")
            response = chat.send_message(question, stream=False)
            
            response_part = response.parts[0]

            # (*** 在这里添加测试代码 ***)
            self.logger.info("="*50)
            self.logger.info("【测试阶段1：检查首次响应】")
            if hasattr(response_part, 'text'):
                # 使用 repr() 获取最详细、最明确的对象表示
                self.logger.info(f"模型返回的 function_call 对象: {repr(response_part.text)}")
            if response_part.function_call:
                self.logger.info(f"提取到的 function_name: '{response_part.function_call.name}'")
                self.logger.info(f"提取到的 function_args: {dict(response_part.function_call.args)}")
            else:
                self.logger.info("模型没有返回 function_call。")
            self.logger.info("="*50)
            # (*** 测试代码结束 ***)
            used_search_tool = False

            # (*** 关键修改：使用标准的、更健壮的方式检查和处理工具调用 ***)
            if response_part.function_call:
                self.logger.info("Gemini请求调用工具...")
                used_search_tool = True
                tool_call = response_part.function_call
                function_name = tool_call.name
                
                if function_name == 'search_stock_news':
                    self.logger.info(f"执行工具: {function_name}")
                    
                    # 安全地将 MappingProxy 转换为字典并获取参数
                    function_args = dict(tool_call.args)
                    query_param = function_args.get('query', question) # 若模型未提供query，则使用原始问题作为后备

                    # 执行本地函数
                    search_results = self.search_stock_news(
                        query=query_param,
                        stock_name=stock_name, 
                        stock_code=stock_code,
                        industry=stock_context.get('industry', '未知'),
                        market_type=market_type
                    )

                    # (*** 在这里添加测试代码 ***)
                    self.logger.info("="*50)
                    self.logger.info("【测试阶段2：检查本地函数返回值】")
                    self.logger.info(f"search_stock_news 返回的结果: {json.dumps(search_results, indent=2, ensure_ascii=False)}")
                    self.logger.info("="*50)
                    # (*** 测试代码结束 ***)
                    
                    self.logger.info("将工具执行结果返回给Gemini...")
                    # (*** 在这里添加测试代码 ***)
                    tool_response_message = protos.Content( # 你当前使用的错误方式
                        role="tool",
                        parts=[
                            protos.FunctionResponse(
                                name=function_name,
                                response={
                                    "content": json.dumps(search_results, ensure_ascii=False)
                                }
                            )
                        ]
                    )
                    self.logger.info("="*50)
                    self.logger.info("【测试阶段3：检查待发送的工具响应】")
                    self.logger.info(f"准备发送的消息对象类型: {type(tool_response_message)}")
                    self.logger.info(f"准备发送的消息内容: {tool_response_message}")
                    self.logger.info("="*50)

                    # (*** 关键修改：使用 protos.Part.from_function_response 封装结果 ***)
                    # response = chat.send_message(
                    #     protos.Content(
                    #         role="tool",
                    #         parts=[
                    #             protos.FunctionResponse(
                    #                 name=function_name,
                    #                 response={
                    #                     "content": json.dumps(search_results, ensure_ascii=False)
                    #                 }
                    #             )
                    #         ]
                    #     ),
                    #     stream=False
                    # )
                    # response_content = response.text
                    try:
                        response = chat.send_message(tool_response_message, stream=False)
                        response_content = response.text
                    except Exception as api_error:
                        self.logger.error("【！！！二次API调用失败！！！】", exc_info=True)
                        response_content = f"抱歉，在处理实时信息时发生API错误: {api_error}"
                # else:
                #     self.logger.warning(f"模型请求调用一个未知的工具: {function_name}")
                #     response_content = f"抱歉，我被要求使用一个无法识别的工具({function_name})。"
            else:
                self.logger.info("Gemini未调用工具，直接生成回答。")
                response_content = response.text
                
            # --- 4. 更新历史并返回结果 ---
            # chat.history 会自动包含所有步骤：用户问题、模型工具调用、工具响应、模型最终回答
            self.conversation_history[conversation_id]["history"] = chat.history
            
            # (*** 优化：历史长度裁剪，只保留最新的对话轮次 ***)
            self._trim_history(conversation_id)
            
            return {
                "conversation_id": conversation_id,
                "question": question,
                "answer": response_content,
                "stock_code": stock_code,
                "stock_name": stock_name,
                "used_search_tool": used_search_tool,
                # 对话轮次 = (总条目数 - 初始2条上下文) / 2
                "conversation_length": (len(chat.history) - 2) // 2 
            }

        except Exception as e:
            self.logger.error(f"智能问答出错: {e}", exc_info=True)
            return {
                "question": question,
                "answer": f"抱歉，回答问题时出错: {str(e)}",
                "stock_code": stock_code,
                "error": str(e)
            }

    def _trim_history(self, conversation_id):
        """裁剪对话历史，防止其无限增长"""
        if conversation_id not in self.conversation_history:
            return

        history = self.conversation_history[conversation_id]["history"]
        
        # 保留初始的2条上下文消息 + 最近的 N 轮对话 (每轮2条)
        max_len = 2 + self.max_qa_rounds * 2
        
        if len(history) > max_len:
            self.logger.info(f"对话 {conversation_id} 历史过长，进行裁剪...")
            # 保留前2条（上下文）和后 max_qa_rounds*2 条（最新对话）
            trimmed_history = history[:2] + history[-self.max_qa_rounds * 2:]
            self.conversation_history[conversation_id]["history"] = trimmed_history

    def _get_stock_context(self, stock_code, market_type='A'):
        """获取股票上下文信息"""
        try:
            # 获取股票信息
            stock_info = self.analyzer.get_stock_info(stock_code)
            stock_name = stock_info.get('股票名称', '未知')
            industry = stock_info.get('行业', '未知')

            # 获取技术指标数据
            df = self.analyzer.get_stock_data(stock_code, market_type)
            df = self.analyzer.calculate_indicators(df)

            # 提取最新数据
            latest = df.iloc[-1]

            # 计算评分
            score = self.analyzer.calculate_score(df)

            # 获取支撑压力位
            sr_levels = self.analyzer.identify_support_resistance(df)

            # 构建上下文
            context = f"""股票信息:
- 代码: {stock_code}
- 名称: {stock_name}
- 行业: {industry}

技术指标(最新数据):
- 价格: {latest['close']}
- 5日均线: {latest['MA5']}
- 20日均线: {latest['MA20']}
- 60日均线: {latest['MA60']}
- RSI: {latest['RSI']}
- MACD: {latest['MACD']}
- MACD信号线: {latest['Signal']}
- 布林带上轨: {latest['BB_upper']}
- 布林带中轨: {latest['BB_middle']}
- 布林带下轨: {latest['BB_lower']}
- 波动率: {latest['Volatility']}%

技术评分: {score}分

支撑位:
- 短期: {', '.join([str(level) for level in sr_levels['support_levels']['short_term']])}
- 中期: {', '.join([str(level) for level in sr_levels['support_levels']['medium_term']])}

压力位:
- 短期: {', '.join([str(level) for level in sr_levels['resistance_levels']['short_term']])}
- 中期: {', '.join([str(level) for level in sr_levels['resistance_levels']['medium_term']])}"""

            # 尝试获取基本面数据
            try:
                # 导入基本面分析器
                from fundamental_analyzer import FundamentalAnalyzer
                fundamental = FundamentalAnalyzer()

                # 获取基本面数据
                indicators = fundamental.get_financial_indicators(stock_code)

                # 添加到上下文
                context += f"""

基本面指标:
- PE(TTM): {indicators.get('pe_ttm', '未知')}
- PB: {indicators.get('pb', '未知')}
- ROE: {indicators.get('roe', '未知')}%
- 毛利率: {indicators.get('gross_margin', '未知')}%
- 净利率: {indicators.get('net_profit_margin', '未知')}%"""
            except Exception as e:
                self.logger.warning(f"获取基本面数据失败: {str(e)}")
                context += "\n\n注意：未能获取基本面数据"

            return {
                "context": context,
                "stock_name": stock_name,
                "industry": industry
            }
        except Exception as e:
            self.logger.error(f"获取股票上下文信息出错: {str(e)}")
            return {
                "context": f"无法获取股票 {stock_code} 的完整信息: {str(e)}",
                "stock_name": "未知",
                "industry": "未知"
            }

    def clear_conversation(self, conversation_id=None, stock_code=None):
        """
        清除特定对话或与特定股票相关的所有对话历史
        
        参数:
            conversation_id: 指定要清除的对话ID
            stock_code: 指定要清除的股票相关的所有对话
        """
        if conversation_id and conversation_id in self.conversation_history:
            # 清除特定对话
            del self.conversation_history[conversation_id]
            return {"message": f"已清除对话 {conversation_id}"}
            
        elif stock_code:
            # 清除与特定股票相关的所有对话
            removed = []
            for conv_id in list(self.conversation_history.keys()):
                if conv_id.startswith(f"{stock_code}_"):
                    del self.conversation_history[conv_id]
                    removed.append(conv_id)
            return {"message": f"已清除与股票 {stock_code} 相关的 {len(removed)} 个对话"}
            
        else:
            # 清除所有对话
            count = len(self.conversation_history)
            self.conversation_history.clear()
            return {"message": f"已清除所有 {count} 个对话"}

    def get_conversation_history(self, conversation_id):
        """获取特定对话的历史记录"""
        if conversation_id not in self.conversation_history:
            return {"error": f"找不到对话 {conversation_id}"}
            
        # 提取用户问题和助手回答
        history = []
        conversation = self.conversation_history[conversation_id]
        
        # 按对话轮次提取历史
        for i in range(0, len(conversation), 2):
            if i+1 < len(conversation):
                history.append({
                    "question": conversation[i]["content"],
                    "answer": conversation[i+1]["content"]
                })
                
        return {
            "conversation_id": conversation_id,
            "history": history,
            "round_count": len(history)
        }

    def search_stock_news(self, query: str, stock_name: str, stock_code: str, industry: str, market_type: str = 'A') -> dict:
        """搜索股票相关新闻和实时信息"""
        try:
            self.logger.info(f"搜索股票新闻: {query}")
            
            # 确定市场名称
            market_name = "A股" if market_type == 'A' else "港股" if market_type == 'HK' else "美股"
            
            # 检查API密钥
            if not self.serp_api_key and not self.tavily_api_key:
                self.logger.warning("未配置搜索API密钥")
                return {
                    "message": "无法搜索新闻，未配置搜索API密钥",
                    "results": []
                }
            
            news_results = []
            
            # 使用SERP API搜索
            if self.serp_api_key:
                try:
                    import requests
                    
                    # 构建搜索查询
                    search_query = f"{stock_name} {stock_code} {market_name} {query}"
                    
                    # 调用SERP API
                    url = "https://serpapi.com/search"
                    params = {
                        "engine": "google",
                        "q": search_query,
                        "api_key": self.serp_api_key,
                        "tbm": "nws",  # 新闻搜索
                        "num": 5  # 获取5条结果
                    }
                    
                    response = requests.get(url, params=params)
                    search_results = response.json()
                    
                    # 提取新闻结果
                    if "news_results" in search_results:
                        for item in search_results["news_results"]:
                            news_results.append({
                                "title": item.get("title", ""),
                                "date": item.get("date", ""),
                                "source": item.get("source", ""),
                                "snippet": item.get("snippet", ""),
                                "link": item.get("link", "")
                            })
                except Exception as e:
                    self.logger.error(f"SERP API搜索出错: {str(e)}")
            
            # 使用Tavily API搜索
            if self.tavily_api_key:
                try:
                    from tavily import TavilyClient
                    
                    client = TavilyClient(self.tavily_api_key)
                    
                    # 构建搜索查询
                    search_query = f"{stock_name} {stock_code} {market_name} {query}"
                    
                    # 调用Tavily API
                    response = client.search(
                        query=search_query,
                        topic="finance",
                        search_depth="advanced"
                    )
                    
                    # 提取结果
                    if "results" in response:
                        for item in response["results"]:
                            # 从URL提取域名作为来源
                            source = ""
                            if item.get("url"):
                                try:
                                    parsed_url = urlparse(item.get("url"))
                                    source = parsed_url.netloc
                                except:
                                    source = "未知来源"
                            
                            news_results.append({
                                "title": item.get("title", ""),
                                "date": datetime.now().strftime("%Y-%m-%d"),  # Tavily不提供日期
                                "source": source,
                                "snippet": item.get("content", ""),
                                "link": item.get("url", "")
                            })
                except ImportError:
                    self.logger.warning("未安装Tavily客户端库，请使用pip install tavily-python安装")
                except Exception as e:
                    self.logger.error(f"Tavily API搜索出错: {str(e)}")
            
            # 去重并限制结果数量
            unique_results = []
            seen_titles = set()
            
            for item in news_results:
                title = item.get("title", "").strip()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_results.append(item)
                    if len(unique_results) >= 5:  # 最多返回5条结果
                        break
            
            # 创建格式化的摘要文本
            summary_text = ""
            for i, item in enumerate(unique_results):
                summary_text += f"{i+1}、{item.get('title', '')}\n"
                summary_text += f"{item.get('snippet', '')}\n"
                summary_text += f"来源: {item.get('source', '')} {item.get('date', '')}\n\n"
            
            return {
                "message": f"找到 {len(unique_results)} 条相关新闻",
                "results": unique_results,
                "summary": summary_text
            }
            
        except Exception as e:
            self.logger.error(f"搜索股票新闻时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "message": f"搜索新闻时出错: {str(e)}",
                "results": []
            }