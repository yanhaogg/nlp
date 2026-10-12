from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import wikipedia
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()

# ============  1.Single Agent   =========
@tool
def wikipedia_search(query):
    '''使用维基百科API搜索信息(支持中文)'''
    try:
        # 设置语言为中文
        wikipedia.set_lang("zh")
        page = wikipedia.page(query,auto_suggest= False)
        return f"标题 : {page.title}\n摘要 : {page.summary}"
    # 捕捉到的错误,第一,防止程序崩溃,会继续运行
    # 第二,捕捉到的报错信息,会帮助 Agent 排查错误
    except wikipedia.DisambiguationError as e:
        options = e.options[:3]
        return f"模糊匹配 : 请从以下选项中选择: {','.join(options)}"
    except wikipedia.PageError:
        return "未找到相关信息"
    except Exception as e:
        return f"搜索错误 : {str(e)}"

llm = LLM(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE")),

single_agent = create_agent(
    model = llm,
    tools = [wikipedia_search],
    system_prompt='''
角色:你是一个专业金融科技作家,
任务:撰写一篇约500字的短文,主题为<量子计算机对金融行业的影响>
约束:
包含至少3个核心观点
文笔简洁专业,将技术概念转化为通俗易懂的文字
检查文章的事实准确性,逻辑连贯性和可读性,对事实错误和表述问题零容忍
输出内容:给出最终优化后的文章
''',
    debug = False
)
user_input = "撰写一篇量子计算对金融行业影响的约500字的短文"
single_output = single_agent.invoke({
    "messages" : [HumanMessage(content = user_input)]
})
single_res = single_output["messages"][-1].content
print(single_res)

# ================ 2.Multi-Agent  =================
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import wikipedia
from dotenv import load_dotenv
load_dotenv()

@tool
def wikipedia_search(query):
    '''使用维基百科API搜索信息(支持中文)'''
    try:
        # 设置语言为中文
        wikipedia.set_lang("zh")
        page = wikipedia.page(query,auto_suggest= False)
        return f"标题 : {page.title}\n摘要 : {page.summary}"
    # 捕捉到的错误,第一,防止程序崩溃,会继续运行
    # 第二,捕捉到的报错信息,会帮助 Agent 排查错误
    except wikipedia.DisambiguationError as e:
        options = e.options[:3]
        return f"模糊匹配 : 请从以下选项中选择: {','.join(options)}"
    except wikipedia.PageError:
        return "未找到相关信息"
    except Exception as e:
        return f"搜索错误 : {str(e)}"

researcher = Agent(
    role = "量子计算金融研究员" , # role这个Agent在 Crew 里面扮演的角色
    goal = "从可靠来源搜索并提炼量子计算对金融行业影响的3个核心观点", # 该Agent具体负责的工作内容/目标
    backstory = "你是一个严谨的金融科技研究员,擅长从学术和行业资料中提取关键洞见.", # 角色的背景故事
    tools = [wikipedia_search],
    llm = LLM(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE")),
    verbose = True, # True 和 False 指的是,是否打印详细思考过程
)

writer = Agent(
    role = "专业金融科技作家",
    goal = "基于研究员提供的核心观点, 撰写一片流畅, 结构清晰的500字短文",
    backstory = "你是一位经验丰富的金融科技作者，文笔简洁专业，擅长将技术概念转化为通俗易懂的文字。",
    llm = LLM(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE")),
    verbose = True,
)

validator = Agent(
    role = "内容校验专家",
    goal = "检查文章的事实准确性,逻辑连贯性和可读性,给出最终优化版本",
    backstory = "你是一个严苛的内容审核专家,对事实错误和表述问题零容忍",
    llm = LLM(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE")),
    verbose = True
)

task1 = Task(
    description = "搜索量子计算对金融行业的影响,提炼出3个最核心,最具有代表性的观点,并列出简要理由",
    expected_output = "3个核心观点,每个观点包含标题和1-2句解释", # 期望输出的内容
    # 可选参数 context
    agent = researcher # 将任务指派给具体的 Agent
)

task2 = Task(
    description = "基于研究员提供的核心观点,撰写一篇约 500 字的短文",
    expected_output = "一篇完整的 500 字短文",
    agent = writer,
    context = [task1] # task1 的输出,会作为 task2 的上下文
)

task3 = Task(
    description = "检查文章的事实,逻辑和可读性",
    expected_output = "最终优化后的文章 + 修改说明",
    agent = validator,
    context = [task1,task2] # 接收前面的所有任务的输出
)

crew = Crew(
    agents = [researcher,writer,validator],
    tasks = [task1,task2,task3],
    process = Process.sequential,
    verbose = True
)

multi_res = crew.kickoff()
print(multi_res.raw)

# ============  3.Judge_llm  =============
#judge_llm = ChatOpenAI(temperature = 0,model='deepseek-v4-pro',top_p = 0.8)
judge_llm = ChatOpenAI(model="deepseek-v4-pro", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE"))

scoring_prompt = """
你是一位严格、公正、专业且经验丰富的金融科技论文评审专家。
现在请你对以下两篇文章进行客观评分，主题为《量子计算对金融行业的影响》。
【文章A - 单 Agent 输出】
{article_a}
【文章B - 多 Agent 输出】
{article_b}
请严格按照以下5个维度分别打分（每个维度满分10分），并给出简要理由：
1. 忠实度（Faithfulness）：文章内容是否忠实于可靠事实，没有幻觉、编造或过时信息。
2. 相关性（Relevancy）：文章是否紧密围绕“量子计算对金融行业的影响”这一主题，没有明显跑题。
3. 结构清晰度（Coherence & Structure）：文章逻辑是否连贯、层次是否分明、结构是否完整（引言、主体、结
论）。
4. 专业性（Professionalism & Depth）：术语使用是否准确、专业深度是否足够，分析是否有洞见。
5. 可读性（Readability）：语言是否流畅、自然、通俗易懂，适合非专业读者阅读。
最后给出总分（满分50分）和最终推荐（哪篇文章更好，为什么）。

请严格以以下 JSON 格式输出，不要添加任何多余文字:
{{
    "faithfulness": xx,
    "relevancy": xx,
    "structure": xx,
    "professionalism": xx,
    "readability": xx,
    "total_score": xx,
    "winner": "A 或 B",
    "reason": "简要总结理由"
}}
"""
prompt = scoring_prompt.format(article_a = single_res,article_b = multi_res.raw)
res = judge_llm.invoke(prompt)
print(res.content)

# 16 : 03