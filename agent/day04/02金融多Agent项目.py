from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import wikipedia
from dotenv import load_dotenv
import os
load_dotenv()

# ================  项目说明   ==================
# 项目 : 撰写一篇 <量子计算对金融行业的影响> 500字短文
# 团队分工 : 研究员 -> 作家 -> 校验(顺序执行)
# 1.crewai不支持langchain的组件,tool必须来自crewai.tools,llm 必须来自crewai.LLM
# 2.verbose 只支持 True 和 False,不支持数字
# 3.这里的Agent默认使用 gpt-4o,即不需要往Agent里面传llm,前提是load_dotenv时,必须有gpt-4o的API_KEY
# 4.把API_KEY里面的 OPENAI_API_KEY和OPENAI_API_BASE 重新切换回deepseek

# 关于日志细节,如果不想看具体的细节,可以把verbose关掉,即设置为False
# 最后的输出 res,相对不是纯净文字, res.raw

# ================ 1. 定义工具  ===================
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

# ================ 2. 定义 Agent  ===============
researcher = Agent(
    role = "量子计算金融研究员" , # role这个Agent在 Crew 里面扮演的角色
    goal = "从可靠来源搜索并提炼量子计算对金融行业影响的3个核心观点", # 该Agent具体负责的工作内容/目标
    backstory = "你是一个严谨的金融科技研究员,擅长从学术和行业资料中提取关键洞见.", # 角色的背景故事
    tools = [wikipedia_search],
    llm = LLM(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE")),
    verbose = False, # True 和 False 指的是,是否打印详细思考过程
)

writer = Agent(
    role = "专业金融科技作家",
    goal = "基于研究员提供的核心观点, 撰写一片流畅, 结构清晰的500字短文",
    backstory = "你是一位经验丰富的金融科技作者，文笔简洁专业，擅长将技术概念转化为通俗易懂的文字。",
    llm = LLM(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE"),temperature = 1.2,top_p = 0.92),
    #llm = LLM(model = "deepseek-v4-flash",temperature = 1.2,top_p = 0.92),
    verbose = False,
)

validator = Agent(
    role = "内容校验专家",
    goal = "检查文章的事实准确性,逻辑连贯性和可读性,给出最终优化版本",
    backstory = "你是一个严苛的内容审核专家,对事实错误和表述问题零容忍",
    #llm = LLM(model = "deepseek-v4-flash",temperature = 0.2,top_p = 0.6),
    llm = LLM(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE"),temperature = 0.2,top_p = 0.6),
    verbose = False
)

# ================  3. 定义三个 Task (带依赖关系)  =========
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

# ================  4.组件 crew  ==============
crew = Crew(
    agents = [researcher,writer,validator],
    tasks = [task1,task2,task3],
    process = Process.sequential,
    verbose = True  # True 和 False 指的是,是否打印详细思考过程
)

res = crew.kickoff()
# print(res)

