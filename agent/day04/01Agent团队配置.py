# # Agent 角色提示词,工具集,记忆

# from crewai import Agent, Task, Crew, Process,Memory, ShortTermMemory, LongTermMemory
# from langchain_core.tools import tool
# from model.toolkit import *
# #from crewai.memory import ShortTermMemory, LongTermMemory
# # 删除无法导入的 ShortTermMemory, LongTermMemory

# # 1.配置 Agent
# researcher = Agent(
#     role = "量子计算金融研究员",  # role这个Agent在 Crew 里面扮演的角色
#     goal = "从可靠来源搜索并提炼量子计算对金融行业影响的3个核心观点",  # 该Agent具体负责的工作内容/目标
#     backstory = "你是一个严谨的金融科技研究员,擅长从学术和行业资料中提取关键洞见.",  # 角色的背景故事
#     tools = [wikipedia_search, get_stock_history],
#     verbose = True,  # True 和 False 指的是,是否打印详细思考过程
#     memory = True,   # 启用记忆（具体配置移到 Crew 层）


#     # 删除 short_term_memory 和 long_term_memory 参数
# )

# # 补全 writer Agent（你需要根据实际需求修改）
# writer = Agent(
#     role = "金融科技内容撰写者",
#     goal = "根据研究员提供的核心观点撰写清晰、有说服力的短文",
#     backstory = "你是一位资深金融科技编辑，擅长将专业观点转化为通俗易懂的文章。",
#     tools = [],  # 根据需要添加工具
#     verbose = True,
#     memory = True,
# )

# # 补全 validator Agent
# validator = Agent(
#     role = "文章审核与优化专家",
#     goal = "检查文章的事实准确性、逻辑连贯性和可读性，并给出修改建议",
#     backstory = "你是一位严谨的资深审稿人，拥有丰富的金融和科技领域编辑经验。",
#     tools = [],
#     verbose = True,
#     memory = True,
# )

# # 2.配置任务 Task
# task1 = Task(
#     description = "搜索量子计算对金融行业的影响,提炼出3个最核心,最具有代表性的观点,并列出简要理由",
#     expected_output = "3个核心观点,每个观点包含标题和1-2句解释",  # 期望输出的内容
#     agent = researcher  # 将任务指派给具体的 Agent
# )

# task2 = Task(
#     description = "基于研究员提供的核心观点,撰写一篇约 500 字的短文",
#     expected_output = "一篇完整的 500 字短文",
#     agent = writer,
#     context = [task1]  # task1 的输出,会作为 task2 的上下文
# )

# task3 = Task(
#     description = """
# 检查 task2 文章的事实准确性、逻辑连贯性和可读性。
# 如果存在问题，请直接修改原文，并输出优化后的完整文章，
# 最后附上简要修改说明。
# """,
# #"检查文章的事实,逻辑和可读性",
#     expected_output = "最终优化后的文章 + 修改说明",
#     agent = validator,
#     context = [task1, task2]  # 接收前面的所有任务的输出
# )

# # 3. 配置团队
# crew = Crew(
#     agents = [researcher, writer, validator],  # 包含所有 Agent
#     tasks = [task1, task2, task3],             # 包含所有任务
#     process = Process.sequential,              # 团队流程 : 任务依赖关系(串行)
#     verbose = True,                               # 显示详细执行日志
#     memory = True,                             # 启用记忆

#     short_term_memory = ShortTermMemory(
#         max_recent_interactions = 10,  # 短期记忆配置
#     ),
#     long_term_memory = LongTermMemory(
#         storage = "chroma",  # 长期记忆存储后端
#         collection_name = "researcher_long_term_memory",  # 长期记忆集合名称
#     ),

# )
# '''
# - Qdrant：CrewAI 有明确的集成示例，可以作为短期/实体记忆后端；Qdrant 官方文档给了自定义 RAGStorage 的方式。Qdrant
# - Pinecone：可以接，但通常要自己封装 StorageBackend
# - Weaviate：同样可以作为自定义后端
# - Milvus / Zilliz Cloud：可以接
# - pgvector / PostgreSQL：可以接，特别适合你已经有 PostgreSQL 的项目
# - OpenSearch / Elasticsearch Vector Search：也可以
# - Chroma：你现在已经在用，适合本地开发
# - LanceDB：CrewAI 现在默认 Memory 后端就是它。GitHub

# '''






import os
import json
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM, Memory
from langchain_openai import OpenAIEmbeddings
from crewai.llms.providers.openai.completion import OpenAICompletion


# =========================================================
# 1. 环境配置
# =========================================================

load_dotenv()

BASE_DIR = Path("agent/day04/crewai_memory")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# 这里使用 CrewAI 的 LLM 类，而不是 LangChain 的 ChatOpenAI
llm = LLM(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE"))


class MemoryTextLLM(OpenAICompletion):
    """通过提示词和本地校验实现结构化记忆，不向接口发送 json_schema。"""

    def call(self, messages, *args, response_model=None, **kwargs):
        if response_model is None:
            return super().call(messages, *args, **kwargs)

        # Memory 自动传入 Pydantic 模型；把结构要求放进提示词，
        # 不把 response_model 交给父类，避免生成不受支持的 response_format。
        schema = response_model.model_json_schema()
        instruction = (
            "Return only a valid JSON object matching this JSON Schema. "
            "Do not use Markdown fences or add fields absent from the schema. "
            "For extracted_metadata use only entities, dates and topics.\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        if isinstance(messages, str):
            prompt = [{"role": "user", "content": messages}]
        else:
            prompt = [dict(message) for message in messages]
        prompt.append({"role": "user", "content": instruction})

        for attempt in range(2):
            content = super().call(prompt, *args, **kwargs)
            text = content.strip()
            # 部分兼容接口仍会包裹 Markdown，解析前去掉外层代码块。
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            try:
                return response_model.model_validate_json(text)
            except ValueError as error:
                if attempt == 1:
                    raise
                prompt.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Correct the JSON validation errors: {error}"},
                ])


memory_llm = MemoryTextLLM(
    model="deepseek-flash",
    api_key=os.getenv("DS_API_KEY"),
    base_url=os.getenv("DS_API_BASE"),
)


# =========================================================
# 2. Embedding 配置
# =========================================================

embeddings = OpenAIEmbeddings(
    model=os.environ["QIANWEN"],
    api_key=os.environ["BAIDU_QIANFAN_API_KEY"],
    base_url=os.environ["BAIDU_QIANFAN_BASE_URL"],
    check_embedding_ctx_length=False,
    chunk_size=5
)

# CrewAI Memory 需要：
# 输入 list[str]
# 输出 list[list[float]]
def embed_texts(texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)


# =========================================================
# 3. 定义记忆空间
# =========================================================

def create_memory(name: str):
    return Memory(
        storage=str(BASE_DIR / name),
        embedder=embed_texts,
        llm=memory_llm
    )


# 3.1 三个 Agent 的私有记忆
researcher_memory = create_memory("researcher")
writer_memory = create_memory("writer")
validator_memory = create_memory("validator")

# 3.2 Crew 的共享记忆
shared_memory = create_memory("shared")


# =========================================================
# 4. 初始化示例记忆
# =========================================================

# 明确存入哪个 scope，避免自动分类带来不确定性

researcher_memory.remember(
    "研究量子计算金融应用时，应区分理论潜力与实证结果。",
    scope="/research/experience",
    categories=["research"]
)

writer_memory.remember(
    "金融科技文章采用准确、简洁、易理解的风格。",
    scope="/writing/style",
    categories=["writing"]
)

validator_memory.remember(
    "审核金融科技文章时，要警惕夸大计算加速和商业成熟度。",
    scope="/review/rules",
    categories=["review"]
)

shared_memory.remember(
    "研究主题是量子计算在金融领域的应用，"
    "重点关注组合优化、风险分析和金融建模。",
    scope="/team/knowledge",
    categories=["project"]
)


# =========================================================
# 5. 查询共享记忆
# =========================================================

def recall_shared_context(question: str) -> str:
    matches = shared_memory.recall(
        question,
        limit=5,
        depth="shallow"
    )

    if not matches:
        return "暂无历史共享知识。"

    return "\n".join(
        f"- {item.record.content}"
        for item in matches
    )


# =========================================================
# 6. 创建 Agents
# =========================================================

researcher = Agent(
    role="量子计算金融研究员",
    goal="提炼量子计算在金融行业的3个核心应用方向",
    backstory=(
        "你是一名严谨的金融科技研究员，"
        "善于分析研究结论与技术局限。"
    ),
    llm=llm,
    memory=researcher_memory,
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role="金融科技内容撰写者",
    goal="根据研究结果撰写清晰易懂的金融科技短文",
    backstory=(
        "你是一位资深金融科技编辑，"
        "擅长将专业内容转化为通俗易懂的文章。"
    ),
    llm=llm,
    memory=writer_memory,
    verbose=True,
    allow_delegation=False
)

validator = Agent(
    role="文章审核与优化专家",
    goal="检查文章的事实准确性、逻辑和可读性",
    backstory=(
        "你是一位严格的专业审稿人，"
        "擅长发现不严谨的技术论述。"
    ),
    llm=llm,
    memory=validator_memory,
    verbose=True,
    allow_delegation=False
)


# =========================================================
# 7. 创建 Tasks
# =========================================================

task1 = Task(
    description=(
        "研究量子计算在金融领域的应用，"
        "总结3个核心观点。"
        "每个观点包含应用方向、潜在价值与局限。"
        "不得编造论文或实验数据。\n\n"
        "团队共享知识：\n{shared_knowledge}"
    ),
    expected_output="3个核心观点，每个观点包含解释和局限。",
    agent=researcher
)

task2 = Task(
    description=(
        "根据研究员的输出撰写一篇约500字的短文。"
        "文章应包含引言、主要应用与局限性。\n\n"
        "团队共享知识：\n{shared_knowledge}"
    ),
    expected_output="约500字的完整中文文章。",
    agent=writer,
    context=[task1]
)

task3 = Task(
    description=(
        "根据研究结果审核撰写的文章。"
        "检查事实、逻辑和可读性。"
        "对有问题的表述直接进行修改，"
        "最后输出完整文章和简要修改说明。\n\n"
        "团队共享知识：\n{shared_knowledge}"
    ),
    expected_output="最终优化后的文章，以及简要修改说明。",
    agent=validator,
    context=[task1, task2]
)


# =========================================================
# 8. 创建 Crew
# =========================================================

crew = Crew(
    agents=[researcher, writer, validator],
    tasks=[task1, task2, task3],
    process=Process.sequential,
    memory=shared_memory,
    verbose=True
)


# =========================================================
# 9. 执行
# =========================================================

shared_context = recall_shared_context(
    "量子计算金融领域的研究主题和应用方向"
)

result = crew.kickoff(
    inputs={
        "shared_knowledge": shared_context
    }
)

print("\n========== 最终结果 ==========")
print(result.raw)


# =========================================================
# 10. 查看记忆存储情况
# =========================================================

print("\n========== 记忆检查 ==========")

memory_spaces = {
    "Researcher私有记忆": researcher_memory,
    "Writer私有记忆": writer_memory,
    "Validator私有记忆": validator_memory,
    "Crew共享记忆": shared_memory
}

for name, mem in memory_spaces.items():
    print(f"\n--- {name} ---")

    matches = mem.recall(
        "量子计算 金融 写作 审核",
        limit=5,
        depth="shallow"
    )

    for item in matches:
        print(item.record.content)

print("\n========== 共享记忆结构 ==========")
print(shared_memory.tree())



