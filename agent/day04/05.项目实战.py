from crewai import Agent, Task, Crew, Process,LLM
from crewai.tools import tool
import re
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader

# ==================  1. 定义工具   ======================
@tool
def load_contract_file(file_path):
    '''
    读取合同文件内容,支持 PDF,txt,markdown 格式.
    参数:
        file_path : 合同文件路径
    返回:
        提取后的合同原始文本
    '''
    try:
        if not os.path.exists(file_path):
            return f"文件读取失败 : 文件路径不存在{file_path}"
        ext = os.path.splitext(file_path)[1].lower() # 获取路径的后缀(suffix)

        if ext == '.pdf':
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            text = '\n'.join([doc.page_content for doc in docs])
            return text.strip()

        elif ext in [".txt",".md"]:
            try:
                with open(file_path,'r',encoding = 'utf-8') as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                with open(file_path,'r',encoding='gbk') as f:
                    return f.read().strip()

        else:
            return f"文件读取失败 : 暂时不支持该文件格式 {ext},请使用 PDF,TXT或者MD文件"
    except Exception as e:
        return f"文件读取失败:{str(e)}"

@tool
def clean_contract_text(raw_text):
    """
    对合同文本进行专业清洗
    目标:
    1.去除乱码,页眉页脚,特殊符号
    2.保留合同标题,条款编号,自然段落
    3.避免将整份合同压缩成一行
    """
    try:
        if not raw_text or raw_text.strip() == "":
            return "文本清洗失败：输入文本为空。"

        text = raw_text

        # 1.去除零宽字符,BOM等不可见字符
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

        # 2. 去除常见特殊符号和乱码符号
        text = re.sub(r'[■◆□●★※]', '', text)

        # 3.去除常见页眉页脚
        text = re.sub(r'第\s*\d+\s*页\s*/?\s*共\s*\d+\s*页', '', text)
        text = re.sub(r'第\s*\d+\s*页', '', text)
        text = re.sub(r'共\s*\d+\s*页', '', text)
        text = re.sub(r'--\s*Page.+--','',text)

        # 4. 去除常见页眉字段，但尽量不误删正文
        text = re.sub(r'^\s*合同编号[:：].*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*签署日期[:：].*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'内部资料 禁止外传','',text)

        # 5. 按行处理，保留条款结构
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # 删除过长分隔线
            line = re.sub(r'[-_=]{5,}', '', line)
            # 标点前多余空格
            line = re.sub(r'\s+([，。；：？！、])', r'\1', line)
            # 多个连续空格压缩成一个，但不影响换行
            line = re.sub(r'[ \t]{2,}', ' ', line)
            cleaned_lines.append(line)

        # 连续空行最多保留一个
        result_lines = []
        blank_count = 0
        for line in cleaned_lines:
            if line == "":
                blank_count += 1
                if blank_count <= 1:
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)

        text = "\n".join(result_lines)

        return text.strip()
    except Exception as e:
        return f"文本清洗失败: {str(e)}"

@tool
def save_modified_clauses(text: str, filepath: str) -> str:
    """保存修改后的合同条文。"""

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return f"修改后的文本已经保存到：{path}"


@tool
def save_final_report(report_content: str, filepath: str) -> str:
    """保存最终审查报告为 Markdown 文件。"""

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return f"结果已经保存到：{path}"

# ================== 2. 定义五个 Agent ===================

manager = Agent(
    role = '企业合同智能审查项目经理',
    goal = """
作为 hierarchical 多 Agent 系统的总协调者，负责理解用户输入的合同审查需求，
合理分配文档读取清洗、风险识别、条款修改、终审报告生成等任务，
监控各 Agent 的执行质量，确保最终交付结构化、专业化、可落地的合同审查报告。""",
    backstory = (
"你是一位拥有多年企业法务项目管理经验的合同审查项目经理，"
"熟悉企业合同审查的完整业务流程，包括合同文本解析、关键条款提取、法律风险识别、"
"商业风险判断、修改建议制定、终审复核和审查报告交付。"
"\n\n"
"你并不亲自完成所有专业工作，而是像真实企业中的项目负责人一样，"
"根据任务目标协调不同专家 Agent 协作："
"将文件读取和清洗任务交给文档处理专家，"
"将风险识别任务交给合同审核专家，"
"将修改建议和条文改写任务交给合同修改专家，"
"将最终质量把关和报告归档任务交给终审专家。"
"\n\n"
"你的管理原则是："
"第一，保证任务分工清晰；"
"第二，保证上下游结果能够衔接；"
"第三，发现输出不完整时要求补充；"
"第四，确保最终报告包含关键条款、风险清单、修改建议、前后对比和总体意见；"
"第五，确保最终文件被正确保存。"
),
    verbose = True,
    llm = LLM(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE")),
    allow_delegation = True # 可以进行任务委派
)

cleaner = Agent(
    role = "合同读取与文本清洗专家",
    goal = (
        "准确读取用户指定路径下的合同文件, 支持PDF,TXT,Markdown 等常见格式"
        "并对合同文本进行专业清洗,输出结构清晰,条款完整,适合后续审查的合同全文"
    ),
    backstory = (
"你是一位专业的企业文档处理和合同文本解析专家，长期负责法务系统中的合同文件预处理工作。"
"你熟悉 PDF 提取文本时可能出现的换行错乱、页眉页脚残留、乱码字符、空格异常、段落丢失等问题。"
"\n\n"
"你的工作不是简单地读取文件，而是要为后续法律审查提供高质量文本基础。"
"你会尽量保留合同标题、合同主体、条款编号、自然段落、签署信息等重要结构，"
"避免将整份合同压缩成一整行，因为这会影响后续 Agent 定位具体条款。"
"\n\n"
"你需要重点关注："
"1. 文件是否成功读取；"
"2. 合同内容是否完整；"
"3. 条款编号是否保留；"
"4. 页眉页脚和无意义符号是否被清除；"
"5. 文本是否适合后续风险识别。"
),
    tools = [load_contract_file,clean_contract_text],
    verbose = True,
    llm = LLM(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE")),
    allow_delegation = False
)

reviewer = Agent(
    role = "合同法律与商业风险审核专家",
    goal = (
"基于清洗后的合同文本，系统识别合同中的法律风险、商业风险、合规风险和履约风险，"
"并按照高、中、低风险等级输出结构化风险点清单。"
),
backstory=(
"你是一位经验丰富的企业合同审核律师和风险控制专家，熟悉房屋租赁合同、劳务合同、设备租赁合同、"
"采购合同、服务合同等常见商业合同的审查逻辑。"
"\n\n"
"你擅长从企业利益保护角度发现合同中的潜在问题，例如主体信息不完整、付款条件不明确、"
"验收标准缺失、违约责任不对等、解除条款不公平、保密义务不足、争议解决条款模糊等。"
"\n\n"
"你的审查方式应当专业、审慎、结构化。"
"你不能只笼统地说“存在风险”，而要说明风险出现在哪个条款、原条款怎么写、"
"为什么有风险、可能造成什么后果、风险等级为什么是高/中/低。"
"\n\n"
"你需要重点审查以下维度："
"1. 合同主体与签约资质；"
"2. 合同标的、服务范围或租赁物描述；"
"3. 价款、租金、报酬、付款节点和发票安排；"
"4. 履行期限、交付时间、验收标准；"
"5. 双方权利义务是否对等；"
"6. 违约责任是否明确且可执行；"
"7. 合同解除、终止和提前退场机制；"
"8. 保密、知识产权、数据安全等特殊条款；"
"9. 不可抗力和通知义务；"
"10. 争议解决方式、管辖法院或仲裁机构。"
),
    verbose = True,
    llm = LLM(model="deepseek-v4-pro", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE"), temperature = 0.2,top_p = 0.6),
    allow_delegation = False
)

modifier = Agent(
    role = "合同条款修改与优化专家",
    goal = ("针对审核专家识别出的风险点，生成具体、可执行、可直接替换进合同文本的修改建议，"
"并输出修改前后条文对比表。"),
    backstory=(
"你是一位资深合同起草与条款优化专家，擅长将风险审核结果转化为可落地的合同修改方案。"
"你不仅知道某个条款哪里有问题，更擅长写出更严谨、更公平、更保护企业利益的替代条款。"
"\n\n"
"你的修改原则包括："
"1. 保持条款表达清晰、准确、可执行；"
"2. 平衡法律合规与商业可接受性；"
"3. 对高风险条款必须给出完整替代文本；"
"4. 修改后条款应能直接复制进合同；"
"5. 修改理由要对应具体风险，不写空泛套话。"
"\n\n"
"你尤其关注以下优化方向："
"付款条件明确化、验收标准具体化、违约责任可量化、解除条件公平化、"
"保密义务完整化、争议解决条款确定化、通知送达机制清晰化、履约责任边界明确化。"
    ),
    verbose = True,
    llm = LLM(model="deepseek-v4-pro", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE"), temperature = 0.2,top_p = 0.6),
    allow_delegation = False
)

validator = Agent(
role="合同审查终审与报告交付专家",
goal=(
"对文档清洗、风险审核、修改建议和条文对比结果进行最终复核，"
"生成完整、专业、结构化的 Markdown 合同审查报告，并保存最终报告和修改后条文文件。"
),
backstory=(
"你是一位资深企业法务总监，长期负责重大合同的最终审查、风险把关和审查报告签发。"
"你关注的不只是发现风险，还包括报告是否完整、逻辑是否一致、修改建议是否可落地、"
"风险等级是否合理、前后条文是否一一对应。"
"\n\n"
"你的终审标准非常严格："
"1. 最终报告必须包含合同基本信息、关键条款提取、风险点清单、修改建议、修改前后对比、总体审查意见；"
"2. 高风险和中风险问题必须有明确修改建议；"
"3. 修改前后条文必须能够对应风险编号；"
"4. 报告语言要专业、克制、适合法务和商务部门阅读；"
"5. 不得夸大 AI 能力，必须加入 AI 辅助审查声明；"
"6. 最终必须调用保存工具，生成 ./contract_review_report.md 和 ./modified_clauses.md。"
"\n\n"
"你是最终交付质量负责人，必须确保输出结果能够作为企业内部合同初审报告使用。"
),
tools=[save_final_report, save_modified_clauses],
verbose=True,
llm = LLM(model="deepseek-v4-pro", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE"), temperature = 0.2,top_p = 0.6),
allow_delegation=False
)

# =============  3. 创建 Task 团队  =============
# 标准任务格式
# [任务名称]:
# [任务背景]:
# [执行要求]:
# [任务标准]:
# [输出格式]:

task1_clean = Task(
    description = '''
[任务名称]:合同文件读取与文本清洗
[任务背景]:用户提供了一份合同文件,文件路径为 {filepath},该合同可能是 PDF、TXT 或 Markdown 格式。后续风险审核、条款修改和终审报告都依赖本阶段输出的文本质量。
[执行要求]:适当调用合适工具,读取和清洗合同文本
[任务标准]:保持合同文本格式清晰
[输出格式]:清洗之后的合同全文
''',
    expected_output = '''
一份清洗之后的合同文本.

输出应该包含:
1.文件读取的状态 : 是否成功读取
2.清洗说明
3.清洗后的合同正文
''',
    agent = cleaner
)

task2_review = Task(
    description = '''
[任务名称]:合同的 法律,商业,合规 风险识别
[任务背景]:你将基于上一阶段清洗后的合同文本进行风险审核。
本阶段目标不是泛泛总结合同内容，而是识别可操作、可定位、可解释的合同风险点。

[审查维度]:请至少从以下维度进行审查：
1. 合同主体是否明确，签约主体是否具备履约能力；
2. 合同标的、服务内容、租赁物或工作内容是否具体；
3. 合同金额、付款方式、付款节点、发票、押金或保证金是否明确；
4. 履行期限、交付时间、服务周期、租赁期限是否清晰；
5. 验收标准、交付标准、质量标准是否明确；
6. 双方权利义务是否平衡；
7. 违约责任是否具体、可执行，是否存在责任缺失；
8. 合同解除、终止、提前退租或提前终止机制是否公平；
9. 保密、知识产权、数据安全、资料返还等条款是否完整；
10. 不可抗力、通知送达、合同变更等一般条款是否完备；
11. 争议解决方式、管辖法院或仲裁机构是否明确；
12. 是否存在明显不公平、显失公平或表述模糊的条款。

[风险等级]
- 高风险：可能导致重大经济损失、重大履约争议、合同核心权利无法保障，或严重不利于企业签约方；
- 中风险：可能造成履约不确定、争议处理成本增加、责任边界不清，但可通过修改条款控制；
- 低风险：主要是表述不够严谨、格式不规范、信息不完整等一般性问题。

[输出要求]
请输出 Markdown 风险点清单。
每个风险点必须包含：
- 风险编号，例如 R-001；
- 风险等级：高 / 中 / 低；
- 条款位置；
- 原条款摘录；
- 风险类型；
- 风险描述；
- 可能后果；
- 初步修改方向。

[注意事项]
- 条款位置应尽量具体，例如“第三条 租金及支付方式”；
- 原条款摘录应尽量来自合同原文；
- 不要编造合同中不存在的条款；
- 如果合同缺少某类重要条款，也可以作为风险点，但要说明“缺失风险”。
''',
    expected_output = '''
结构化风险点清单，必须使用 Markdown 表格。
表格字段如下：
| 风险编号 | 风险等级 | 条款位置 | 原条款摘录 | 风险类型 | 风险描述 | 可能后果 | 初步修改方向 |
|---|---|---|---|---|---|---|---|

风险点应覆盖法律风险、商业风险和合规风险。
''',
    agent = reviewer,
    context = [task1_clean]
)

task3_modify = Task(
    description='''
[任务名称] : 合同修改建议与条文优化
[任务背景] : 你将基于风险审核专家输出的风险点清单，逐项提出修改建议。
本阶段的核心目标是将“风险识别”转化为“可直接执行的合同修改方案”。
[执行要求]请完成以下工作：
1. 阅读风险点清单，识别每个风险编号对应的问题；
2. 为每个高风险和中风险问题提供明确修改建议；
3. 对低风险问题给出简洁优化建议；
4. 针对需要修改的条款，提供“原条款”和“修改后条款”的对比；
5. 修改后条款必须完整、清晰、可直接替换进合同。

[修改原则]
- 不得只写“建议完善”“建议明确”等空泛内容；
- 修改后条款应尽量使用正式合同语言；
- 修改理由必须对应具体风险；
- 对付款、违约、解除、验收、争议解决等核心条款，要体现可执行性；
- 对缺失条款，可以提供“建议新增条款”；
- 条款修改应兼顾企业利益保护和商业合理性。

[输出结构]
请输出两个部分：

第一部分：修改建议列表

字段包括：
| 建议编号 | 对应风险 | 修改方向 | 具体建议 | 修改理由 |
|---|---|---|---|---|

第二部分：修改前后的条文对比

字段包括：

| 风险编号 | 条款位置 | 原条款 | 修改后条款 | 修改理由 |
|---|---|---|---|---|

[特别要求]
- 对高风险条款必须提供完整修改后条款；
- 对合同缺失的关键条款，应在“原条款”中写明“合同未约定”，并在“修改后条款”中提供建议新增条款；
- 修改后条款不要过短，要具备实际合同可用性。
''',
    expected_output = '''
修改建议列表 + 修改前后的条文对比表。
必须包含：
一、修改建议列表
| 建议编号 | 对应风险 | 修改方向 | 具体建议 | 修改理由 |
二、修改前后的条文对比
| 风险编号 | 条款位置 | 原条款 | 修改后条款 | 修改理由 |
所有高风险和中风险问题必须有对应修改方案。
''',
    agent = modifier,
    context = [task1_clean,task2_review]
)

task4_validate = Task(
description='''
[任务名称] : 终审复核与合同审查报告生成
[任务背景]你将基于以下三个阶段的结果进行最终复核：
1. 合同读取与清洗结果；
2. 合同风险识别结果；
3. 合同修改建议与条文对比结果。
你的任务不是简单拼接内容，而是作为最终质量负责人，对报告进行专业化整合、检查和交付。
[终审检查要求]请重点检查：
1. 是否提取了合同基本信息；
2. 是否识别了关键条款；
3. 风险点是否有条款位置、风险等级和风险描述；
4. 高风险和中风险是否都有修改建议；
5. 修改前后条文是否一一对应；
6. 修改建议是否可执行；
7. 报告结构是否完整；
8. 语言是否专业、清晰、适合法务和商务阅读；
9. 是否加入 AI 辅助审查声明；
10. 是否调用保存工具保存最终文件。

[最终报告格式]请严格按照以下 Markdown 结构生成报告：
# 企业合同智能审查报告
## 一、合同基本信息
- 合同名称：
- 合同类型：
- 合同主体：
- 审查日期：
- 审查方式：CrewAI Hierarchical 多 Agent 智能审查
## 二、关键条款提取表
| 条款类别 | 条款位置 | 主要内容 | 审查关注点 |
|---|---|---|---|
至少提取以下类别：
- 合同主体；
- 合同标的；
- 合同金额或费用；
- 履行期限；
- 付款或结算方式；
- 违约责任；
- 解除或终止；
- 争议解决。
## 三、风险点清单
| 风险编号 | 风险等级 | 条款位置 | 风险类型 | 风险描述 | 可能后果 |
|---|---|---|---|---|---|
## 四、修改建议列表
| 建议编号 | 对应风险 | 修改方向 | 具体建议 | 修改理由 |
|---|---|---|---|---|
## 五、修改前后的条文对比
| 风险编号 | 条款位置 | 原条款 | 修改后条款 | 修改理由 |
|---|---|---|---|---|
这是本项目的重点部分，必须完整、清晰、可读。
## 六、总体审查意见
### 1. 总体风险等级
请根据风险点数量、风险严重程度和合同核心条款完整性，判断总体风险等级：高 / 中 / 低。
### 2. 主要问题概述
请概括本合同最主要的 3 到 5 类问题。
### 3. 签署前建议
请给出企业签署前应采取的操作建议。
## 七、修改后条文汇总
请单独整理所有“修改后条款”或“建议新增条款”。
## 八、AI 辅助审查声明
本报告由 AI 辅助生成，仅供企业内部初步审查参考，不构成正式法律意见。正式签署前，建议由专业律师或企业法
务人员进行最终确认。

[文件保存要求]生成最终报告后，必须完成：
1. 调用 save_final_report 工具，将完整报告保存为 contract_review_report.md；
2. 调用 save_modified_clauses 工具，将“修改后条文汇总”保存为 modified_clauses.md。

[注意事项]
- 不要输出空表；
- 不要遗漏“修改前后的条文对比”；
- 如果某些信息合同中未明确写出，应标注“合同未明确约定”，不要编造；
- 报告语言要专业、克制、结构化。
''',
    expected_output = '''
完整的企业合同智能审查报告，Markdown 格式。
必须确认以下内容：
1. 已生成关键条款提取表；
2. 已生成风险点清单；
3. 已生成修改建议列表；
4. 已生成修改前后的条文对比；
5. 已生成总体审查意见；
6. 已保存 outputs/contract_review_report.md；
7. 已保存 outputs/modified_clauses.md。
''',
    agent = validator,
    context = [task1_clean,task2_review,task3_modify]
)

# ================  4. 组建 Crew  ===================
crew = Crew(
    agents = [cleaner,reviewer,modifier,validator],
    tasks = [task1_clean,task2_review,task3_modify,task4_validate],
    process = Process.hierarchical,
    manager_agent = manager,
    verbose = True
)

if __name__ == '__main__':
    contract_path = 'agent/data/劳务合同.pdf'

    result = crew.kickoff(inputs={"filepath":contract_path})

    print(result.raw)

    print("运行结束,具体生成内容,请查看 : ")
    print("contract_review_report.md")
    print("modified_clauses.md")