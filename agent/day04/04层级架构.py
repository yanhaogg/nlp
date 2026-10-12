from crewai import Agent,Crew,Process

manager = Agent(
    role = '企业合同智能审查项目经理',
    goal = """
作为 hierarchical 多 Agent 系统的总协调者，负责理解用户输入的合同审查需求，
合理分配文档读取清洗、风险识别、条款修改、终审报告生成等任务，
监控各 Agent 的执行质量，确保最终交付结构化、专业化、可落地的合同审查报告。""",
    backstory = """
你是一位拥有多年企业法务项目管理经验的合同审查项目经理，熟悉企业合同审查的完整业务流程，包括合同文本解析、关键条款提取、法律风险识别、商业风险判断、修改建议制定、终审复核和审查报告交付。""",
    verbose = True,
    allow_delegation = True # 可以进行任务委派
)

crew = Crew(
    agents = [], # manager不是组员,所以不能放在该列表里面
    process = Process.hierarchical, # 把 sequencial(顺序执行)  改为hierarchical 层级架构
    manager_agent = manager # 有manager专属的参数位置
)