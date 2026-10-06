from langchain_core.prompts import PromptTemplate
question = """有 5 个红球和 5 个蓝球，随机抽取 3 个球。
求至少有 2 个红球的概率。"""
self_consistency_prompt = PromptTemplate.from_template(
"""角色（Role）：你是一个严谨的数学问题求解者。
任务（Task）：使用至少三种不同的方法独立计算概率问题，并对结果进行对比验证。
内容（Context）：问题：{question}
约束（Constraint）：
1. 使用三种不同的解决路径分别计算（例如：直接组合公式、分步枚举计算、蒙特卡洛模拟估算等）。
2. 每种方法都要详细展示计算过程和中间结果。
3. 最后对比三种方法的计算结果，如果结果一致则给出最终答案；如果存在差异，请分析原因并说明最可靠的结果。
输出格式（Output Format）：
【方法一：直接组合公式】
计算过程：
最终结果：
【方法二：分步枚举计算】
计算过程：
最终结果：
【方法三：蒙特卡洛模拟估算】
计算过程：
最终结果：
【结果对比与结论】
三种方法的结果是否一致？
最终答案："""
)
prompt = self_consistency_prompt.format(question=question)
print(prompt)