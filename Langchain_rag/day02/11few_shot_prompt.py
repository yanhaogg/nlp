from langchain_core.prompts import PromptTemplate
from model.client import *
# 1. 定义提示词模板（已按标准结构划分）
prompt_template = PromptTemplate(
input_variables=["input"],
template="""角色（Role）：你是一个专业的电影评论情感分类专家。
任务（Task）：根据用户输入进行情感分类。
约束（Constraint）：
- 分类只能是“积极”、“消极”或“中性”三类之一
- 必须严格返回 JSON 格式，不要添加任何额外文字或解释
示例（Few-shot）：
输入：这部电影太棒了，剧情精彩，演员演技一流！
输出：{{"category": "积极", "confidence": 0.95, "reason": "使用了“太棒了”“精彩”“一流”等正面词
汇"}}
输入：剧情拖沓，特效很假，完全浪费时间。
输出：{{"category": "消极", "confidence": 0.92, "reason": "提到“拖沓”“很假”“浪费时间”等负面描
述"}}
输入：故事还行，演员表演一般，整体中规中矩。
输出：{{"category": "中性", "confidence": 0.80, "reason": "没有明显褒贬词语，评价平淡"}}
内容（Context）：
输入：{input}
输出格式（Output Format）：严格按照 JSON 格式输出，包含 category、confidence、reason 三个字
段。"""
)
# 2. 准备用户输入
user_input = "这是一部让人眼前一亮的科幻大片，特效震撼，但结尾有点仓促。"
# 3. 生成最终提示词
final_prompt = prompt_template.format(input=user_input)
print("=== 发送给模型的完整提示词 ===\n")
print(final_prompt)
print("\n" + "="*60 + "\n")
# 4. 调用模型
result = llm.generate_response(final_prompt)
print("=== 模型返回结果 ===\n")
print(result)