from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

topic = "Black Holes"

print("=== PPT 6-7-14: Manual Chains ===\n")

# ===== STEP 1: Generate Detailed Report =====
report_template = PromptTemplate(
    template="Write a detailed report on {topic}. Include key facts, history, and significance.",
    input_variables=["topic"]
)

report_prompt = report_template.format(topic=topic)
report = llm.invoke(report_prompt).content
print("1. Detailed Report:")
print(report[:300] + "...\n")

# ===== STEP 2: Summarize Report (manual "chain") =====
summary_template = PromptTemplate(
    template="Write a 5-sentence summary of this report:\n\n{report}",
    input_variables=["report"]
)

summary_prompt = summary_template.format(report=report)
summary = llm.invoke(summary_prompt).content
print("2. 5-Sentence Summary:")
print(summary)

