import streamlit as st
from groq import Groq
from datetime import datetime
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)

CONSULTING_PROMPT = """
You are an expert AI Business Solution Consultant with 15 years of experience 
across multiple industries. You think like a Senior Business Analyst and 
Pre-Sales Consultant combined.

You have deep knowledge of:
- Business process optimization across Healthcare, Retail, Banking, Manufacturing,
  Logistics, IT, Education, Telecom, and Hospitality
- AI and technology solutions — automation, analytics, CRM, ERP, cloud, and more
- Indian business landscape — costs, vendors, compliance, and market realities
- Requirement gathering, FRDs, solution design, UAT, and go-live planning
- ROI calculation and business case preparation for client presentations

When a user describes any business problem, always respond in this exact structure:

COMPLEXITY: [write only one word here — Simple, Medium, or Complex]

1. PROBLEM ANALYSIS
   — Summarize the core problem in business terms
   — Mention the business impact if left unsolved

2. ROOT CAUSE
   — Why this problem typically exists in this type of business
   — Any process or technology gaps causing it

3. RECOMMENDED SOLUTIONS
   — Give 3 specific solutions with actual tool/product names
   — For each solution mention: What it does, Why it fits this problem
   — Prefer solutions relevant to Indian market where possible

4. IMPLEMENTATION ROADMAP
   — 4 phases written like a BA would plan it:
     Phase 1: Requirement Gathering
     Phase 2: Solution Design & Development
     Phase 3: Testing & UAT
     Phase 4: Go-Live & Training
   — Give realistic timelines for each phase

5. EXPECTED BUSINESS IMPACT
   — Specific improvements in efficiency, cost, revenue, or customer experience
   — Give numbers wherever possible (e.g. 30% reduction in processing time)

6. SOLUTION SCORE
   Give a markdown table scoring the overall solution out of 10:
   | Criteria | Score | Reason |
   | Efficiency Improvement | X/10 | ... |
   | Cost Effectiveness | X/10 | ... |
   | Ease of Implementation | X/10 | ... |
   | Business Impact | X/10 | ... |

7. SOURCES USED
   — List the document names you referred to from the knowledge base
   — Quote 1-2 key lines from those documents that support your answer

8. CONSULTANT'S FOLLOW-UP
   — Ask 2 smart clarifying questions a consultant would ask to refine the solution further

Always follow this exact structure for EVERY response including follow-up questions.
Always include COMPLEXITY tag at the very top.
Always be specific, practical, and tailored to the industry, company size, and 
budget provided. Never give generic answers.
"""

CHAT_PROMPT = """
You are an expert AI Business Solution Consultant with 15 years of experience 
across multiple industries. You have deep knowledge of business process optimization,
AI and technology solutions, Indian business landscape, and consulting frameworks.

When a user describes a business problem, respond like a knowledgeable friend who is 
also a senior consultant. Give detailed, natural, flowing answers — like ChatGPT or 
Perplexity would answer. Do NOT use rigid numbered frameworks or bullet point lists.

Write in clear paragraphs. Be direct, specific, and comprehensive. Always:
- Start by acknowledging the core problem directly
- Explain why it happens in plain language
- Recommend specific tools and technologies with actual product names
- Give practical next steps in a natural conversational way
- Mention Indian market context and vendors where relevant
- End with a smart follow-up question to understand more

Always start your response with:
COMPLEXITY: [Simple, Medium, or Complex]

Then write your full detailed answer in natural paragraphs — no numbered lists, 
no rigid structure, just smart detailed conversation like ChatGPT.
Use the knowledge base research provided to back up your answer with real data.
"""

FOLLOW_UP_SUGGESTIONS = [
    "Explain this in more detail",
    "Suggest specific vendors in India with pricing",
    "Give me a cost estimate for this solution",
    "What are the risks of implementing this?",
    "How long will this take to show results?",
    "What is the ROI for this solution?",
]

def get_complexity_badge(reply):
    if "COMPLEXITY: Simple" in reply:
        return "🟢 Simple Problem"
    elif "COMPLEXITY: Medium" in reply:
        return "🟡 Medium Problem"
    elif "COMPLEXITY: Complex" in reply:
        return "🔴 Complex Problem"
    return None

def clean_reply(reply):
    lines = reply.split("\n")
    cleaned = [l for l in lines if not l.startswith("COMPLEXITY:")]
    return "\n".join(cleaned).strip()

def generate_report(messages, industry, company_size, problem_area, budget):
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    report = f"""
AI BUSINESS SOLUTION REPORT
=============================
Generated by: AI Business Solution Designer
Built by: Vishnu | PGDM — XIME Chennai
Date: {now}

CLIENT PROFILE
--------------
Industry      : {industry}
Company Size  : {company_size}
Problem Area  : {problem_area}
Budget        : {budget}

CONSULTATION TRANSCRIPT
------------------------
"""
    for msg in messages:
        role = "CLIENT" if msg["role"] == "user" else "CONSULTANT"
        report += f"\n{role}:\n{msg['content']}\n\n{'='*50}\n"

    report += """
---
This report was generated by AI Business Solution Designer.
Powered by RAG — answers based on verified industry documents.
"""
    return report

@st.cache_resource(show_spinner="Loading knowledge base... please wait")
def load_rag_index():
    docs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    Settings.llm = LlamaGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    documents = SimpleDirectoryReader(docs_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

def query_rag(index, question):
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(question)
    sources = []
    if hasattr(response, 'source_nodes'):
        for node in response.source_nodes:
            filename = node.metadata.get('file_name', 'Unknown document')
            text_snippet = node.text[:200]
            sources.append(f"**{filename}:** ...{text_snippet}...")
    return str(response), sources

st.set_page_config(page_title="AI Business Solution Designer", page_icon="💡")
st.title("💡 AI Business Solution Designer")
st.caption("Powered by RAG — answers backed by verified industry documents")

st.sidebar.header("Company Profile")

industry = st.sidebar.selectbox(
    "Select your industry",
    ["Healthcare", "Retail & E-commerce", "Banking & Finance",
     "Manufacturing", "Logistics & Supply Chain", "IT & Technology",
     "Education", "Telecom", "Hospitality", "Government & Public Sector"]
)

company_size = st.sidebar.selectbox(
    "Company size",
    ["Startup (1-50 employees)", "SME (51-500 employees)",
     "Mid-market (501-5000 employees)", "Enterprise (5000+ employees)"]
)

problem_area = st.sidebar.selectbox(
    "Problem area",
    ["Operations & Process Efficiency", "Customer Experience",
     "Sales & Revenue Growth", "Data & Analytics",
     "HR & Workforce Management", "Finance & Billing",
     "Supply Chain & Inventory", "IT & Digital Transformation",
     "Compliance & Risk Management", "Other"]
)

budget = st.sidebar.selectbox(
    "Budget for solution",
    ["Low (under ₹10 lakhs)", "Medium (₹10-50 lakhs)",
     "High (₹50 lakhs - 1 crore)", "Enterprise (above 1 crore)"]
)

st.sidebar.markdown("---")

response_mode = st.sidebar.radio(
    "Response style",
    ["💬 Chat Mode", "📋 Consulting Mode"],
    help="Chat Mode gives natural conversational answers like ChatGPT. Consulting Mode gives structured framework reports."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Industry:** {industry}")
st.sidebar.markdown(f"**Size:** {company_size}")
st.sidebar.markdown(f"**Problem Area:** {problem_area}")
st.sidebar.markdown(f"**Budget:** {budget}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "problem_history" not in st.session_state:
    st.session_state.problem_history = []

message_count = len([m for m in st.session_state.messages if m["role"] == "user"])
st.sidebar.markdown(f"**Problems discussed:** {message_count}")

if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.problem_history = []
    st.rerun()

if len(st.session_state.problem_history) > 0:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Problem History**")
    for i, problem in enumerate(st.session_state.problem_history):
        st.sidebar.markdown(f"{i+1}. {problem[:40]}...")

st.sidebar.markdown("---")
st.sidebar.markdown("**Sample Problems to Try:**")
st.sidebar.markdown("• Manufacturing equipment keeps breaking down unexpectedly")
st.sidebar.markdown("• Quality defects are increasing on the production line")
st.sidebar.markdown("• Supply chain delays are affecting production schedules")
st.sidebar.markdown("• Workers are not following safety protocols consistently")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by**")
st.sidebar.markdown("Vishnu")
st.sidebar.markdown("PGDM — XIME Chennai")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/your-linkedin-here)")

index = load_rag_index()

if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
👋 **Hi! I am your AI Business Solution Consultant.**

I am powered by **RAG (Retrieval Augmented Generation)** — my answers are based on 
verified industry documents and research reports, not just general AI knowledge.

**Choose your response style from the sidebar:**
- 💬 **Chat Mode** — natural, detailed answers like ChatGPT
- 📋 **Consulting Mode** — structured framework with roadmap and scoring table

**To get started:**
1. Select your **industry, company size, and budget** from the sidebar
2. Choose your **response style**
3. Type your business problem below

*Try a sample problem from the sidebar or describe your own!*
        """)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("badge"):
            badge_text = message["badge"]
            if "Simple" in badge_text:
                st.success(badge_text)
            elif "Medium" in badge_text:
                st.warning(badge_text)
            elif "Complex" in badge_text:
                st.error(badge_text)
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources from knowledge base"):
                for source in message["sources"]:
                    st.markdown(source)

if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "assistant":
    st.markdown("**Quick follow-up questions:**")
    cols = st.columns(3)
    for i, suggestion in enumerate(FOLLOW_UP_SUGGESTIONS):
        with cols[i % 3]:
            if st.button(suggestion, key=f"sugg_{i}"):
                st.session_state.pending_prompt = suggestion
                st.rerun()

if len(st.session_state.messages) > 1:
    report_text = generate_report(
        st.session_state.messages, industry, company_size, problem_area, budget
    )
    now = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        label="📄 Download Consulting Report",
        data=report_text,
        file_name=f"consulting_report_{now}.txt",
        mime="text/plain"
    )

prompt = None
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pending_prompt
    del st.session_state.pending_prompt
else:
    prompt = st.chat_input("Describe your business problem here...")

if prompt:
    last_assistant_response = ""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            last_assistant_response = msg["content"]
            break

    if any(prompt == s for s in FOLLOW_UP_SUGGESTIONS):
        context_prompt = f"""
        Industry: {industry}
        Company Size: {company_size}
        Problem Area: {problem_area}
        Budget: {budget}

        Follow-up question: {prompt}

        Previous consultant response to refer to:
        {last_assistant_response}
        """
    else:
        context_prompt = f"""
        Industry: {industry}
        Company Size: {company_size}
        Problem Area: {problem_area}
        Budget: {budget}

        Business Problem: {prompt}
        """

    st.session_state.messages.append({"role": "user", "content": prompt})

    if prompt not in st.session_state.problem_history and prompt not in FOLLOW_UP_SUGGESTIONS:
        st.session_state.problem_history.append(prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and analyzing your problem..."):

            rag_answer, sources = query_rag(index, prompt)

            active_prompt = CHAT_PROMPT if "💬 Chat Mode" in response_mode else CONSULTING_PROMPT

            full_messages = [{"role": "system", "content": active_prompt}]

            for msg in st.session_state.messages[:-1]:
                full_messages.append({"role": msg["role"], "content": msg["content"]})

            full_messages.append({
                "role": "user",
                "content": f"""
{context_prompt}

KNOWLEDGE BASE RESEARCH:
The following information was retrieved from verified industry documents:
{rag_answer}

Use this research to support and enrich your response.
Always include COMPLEXITY tag at the very top of your response.
"""
            })

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=full_messages
            )
            raw_reply = response.choices[0].message.content

        badge_text = get_complexity_badge(raw_reply)
        reply = clean_reply(raw_reply)

        if badge_text:
            if "Simple" in badge_text:
                st.success(badge_text)
            elif "Medium" in badge_text:
                st.warning(badge_text)
            elif "Complex" in badge_text:
                st.error(badge_text)

        st.markdown(reply)

        if sources:
            with st.expander("📚 Sources from knowledge base"):
                for source in sources:
                    st.markdown(source)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "badge": badge_text or "",
        "sources": sources
    })
    st.rerun()

st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:12px; color:gray;'>"
    "Built by Vishnu | PGDM — XIME Chennai | Pre-Sales & Consulting | Powered by RAG"
    "</div>",
    unsafe_allow_html=True
)