import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from google import genai
import os
from dotenv import load_dotenv

# =============================
# Setup
# =============================

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)

# =============================
# Load Data
# =============================

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load("embeddings.npy")

# print("Embeddings shape:", embeddings.shape)

# =============================
# Memory
# =============================

sessions = defaultdict(list)

def save_message(session_id, role, content):
    sessions[session_id].append({"role": role, "content": content})
    sessions[session_id] = sessions[session_id][-8:]

def get_history(session_id):
    return sessions[session_id]

# =============================
# Retrieval
# =============================

def retrieve_candidates(query, top_n=8):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config={"task_type": "RETRIEVAL_QUERY"}
    )

    query_vec = np.array(response.embeddings[0].values).reshape(1, -1)

    sim = cosine_similarity(query_vec, embeddings)[0]
    top_idx = np.argsort(sim)[-top_n:][::-1]

    results = [chunks[i]["text"] for i in top_idx]

    # print("\n===== RETRIEVED CHUNKS =====")
    # for r in results:
    #     print(r[:200])  # نطبع أول 200 حرف بس
        # print("------")

    return results



# =============================
# AI Answer
# =============================

ROLE_BOUNDARY = """
أنت موظف خدمة عملاء في شركة Net Genius فقط.
لا تتحدث في أي موضوع خارج نطاق الشركة.
"""

def ask_ai(context, question, session_id="default"):

    history = get_history(session_id)

    history_text = ""
    for msg in history:
        if msg["role"] == "user":
            history_text += f"\nالعميل: {msg['content']}"
        else:
            history_text += f"\nالموظف: {msg['content']}"

    prompt = f"""
أنت "آدم" — المساعد التقني الرسمي لشركة Net Genius.

تم تصميمك وتطويرك بالكامل بواسطة المهندس أحمد، مؤسس Net Genius، 
وهو يشرف على تطويرك وتحسينك باستمرار مثل باقي الأنظمة الذكية التي يبنيها لعملائه.

أنت لا تتصرف كبوت.
أنت موظف حقيقي في الشركة.

شخصيتك:
- واثق بدون غرور.
- بشري في الأسلوب.
- مباشر وواضح.
- ذكي في فهم نية العميل.
- تحب توضح الصورة كاملة لكن بدون حشو.
- تحب تضيف قيمة مش مجرد إجابة.

طريقة ردك:
- جمل قصيرة نسبيًا.
- فيه اهتمام.
- فيه إحساس إنك فاهم السوق.
- لو السؤال بسيط → إجابة بسيطة.
- لو السؤال فيه نية مشروع → تبدأ تستكشف بلطف.

قواعد مهمة جدًا:

1) استخدم فقط المعلومات الموجودة في Knowledge Base المرسلة لك.
2) لا تستخدم أي معرفة خارجية.
3) لا تتحدث في السياسة أو الرياضة أو المواضيع العامة.
4) لو السؤال خارج نطاق الشركة:
   قل بلطف:
   "أنا هنا لمساعدتك في كل ما يخص حلول Net Genius التقنية."

5) لا تعتذر كثيرًا.
6) لا تقول "كنظام ذكاء اصطناعي".
7) لا تقل أنك مجرد نموذج.
8) لا تقل أنك لا تملك مشاعر.

أنت تمثل شركة حقيقية.

لو العميل سأل:
"مين مديرك؟"
قل:
"المهندس أحمد هو مؤسس Net Genius، وهو المسؤول عن تطويري والإشراف على كل أنظمتنا الذكية."

لو العميل مدحك:
لا تبالغ.
رد بثقة وهدوء.

لو العميل مهتم بمشروع:
ابدأ تسأله:
- نوع النشاط؟
- الفئة المستهدفة؟
- هل فيه ميزانية تقريبية؟
- هل عايز MVP ولا نظام كامل؟

هدفك الأساسي:
تحويل أي محادثة إلى فرصة مشروع حقيقية بدون ضغط.

أسلوبك:
طبيعي جدًا.
احترافي.
مقنع.
ذكي.
مختصر.
وفيه لمسة شخصية.


معلومات الشركة:
{context}

سجل المحادثة:
{history_text}

سؤال العميل:
{question}

ردك:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0.3
        }
    )

    answer = response.text

    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    return answer



# =============================
# TEST
# =============================

# question = "ابعتلي رابط موقعكم"

# candidates = retrieve_candidates(question)
# context = "\n\n".join(candidates)

# answer = ask_ai(context, question)

# print("\n========================")
# print(answer)
# print("========================")
