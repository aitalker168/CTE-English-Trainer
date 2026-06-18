import os, requests, streamlit as st
import io
import speech_recognition as sr
from pathlib import Path

# ===== Token 设置 =====
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
HEADERS = None

def set_token(token: str):
    global HEADERS
    HEADERS = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

if GITHUB_TOKEN:
    set_token(GITHUB_TOKEN)

# ===== API 调用 =====
API_URL = "https://models.inference.ai.azure.com/chat/completions"

def call_ai(system: str, user: str, temp: float = 0.5, max_tokens=2000, timeout=120) -> str:
    if not HEADERS:
        return "[错误: 未设置 API Token]"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": temp,
        "max_tokens": max_tokens
    }
    try:
        resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"[API 错误 {resp.status_code}]"
    except Exception as e:
        return f"[网络错误] {e}"

# ===== 语音识别函数（从字节流识别） =====
def recognize_audio_bytes(audio_bytes):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language="en-US")
        return text
    except sr.UnknownValueError:
        return None  # 未识别出语音
    except Exception as e:
        return f"[识别错误] {e}"

# ===== 系统提示词（和你原来完全一致） =====
CORRECTION_SYSTEM = (
    "你是一个英文听写助手。用户通过语音输入了一段英文，可能包含因为发音不标准导致的"
    "拼写错误或用词不当。请纠正这些错误，输出语法正确、拼写正确的英文句子。"
    "保留用户原本想表达的意思，不要添加额外内容。"
)

POLISH_SYSTEM = (
    "你是一个英语口语教练。用户提供了一个中文意图和一段初步修正的英文。\n"
    "请按以下要求严格输出：\n\n"
    "第一步：将修正后的英文润色为高水平、地道、自然的口语版英语（仍保持口语风格，但用词和句式更高级、更准确）。只输出润色后的句子，不要任何额外文字。\n\n"
    "第二步：在润色句子之后，立即换行，然后输出以下分隔符：\n"
    "===EXPLANATION===\n"
    "然后换行，再按照以下三个序列号输出详细的解释：\n"
    "序列号1: 详细说明你如何从用户输入的原始英文思考并生成这个高水平句子的过程。特别说明你增加了哪些形容词、副词、动词以及为什么增加。\n"
    "序列号2: 列出润色后句子中所有的动词、形容词、副词，并逐个解释为什么选用这个词。\n"
    "序列号3: 分析润色后句子的语法架构（主句、从句、连接词、语序等），并说明为什么采用这个架构。\n\n"
    "第三步：在第二步的解释全部结束后，换行，然后输出以下分隔符：\n"
    "===VOCABULARY===\n"
    "然后换行，逐行输出润色后句子中包含的所有 动词、形容词、副词、名词 的词汇表。\n"
    "每行格式严格如下，一个词一行：\n"
    "词 | 词性（verb/adj/adv/noun） | 中文翻译 | 典型例句 | 例句中文翻译\n"
    "注意：\n"
    "- 典型例句应选用该词在润色句子中的用法（或一个同样自然的替代例句），例句及其中文翻译必须与润色句子上下文相关。\n"
    "- 词性用小写英文标注。\n"
    "- 请确保覆盖所有重要的动词、形容词、副词、名词，不要遗漏。\n"
    "请务必先输出润色句子，然后换行，再输出 ===EXPLANATION===，然后换行输出序列号1-3，最后换行输出 ===VOCABULARY=== 和词汇表。"
)

EXERCISE_SYSTEM = (
    "你是一个英语练习题生成专家。用户将提供一句高水平的口语英文句子。\n"
    "请你根据这个句子生成30道练习题，目的是帮助用户加深对句子中重要形容词、副词、动词的记忆和理解。\n"
    "题目必须混合包含以下三种题型：选择题（至少10道）、填空题（至少10道）、判断题（至少10道）。\n"
    "每道题必须直接与句子中的关键词汇（形容词、副词、动词）相关。\n\n"
    "输出格式要求：\n"
    "每道题输出一个节，用空行隔开。格式如下：\n"
    "题号. 题目内容\n"
    "（如果是选择题，列出A. B. C. D. 选项）\n"
    "正确答案: (内容)\n\n"
    "请严格按照此格式输出。务必包含30道题。"
)

SUMMARY_SYSTEM = (
    "你是一位资深英语学习总结助理。用户完成了一轮完整的CTE（主句+2从句）英语句子训练，包括："
    "中文意图输入、语音英文识别、AI修正、高级口语润色、词汇解释、30道练习题。"
    "现在需要你根据以下提供的全部内容，生成一份结构清晰、便于复习的学习总结报告。\n"
    "报告必须包含以下章节（请使用中文写报告，但保留所有英文原文）：\n"
    "1. 学习目标（用户想表达的中文意思）\n"
    "2. 原始语音识别文本（用户说的英文，未经修正）\n"
    "3. AI修正后的英文（修正拼写/语法后的句子）\n"
    "4. 高水平口语润色结果（最终润色句子）\n"
    "5. 重要词汇与用法解析（从解释中提取关键形容词、副词、动词及其作用）\n"
    "6. 句子结构分析（主句、从句、连接词等）\n"
    "7. 错题重点回顾（列出练习题中用户可能易错的题目类型与知识点）\n"
    "8. 整体学习建议（基于本句练习，给出一到两条针对性的建议）\n\n"
    "请将以上所有信息整合成一份阅读体验良好的总结报告，直接输出，不要额外说明。"
)

# ===== Streamlit 界面 =====
st.set_page_config(page_title="🧠 CTE 英语句子训练器", page_icon="🧠", layout="wide")

st.title("🧠 CTE 英语句子训练器")
st.markdown("**步骤：** 输入中文意图 → 录音并识别英文 → AI 修正 → 润色与词汇 → 练习题 + 总结")

# Token 输入
if not HEADERS:
    token_input = st.text_input("请输入 GitHub Token（以 github_pat_ 开头）", type="password")
    if token_input:
        set_token(token_input)
        st.success("Token 已设置！")
    else:
        st.warning("请在上方输入 Token 后才能使用。")
        st.stop()

# 初始化 session_state
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""
if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False

with st.form("main_form"):
    col1, col2 = st.columns(2)
    with col1:
        chinese = st.text_area("第一步：输入中文意思（可模糊）", height=150,
                               placeholder="例如：我想表达我昨天因为下雨没能去公园散步")
    with col2:
        st.markdown("**第二步：录制英语语音**")
        # 录音组件
        audio_bytes = st.audio_input("🎤 点击开始录音（手机长按图标即可说话）")
        if audio_bytes is not None:
            # 如果有新录音且尚未处理
            if not st.session_state.audio_processed:
                with st.spinner("正在识别语音..."):
                    result = recognize_audio_bytes(audio_bytes.getvalue())
                if result is None:
                    st.warning("未识别到语音，请重录或手动输入")
                    st.session_state.voice_text = ""
                elif result.startswith("[识别错误]"):
                    st.error(result)
                    st.session_state.voice_text = ""
                else:
                    st.session_state.voice_text = result
                    st.success(f"识别结果：{result}")
                st.session_state.audio_processed = True
                st.rerun()  # 刷新以显示文本框内容
        else:
            st.session_state.audio_processed = False

        # 文本框：显示识别结果，可手动修改
        voice_input = st.text_area("或手动输入/修改英文", value=st.session_state.voice_text,
                                   height=100, placeholder="识别结果将自动填入")
        st.caption("* 录制后自动识别填入，你也可以直接打字。")

    col_a, col_b = st.columns(2)
    with col_a:
        submitted = st.form_submit_button("🚀 开始处理", use_container_width=True)
    with col_b:
        restarted = st.form_submit_button("🔄 重置", use_container_width=True)

if submitted:
    if not chinese or not voice_input.strip():
        st.error("请同时填写中文意图和英文文本！")
    else:
        # Step2: 修正
        with st.spinner("正在 AI 修正语音错误..."):
            corrected = call_ai(CORRECTION_SYSTEM, voice_input.strip())
        st.subheader("修正后的英文")
        corrected_fixed = st.text_area("你可以手动微调（直接修改）", corrected, height=80)

        if corrected_fixed:
            with st.spinner("正在 AI 润色成高水平口语..."):
                user_msg = f"中文意图：{chinese}\n修正后的英文：{corrected_fixed}"
                result = call_ai(POLISH_SYSTEM, user_msg, temp=0.7, max_tokens=2000)

            # 解析
            polished, explanation, vocab = "", "", ""
            if "===EXPLANATION===" in result:
                parts = result.split("===EXPLANATION===", 1)
                polished = parts[0].strip()
                rest = parts[1]
                if "===VOCABULARY===" in rest:
                    expl_parts = rest.split("===VOCABULARY===", 1)
                    explanation = expl_parts[0].strip()
                    vocab = expl_parts[1].strip()
                else:
                    explanation = rest.strip()
            else:
                lines = result.split('\n', 1)
                polished = lines[0].strip()
                explanation = lines[1].strip() if len(lines) > 1 else ""

            st.success("润色完成！")
            st.subheader("🎯 高水平口语润色结果")
            st.info(polished)

            with st.expander("📖 详细解释（含句子结构分析）", expanded=False):
                st.markdown(explanation)
            if vocab:
                with st.expander("📚 词汇表（中文翻译 + 例句）", expanded=False):
                    st.text(vocab)

            # 练习题按钮
            if st.button("📝 生成 30 道练习题", type="primary"):
                with st.spinner("AI 正在生成练习题（约30秒）..."):
                    exercise_full = call_ai(EXERCISE_SYSTEM,
                                            f"请根据以下句子生成30道练习题：\n{polished}",
                                            temp=0.7, max_tokens=3500)
                st.subheader("练习题库")
                show_answer = st.checkbox("👁 显示答案")
                if show_answer:
                    st.text(exercise_full)
                else:
                    lines = exercise_full.split('\n')
                    show_lines = [l for l in lines if not l.strip().startswith("正确答案:")]
                    st.text('\n'.join(show_lines).strip())

                # 总结按钮
                if st.button("📋 生成学习总结报告"):
                    with st.spinner("正在生成总结..."):
                        summary_input = (
                            f"【中文意图】{chinese}\n"
                            f"【原始语音识别文本】{voice_input.strip()}\n"
                            f"【AI修正后的英文】{corrected_fixed}\n"
                            f"【高水平口语润色结果】{polished}\n"
                            f"【词汇与结构解释】{explanation}\n"
                            f"【练习题（含答案）】{exercise_full}"
                        )
                        summary = call_ai(SUMMARY_SYSTEM, summary_input, temp=0.5, max_tokens=3000)
                    st.subheader("📄 学习总结报告")
                    st.markdown(summary)
                    st.download_button("💾 下载报告（TXT）", summary, file_name="CTE_summary.txt")
        else:
            st.warning("修正后的英文不能为空。")

# 重置按钮处理
if restarted:
    for key in ["voice_text", "audio_processed"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
