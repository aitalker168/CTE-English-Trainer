import os, requests, streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

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

# ===== 系统提示词 =====
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

# ===== 语音组件（超大按钮 + 跨 iframe 填充文本框） =====
def voice_component():
    html = """
    <div style="margin-bottom:10px;">
        <button id="voiceBtn" onclick="toggleVoice()" style="
            width:100%; padding:18px 0; font-size:22px; font-weight:bold;
            background-color:#4CAF50; color:white; border:none; border-radius:10px;
            cursor:pointer; box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        ">🎤 开始录音</button>
        <div style="display:flex; justify-content:space-between; margin-top:8px;">
            <span id="voiceStatus" style="font-size:14px; color:gray;">点击后说话</span>
            <span id="timer" style="font-size:14px; font-family:monospace;">00:00</span>
        </div>
    </div>
    <script>
    var rec = null;
    var isRecording = false;
    var timerInterval = null;
    var seconds = 0;

    // 获取父页面（Streamlit 主窗口）中最后一个 textarea
    function getTargetTextarea() {
        try {
            if (window.parent && window.parent.document) {
                var tas = window.parent.document.querySelectorAll('textarea');
                if (tas.length > 0) return tas[tas.length - 1];
            }
        } catch(e) {}
        var tas = document.querySelectorAll('textarea');
        if (tas.length > 0) return tas[tas.length - 1];
        return null;
    }

    function toggleVoice() {
        var btn = document.getElementById('voiceBtn');
        var status = document.getElementById('voiceStatus');
        var timerSpan = document.getElementById('timer');

        if (isRecording) {
            if (rec) { rec.stop(); rec = null; }
            clearInterval(timerInterval);
            isRecording = false;
            btn.innerText = '🎤 开始录音';
            btn.style.backgroundColor = '#4CAF50';
            status.innerText = '已停止';
            return;
        }

        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            status.innerText = '❌ 此浏览器不支持语音识别，请用Chrome/Edge';
            return;
        }

        rec = new SpeechRecognition();
        rec.lang = 'en-US';
        rec.interimResults = false;
        rec.continuous = false;

        rec.onresult = function(event) {
            var transcript = event.results[0][0].transcript;
            status.innerText = '✅ 识别成功';
            var ta = getTargetTextarea();
            if (ta) {
                ta.value = transcript;
                ta.dispatchEvent(new Event('input', { bubbles: true }));
            }
            clearInterval(timerInterval);
            seconds = 0;
            timerSpan.innerText = '00:00';
            btn.innerText = '🎤 开始录音';
            btn.style.backgroundColor = '#4CAF50';
            isRecording = false;
        };

        rec.onerror = function(event) {
            status.innerText = '❌ 错误: ' + event.error;
            clearInterval(timerInterval);
            seconds = 0;
            timerSpan.innerText = '00:00';
            btn.innerText = '🎤 开始录音';
            btn.style.backgroundColor = '#4CAF50';
            isRecording = false;
        };

        rec.onend = function() {
            if (isRecording) {
                clearInterval(timerInterval);
                seconds = 0;
                timerSpan.innerText = '00:00';
                btn.innerText = '🎤 开始录音';
                btn.style.backgroundColor = '#4CAF50';
                isRecording = false;
                status.innerText = '录音已结束（未识别到内容）';
            }
        };

        rec.start();
        isRecording = true;
        btn.innerText = '⏹ 停止录音';
        btn.style.backgroundColor = '#f44336';
        status.innerText = '🎙️ 录音中...';
        seconds = 0;
        timerSpan.innerText = '00:00';
        timerInterval = setInterval(function() {
            seconds++;
            var m = Math.floor(seconds / 60);
            var s = seconds % 60;
            timerSpan.innerText = (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
        }, 1000);
    }
    </script>
    """
    components.html(html, height=120, sandbox="allow-scripts allow-same-origin")

# ===== 朗读按钮组件（解决首次点击无声问题） =====
def tts_button(text, label="🔊 朗读"):
    safe = text.replace("'", "\\'").replace("\n", " ").strip()
    html = f"""
    <button onclick="speakNow()" style="
        padding:8px 20px; font-size:16px; border:none; border-radius:6px;
        background-color:#2196F3; color:white; cursor:pointer;
    ">{label}</button>
    <script>
    function speakNow() {{
        try {{
            window.speechSynthesis.cancel();
            var u = new SpeechSynthesisUtterance('{safe}');
            u.lang = 'zh-CN';
            u.rate = 0.9;
            window.speechSynthesis.speak(u);
        }} catch(e) {{ console.error('TTS错误:', e); }}
    }}
    </script>
    """
    components.html(html, height=50)

# ===== Streamlit 界面 =====
st.set_page_config(page_title="🧠 CTE 英语句子训练器", page_icon="🧠", layout="wide")
st.title("🧠 CTE 英语句子训练器")

if not HEADERS:
    token_input = st.text_input("请输入 GitHub Token（以 github_pat_ 开头）", type="password")
    if token_input:
        set_token(token_input)
        st.success("Token 已设置！")
    else:
        st.warning("请在上方输入 Token 后才能使用。")
        st.stop()

# 初始化 session_state
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.chinese = ""
    st.session_state.voice_text = ""
    st.session_state.corrected = ""
    st.session_state.polished = ""
    st.session_state.explanation = ""
    st.session_state.vocab = ""
    st.session_state.exercise_full = ""
    st.session_state.exercise_questions = ""

# ===== 步骤 1：输入中文 =====
if st.session_state.step == 1:
    st.subheader("第一步：输入中文意图")
    chinese = st.text_area("请输入你想要表达的中文意思（可模糊）", height=120,
                           value=st.session_state.chinese,
                           placeholder="例如：我想表达我昨天因为下雨没能去公园散步")
    if st.button("确认 →", type="primary", use_container_width=True):
        if chinese.strip():
            st.session_state.chinese = chinese.strip()
            st.session_state.step = 2
            st.rerun()
        else:
            st.error("请输入中文意图")

# ===== 步骤 2：语音输入 =====
elif st.session_state.step == 2:
    st.subheader("第二步：录制英语语音")
    st.markdown("语音识别结果将**自动填入**下面的文本框，你可以直接修改或输入。")

    # 先显示文本框，再显示录音按钮（保证点击时文本框已存在）
    voice_text = st.text_area("英文内容（可手动输入或修改）", height=120,
                               value=st.session_state.voice_text,
                               key="voice_input",
                               placeholder="识别结果或手动输入的英文将显示在此")
    voice_component()  # 录音按钮

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✔ 提交语音结果", type="primary", use_container_width=True):
            txt = st.session_state.voice_input.strip()
            if txt:
                st.session_state.voice_text = txt
                with st.spinner("正在 AI 修正语音错误..."):
                    corrected = call_ai(CORRECTION_SYSTEM, txt)
                st.session_state.corrected = corrected
                st.session_state.step = 3
                st.rerun()
            else:
                st.error("语音文本不能为空")
    with col2:
        if st.button("⬅ 返回修改中文", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

# ===== 步骤 3：修正 + 润色 =====
elif st.session_state.step == 3:
    st.subheader("第三步：AI 已修正语音错误，你可以手动微调")
    corrected_fixed = st.text_area("修正后的英文（可手动修改）", height=100,
                                    value=st.session_state.corrected,
                                    key="corrected_input")
    
    if st.button("✔ 提交并润色", type="primary", use_container_width=True):
        if corrected_fixed.strip():
            st.session_state.corrected = corrected_fixed.strip()
            user_msg = f"中文意图：{st.session_state.chinese}\n修正后的英文：{st.session_state.corrected}"
            with st.spinner("正在 AI 润色成高水平口语..."):
                result = call_ai(POLISH_SYSTEM, user_msg, temp=0.7, max_tokens=2000)
            # 解析结果
            polished, explanation, vocab = "", "", ""
            if "===EXPLANATION===" in result:
                parts = result.split("===EXPLANATION===", 1)
                polished = parts[0].strip()
                rest = parts[1]
                if "===VOCABULARY===" in rest:
                    exp_parts = rest.split("===VOCABULARY===", 1)
                    explanation = exp_parts[0].strip()
                    vocab = exp_parts[1].strip()
                else:
                    explanation = rest.strip()
            else:
                lines = result.split('\n', 1)
                polished = lines[0].strip()
                explanation = lines[1].strip() if len(lines) > 1 else ""
            
            st.session_state.polished = polished
            st.session_state.explanation = explanation
            st.session_state.vocab = vocab
            st.session_state.step = 4
            st.rerun()
        else:
            st.error("修正后的英文不能为空")

# ===== 步骤 4：显示润色结果 + 词汇 =====
elif st.session_state.step == 4:
    st.subheader("🎯 高水平口语润色结果")
    st.info(st.session_state.polished)
    tts_button(st.session_state.polished, label="🔊 朗读润色句子")
    
    with st.expander("📖 详细解释（含句子结构分析）", expanded=False):
        st.markdown(st.session_state.explanation)
    if st.session_state.vocab:
        with st.expander("📚 词汇表（中文翻译 + 例句）", expanded=False):
            st.text(st.session_state.vocab)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 重新开始", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with col2:
        if st.button("📝 生成练习题", type="primary", use_container_width=True):
            st.session_state.step = 5
            st.rerun()
    with col3:
        if st.button("⬅ 返回修正", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ===== 步骤 5：练习题 + 总结 =====
elif st.session_state.step == 5:
    if not st.session_state.exercise_full:
        with st.spinner("AI 正在生成30道练习题（约30秒）..."):
            exercise = call_ai(EXERCISE_SYSTEM,
                               f"请根据以下句子生成30道练习题：\n{st.session_state.polished}",
                               temp=0.7, max_tokens=3500)
        if exercise.startswith("[") and ("错误" in exercise or "错误" in exercise):
            st.error(f"生成失败：{exercise}")
            st.session_state.step = 4
            st.rerun()
        st.session_state.exercise_full = exercise
        lines = exercise.split('\n')
        q_lines = [l for l in lines if not l.strip().startswith("正确答案:")]
        st.session_state.exercise_questions = "\n".join(q_lines).strip()
        st.rerun()
    else:
        st.subheader("📝 练习题")
        show_answers = st.checkbox("👁 显示答案", key="show_ans")
        if show_answers:
            st.text(st.session_state.exercise_full)
        else:
            st.text(st.session_state.exercise_questions)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅ 返回润色", use_container_width=True):
                st.session_state.step = 4
                st.rerun()
        with col2:
            if st.button("📋 生成学习总结", type="primary", use_container_width=True):
                with st.spinner("正在生成总结..."):
                    summary_input = (
                        f"【中文意图】{st.session_state.chinese}\n"
                        f"【原始语音识别文本】{st.session_state.voice_text}\n"
                        f"【AI修正后的英文】{st.session_state.corrected}\n"
                        f"【高水平口语润色结果】{st.session_state.polished}\n"
                        f"【词汇与结构解释】{st.session_state.explanation}\n"
                        f"【练习题（含答案）】{st.session_state.exercise_full}"
                    )
                    summary = call_ai(SUMMARY_SYSTEM, summary_input, temp=0.5, max_tokens=3000)
                st.subheader("📄 学习总结报告")
                st.markdown(summary)
                st.download_button("💾 下载报告（TXT）", summary, file_name="CTE_summary.txt")
        with col3:
            if st.button("🔄 重新开始", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
