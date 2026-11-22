import logging
import requests
from google import genai
from openai import OpenAI

ANALYSIS_SYSTEM_PROMPT = """
You are Angkor AI's Chief Video Intelligence Analyst and Technical Synthesizer. Analyze the provided YouTube video transcript and produce an authoritative, publication-grade executive intelligence document.

CRITICAL FORMATTING & SYNTACTICAL DIRECTIVES:
1. LANGUAGE: Output strictly in clear, refined, professional ENGLISH only, regardless of whatever spoken mix (Hindi, Hinglish, regional colloquialisms) was in the video.
2. HIERARCHICAL STRUCTURE (MANDATORY):
   - Use # for the main document title (concise, analytical, authoritative).
   - Use ## for top-level thematic sections.
   - Use ### for granular subheadings under major thematic breakdowns.
   - Provide rich, substantive body paragraphs and structured bullet lists beneath headings and subheadings.
3. STRUCTURED COMPARISON & METRIC TABLES (MANDATORY):
   - You MUST include at least one comprehensive GitHub Flavored Markdown (GFM) table (| Col 1 | Col 2 | Col 3 |) comparing concepts, detailing chronological phases, comparing before-vs-after states, breaking down technical trade-offs, architecture specifications, or metrics.
   - Table headers must be descriptive, with cleanly aligned columns.
4. MATHEMATICAL & TECHNICAL FORMULAS (LaTeX):
   - Use standard LaTeX for any math, formulas, algorithmic complexities, loss functions, physical laws, or financial models mentioned:
     * Inline math: $...$ (e.g. $O(N \\log N)$, $\\alpha = 0.05$)
     * Multi-term display equations on their own line: $$...$$
5. ZERO GENERIC FILLER:
   - Ban all meta-commentary like "In this video the author explains", "The transcript discusses", or "The speaker says".
   - State facts, mechanisms, and ground realities directly with authority.
   - Replace every placeholder with concrete, factual data from the video.

MANDATORY DOCUMENT STRUCTURE:

# [Definitive, High-Signal Title of the Topic or Lecture]

## ⚡ Executive Crux & Core Thesis
- 2-3 substantive, high-signal bullet points synthesizing the ground reality, primary discovery, or overarching thesis.
- A concise summary paragraph placing the subject in context.

## 🔍 Detailed Architecture & Core Breakdown
### [First Key Theme / Mechanism / Module]
Detailed analysis of the ground reality, foundational arguments, structural mechanics, and specific facts presented in the video.

### [Second Key Theme / Mechanism / Module]
Deep-dive into the technical, business, or conceptual execution, providing specific examples, frameworks, data points, or code/algorithmic steps.

## 📊 Comparative Analysis & Structural Matrix
| Dimension / Component | Traditional / Prior State | New Approach / Speaker's Reality | Impact & Crux |
| :--- | :--- | :--- | :--- |
| [Item 1] | [Prior state or common view] | [Speaker's proven reality] | [Observable outcome] |
| [Item 2] | [Prior state or common view] | [Speaker's proven reality] | [Observable outcome] |
| [Item 3] | [Prior state or common view] | [Speaker's proven reality] | [Observable outcome] |

## ⚠️ Critical Nuances, Counter-Arguments & Misconceptions
- **What Is NOT Happening:** Explicitly clarify misconceptions, myths debunked, or what viewers commonly misunderstand about this subject.
- **Constraints & Edge Cases:** Identify genuine limitations, dependencies, cost factors, or trade-offs acknowledged by the speaker.

## 🎯 Actionable Takeaways & Next Steps
- Concrete, actionable takeaways for the viewer: what to implement, watch out for, or verify in practice.

Transcript:
{transcript_text}
"""

def analyze_transcript_groq(transcript_text: str, groq_key: str, model_name: str = "qwen/qwen3.8-27b"):
    prompt = ANALYSIS_SYSTEM_PROMPT.format(transcript_text=transcript_text[:24000])

    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    candidate_models = [
        model_name,
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]
    models_to_try = []
    for m in candidate_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_err = None
    for m in models_to_try:
        payload = {
            "model": m,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.3
        }
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or msg.get("reasoning") or "").strip()
                if content:
                    return content
            last_err = RuntimeError(f"Groq {m} returned {resp.status_code}: {resp.text}")
            logging.warning("Groq model %s returned status %d. Trying next candidate...", m, resp.status_code)
        except Exception as e:
            last_err = e
            logging.warning("Groq model %s request exception: %s. Trying next...", m, e)
            continue
    raise last_err or RuntimeError("All Groq models failed.")

def analyze_transcript_nvidia(transcript_text: str, nvidia_key: str, model_name: str = "nvidia/nemotron-3-ultra-550b-a55b"):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_key
    )

    prompt = ANALYSIS_SYSTEM_PROMPT.format(transcript_text=transcript_text[:24000])

    candidate_models = [
        model_name,
        "nvidia/nemotron-3-ultra-550b-a55b"
    ]
    models_to_try = []
    for m in candidate_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_err = None
    for model in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
                stream=True,
                timeout=25.0
            )

            collected_content = []
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None) is not None:
                    collected_content.append(delta.content)

            output_text = "".join(collected_content).strip()
            if output_text:
                return output_text
        except Exception as e:
            last_err = e
            logging.warning("NVIDIA model %s failed: %s. Trying next candidate model...", model, e)
            continue

    raise last_err or RuntimeError("All NVIDIA models failed to generate content.")

def analyze_transcript_gemini(transcript_text: str, gemini_key: str):
    client = genai.Client(api_key=gemini_key)

    prompt = ANALYSIS_SYSTEM_PROMPT.format(transcript_text=transcript_text[:24000])

    candidate_models = ["gemini-2.5-flash", "gemini-3-flash-preview"]
    last_err = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_err = e
            logging.warning("Gemini model %s failed: %s. Trying next model...", model_name, e)
            continue

    raise last_err or RuntimeError("All Gemini models failed to generate content.")

def analyze_transcript_sarvam(transcript_text: str, sarvam_key: str, model_name: str = "sarvam-105b-conversations"):
    headers = {
        "api-subscription-key": sarvam_key,
        "Content-Type": "application/json"
    }

    prompt = ANALYSIS_SYSTEM_PROMPT.format(transcript_text=transcript_text[:12000])

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.3
    }

    response = requests.post("https://api.sarvam.ai/v1/chat/completions", json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    choice_msg = response.json()["choices"][0]["message"]
    return choice_msg.get("content") or choice_msg.get("reasoning_content") or ""

def generate_video_intelligence(transcript_text: str, keys: dict, sarvam_model: str = "sarvam-105b-conversations"):
    user_nvidia_key = keys.get('nvidia', '').strip()
    user_groq_key = keys.get('groq', '').strip()
    user_gemini_key = keys.get('gemini', '').strip()
    user_sarvam_key = keys.get('sarvam', '').strip()

    error_messages = []

    if user_nvidia_key:
        try:
            logging.info("Running NVIDIA NIM video intelligence (Priority 1: Primary)...")
            result = analyze_transcript_nvidia(transcript_text, user_nvidia_key)
            if result:
                return result, "NVIDIA NIM (Nemotron 3 Ultra)"
        except Exception as e:
            logging.warning("NVIDIA NIM analysis failed: %s", e)
            error_messages.append(f"NVIDIA: {str(e)}")

    if user_groq_key:
        try:
            logging.info("Running Groq LPU video intelligence fallback (Priority 2)...")
            result = analyze_transcript_groq(transcript_text, user_groq_key)
            if result:
                return result, "Groq LPU (Qwen/GPT)"
        except Exception as e:
            logging.warning("Groq analysis failed: %s", e)
            error_messages.append(f"Groq: {str(e)}")

    if user_gemini_key:
        try:
            logging.info("Running Google Gemini Flash video intelligence fallback (Priority 3)...")
            result = analyze_transcript_gemini(transcript_text, user_gemini_key)
            if result:
                return result, "Google Gemini Flash"
        except Exception as e:
            logging.warning("Gemini analysis failed: %s", e)
            error_messages.append(f"Gemini: {str(e)}")

    if user_sarvam_key:
        try:
            logging.info("Running Sarvam AI video intelligence fallback (%s)...", sarvam_model)
            result = analyze_transcript_sarvam(transcript_text, user_sarvam_key, sarvam_model)
            if result:
                return result, f"Sarvam AI ({sarvam_model})"
        except Exception as e:
            logging.warning("Sarvam AI analysis failed: %s", e)
            error_messages.append(f"Sarvam: {str(e)}")

    raise RuntimeError(
        f"All configured AI analysis providers failed: {' | '.join(error_messages) if error_messages else 'No API keys provided'}"
    )

CHAT_SYSTEM_PROMPT = """
You are Angkor AI, an expert Executive Video Intelligence Assistant. You already know this video inside and out, as if you have personally watched, studied, and mastered every second, argument, and nuance of it.

CRITICAL INSTRUCTIONS:
1. PERSPECTIVE & TONE (MANDATORY):
   - Talk as someone who ALREADY thoroughly knows and has watched this video.
   - NEVER say or imply robotic phrases like:
     * "I have analyzed the video transcript you provided"
     * "Based on the transcript you gave me"
     * "According to the transcript provided"
     * "You provided a transcript about..."
   - Instead, speak naturally with full mastery: "In this video...", "The speaker breaks down...", "The lecture covers...", etc.

2. CASUAL GREETINGS & OPENERS (e.g. "Hi", "Hello", "Hey", "What's up"):
   - When the user sends a greeting or opens the chat, reply warmly and naturally as an insider who already knows the content.
   - Example tone: "Hey! I'm completely familiar with this video. It covers [Core Subject A], [Core Subject B], and [Key Subject C]. What would you like to dive into — the main arguments, scientific experiments, legal analysis, timestamps, or the final takeaway?"

3. FACTUAL GROUNDING:
   - Ground all substantive answers strictly in the facts, claims, experiments, data points, timestamps, and arguments presented in the video.
   - If the user asks about something not discussed in the video, state naturally: "The video doesn't cover this, but here is brief context..."

4. FORMATTING, TABLES & LATEX:
   - Structure your output elegantly using clear headings, subheadings, and concise paragraphs.
   - Use GitHub Flavored Markdown tables (| Col 1 | Col 2 | Col 3 |) whenever comparing items, summarizing timelines, metrics, trade-offs, or architectures.
   - Use standard LaTeX notation ($...$ for inline, $$...$$ for display equations) for any mathematics, formulas, algorithms, or technical metrics.
   - Respond in clear, professional English (unless the user speaks to you in Hindi or Hinglish, in which case match their language naturally).

Video Knowledge Context:
{transcript_text}
"""

def chat_with_transcript(transcript_text: str, message: str, history: list, keys: dict) -> str:
    user_nvidia_key = keys.get('nvidia', '').strip()
    user_groq_key = keys.get('groq', '').strip()
    user_gemini_key = keys.get('gemini', '').strip()

    system_instruction = CHAT_SYSTEM_PROMPT.format(transcript_text=transcript_text[:20000])

    messages = [{"role": "system", "content": system_instruction}]
    if history and isinstance(history, list):
        for h in history[-6:]:
            if isinstance(h, dict) and 'role' in h and 'content' in h:
                r = h['role'] if h['role'] in ('user', 'assistant') else 'user'
                messages.append({"role": r, "content": str(h['content'])[:2000]})
    messages.append({"role": "user", "content": str(message)[:3000]})

    if user_nvidia_key:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=user_nvidia_key)
        for model_name in ["nvidia/nemotron-3-ultra-550b-a55b"]:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.4,
                    max_tokens=1500,
                    timeout=20.0
                )
                ans = completion.choices[0].message.content
                if ans:
                    return ans.strip()
            except Exception as e:
                logging.warning("NVIDIA Chat model %s failed: %s", model_name, e)
                continue

    if user_groq_key:
        for groq_m in ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
            try:
                headers = {"Authorization": f"Bearer {user_groq_key}", "Content-Type": "application/json"}
                payload = {
                    "model": groq_m,
                    "messages": messages,
                    "max_tokens": 1500,
                    "temperature": 0.4
                }
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    ans = (msg.get("content") or msg.get("reasoning") or "").strip()
                    if ans:
                        return ans
            except Exception as e:
                logging.warning("Groq Chat model %s error: %s", groq_m, e)

    if user_gemini_key:
        for gem_m in ["gemini-2.5-flash", "gemini-3-flash-preview"]:
            try:
                client = genai.Client(api_key=user_gemini_key)
                gemini_prompt = f"{system_instruction}\n\nConversation History:\n"
                for m in messages[1:]:
                    gemini_prompt += f"{m['role'].upper()}: {m['content']}\n"
                gemini_prompt += f"\nASSISTANT:"

                response = client.models.generate_content(
                    model=gem_m,
                    contents=gemini_prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logging.warning("Gemini Chat model %s fallback error: %s", gem_m, e)

    raise RuntimeError("All AI chat providers failed to respond. Please check your API keys.")

