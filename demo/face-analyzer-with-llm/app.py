"""
Face and Pose Analyzer
- Image tab: upload or snap, analyze on demand
- Realtime tab: gr.Image streaming=True sends webcam frames to Python -> llama-server
"""

import base64
import io

import gradio as gr
import numpy as np
import requests
from PIL import Image

SERVER_URL = "http://localhost:8080"
MEDIA_MARKER = "<__media__>"

INSTRUCTION = "Describe the facial expression, body posture, and mood of the person. Be brief."

PRESET_QUESTIONS = [
    "Describe the facial expression, body posture, and mood.",
    "Does this person look confident? Explain.",
    "What is the dominant emotion visible?",
    "Are there signs of tension or anxiety?",
    "Are the facial expression and body language consistent?",
]


def ndarray_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame.astype("uint8"), "RGB")


def image_to_base64(image: Image.Image, max_size: int = 320) -> str:
    """Resize and encode to base64 JPEG. Keeps inference fast."""
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def get_marker() -> str:
    try:
        r = requests.get(f"{SERVER_URL}/props", timeout=3)
        return r.json().get("default_generation_settings", {}).get("mtmd_marker", MEDIA_MARKER)
    except Exception:
        return MEDIA_MARKER


def call_llama(image: Image.Image, instruction: str) -> str:
    marker = get_marker()
    payload = {
        "prompt": {
            "prompt_string": f"<|im_start|>user\n{marker}\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
            "multimodal_data": [image_to_base64(image)],
        },
        "temperature": 0.2,
        "n_predict": 150,
        "repeat_penalty": 1.3,
        "stop": ["<|im_end|>", "<|im_start|>"],
    }
    resp = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json().get("content", "").strip()


# ── Image tab handler ──────────────────────────────────────────────
def analyze_image(image: Image.Image, question: str) -> str:
    if image is None:
        return "Please upload or capture an image first."
    try:
        return call_llama(image, question.strip() or INSTRUCTION)
    except requests.exceptions.ConnectionError:
        return (
            "Could not connect to llama-server.\n\n"
            "Start with:\n"
            "  llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF \\\n"
            "    --port 8080 --n-gpu-layers 99 --ctx-size 4096 --jinja"
        )
    except Exception as exc:
        return f"Error: {exc}"


# ── Realtime tab handler ───────────────────────────────────────────
# gr.Image streaming=True calls this function for every captured frame.
# Returns the text result which updates the Textbox output.
def analyze_stream(frame: np.ndarray, question: str) -> str:
    if frame is None:
        return ""
    try:
        image = ndarray_to_pil(frame)
        return call_llama(image, question.strip() or INSTRUCTION)
    except Exception as exc:
        return f"Error: {exc}"


# ── CSS ───────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap');
*, body, .gradio-container { font-family: 'IBM Plex Sans', sans-serif !important; }
.gradio-container { background: #f5f4f0 !important; color: #1a1a1a !important; }
.header { border-bottom: 2px solid #1a1a1a; padding-bottom: 14px; margin-bottom: 20px; }
.header h1 { font-size: 1.4rem; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 3px; }
.header p  { font-size: 0.8rem; color: #666; font-family: 'IBM Plex Mono', monospace !important; margin: 0; }
button.primary   { background: #1a1a1a !important; color: #f5f4f0 !important; border: none !important;
                   border-radius: 3px !important; font-weight: 500 !important; }
button.secondary { background: #fff !important; color: #1a1a1a !important;
                   border: 1px solid #c0bdb7 !important; border-radius: 3px !important; font-size: 0.78rem !important; }
.footer-note { text-align: center; font-size: 0.72rem; font-family: 'IBM Plex Mono', monospace;
               color: #999; border-top: 1px solid #d0cec8; padding-top: 12px; margin-top: 8px; }
"""

# ── UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Face & Pose Analyzer") as demo:
    gr.HTML("""
        <div class="header">
            <h1>Face & Pose Analyzer</h1>
            <p>llama-server / SmolVLM-500M-Instruct / Metal / localhost:8080</p>
        </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Image ──────────────────────────────────────────
        with gr.Tab("Image"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        type="pil",
                        sources=["upload", "webcam"],
                        label="Image",
                        height=320,
                    )
                    img_question = gr.Textbox(
                        label="Instruction",
                        value=INSTRUCTION,
                        lines=2,
                    )
                    gr.HTML("<div style='font-size:0.75rem;color:#888;margin:6px 0 4px'>Quick prompts:</div>")
                    with gr.Row():
                        for q in PRESET_QUESTIONS[:3]:
                            b = gr.Button(q[:36] + "...", size="sm", variant="secondary")
                            b.click(fn=lambda t=q: t, outputs=img_question)
                    img_btn = gr.Button("Analyze", variant="primary", size="lg")

                with gr.Column(scale=1):
                    img_output = gr.Textbox(
                        label="Analysis Result",
                        lines=16,
                        placeholder="Analysis will appear here.",
                    )

            img_btn.click(fn=analyze_image, inputs=[img_input, img_question], outputs=img_output)

        # ── Tab 2: Realtime ───────────────────────────────────────
        with gr.Tab("Realtime"):
            gr.Markdown("Webcam frames are sent to llama-server continuously. Adjust the interval with `stream_every`.")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    # streaming=True + stream_every controls capture rate
                    rt_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Webcam",
                        height=320,
                        )
                    rt_question = gr.Textbox(
                        label="Instruction",
                        value=INSTRUCTION,
                        lines=2,
                    )

                with gr.Column(scale=1):
                    rt_output = gr.Textbox(
                        label="Live Analysis",
                        lines=16,
                        placeholder="Start the webcam — analysis will update here.",
                    )

            # stream() fires analyze_stream every stream_every seconds
            rt_input.stream(
                fn=analyze_stream,
                inputs=[rt_input, rt_question],
                outputs=rt_output,
                stream_every=2,   # seconds between frames sent to server
                time_limit=300,   # stop after 5 min of continuous streaming
            )

    gr.HTML("""
        <div class="footer-note">
            runs 100% locally &nbsp;|&nbsp; no data is sent to external servers
        </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        css=CSS,
    )