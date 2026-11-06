"""Web chat UI for the Bright Studio assistant.  Run: python app.py  ->  http://127.0.0.1:7860"""
import gradio as gr

from rag import answer as _a, graph_build as _gb, store as _st

EXAMPLES = [
    "What packages do you offer and how much do they cost?",
    "Who should I talk to about branding?",
    "How do I cancel my contract?",
    "How many clients did we have at the end of 2025?",
    "Do you run Google Ads?",
]

# Paper grain + fibres as inline SVG (feTurbulence), so no image files are needed.
_GRAIN = (
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300'>"
    "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitch'/>"
    "<feColorMatrix values='0 0 0 0 .35  0 0 0 0 .25  0 0 0 0 .15  0 0 0 .55 0'/></filter>"
    "<rect width='100%' height='100%' filter='url(%23n)' opacity='.5'/></svg>\")"
)
_FIBRES = (
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='600' height='600'>"
    "<filter id='f'><feTurbulence type='fractalNoise' baseFrequency='.012 .35' numOctaves='2' seed='7'/>"
    "<feColorMatrix values='0 0 0 0 .45  0 0 0 0 .33  0 0 0 0 .2  0 0 0 .5 -.12'/></filter>"
    "<rect width='100%' height='100%' filter='url(%23f)' opacity='.55'/></svg>\")"
)

CSS = """
:root {
  --ink: #2b2118; --ink-soft: #6b5b4b; --accent: #c2410c; --accent-2: #ea580c;
  --glass: rgba(255,251,245,.55); --glass-edge: rgba(255,255,255,.65);
}
html, body { background: #e7dfd2 !important; }
/* desk */
body {
  background:
    radial-gradient(1200px 700px at 15% 0%, rgba(255,255,255,.55), transparent 60%),
    GRAIN, #e4dccd !important;
}
/* the sheet of paper */
body::before {
  content: ""; position: fixed; inset: 3vh 6vw; z-index: 0; pointer-events: none;
  background:
    radial-gradient(420px 320px at 12% 18%, rgba(234,88,12,.30), transparent 70%),
    radial-gradient(380px 380px at 88% 78%, rgba(124,45,18,.22), transparent 70%),
    radial-gradient(300px 240px at 80% 12%, rgba(250,204,21,.28), transparent 70%),
    repeating-linear-gradient(transparent 0 31px, rgba(110,140,190,.22) 31px 32px),
    FIBRES, GRAIN,
    linear-gradient(180deg, #fbf7ee, #f3ecdd);
  border-radius: 3px;
  box-shadow: 0 1px 1px rgba(0,0,0,.12), 0 8px 18px rgba(60,40,20,.18), 0 40px 80px rgba(60,40,20,.22);
  transform: rotate(-.6deg);
}
/* red margin line like notebook paper */
body::after {
  content: ""; position: fixed; top: 3vh; bottom: 3vh; left: calc(6vw + 64px); width: 2px; z-index: 0;
  background: rgba(200,60,50,.35); pointer-events: none; transform: rotate(-.6deg);
}

.gradio-container {
  position: relative; z-index: 1; background: transparent !important;
  max-width: 860px !important; margin: 0 auto; padding-top: 4vh !important;
  color: var(--ink);
  --body-text-color: var(--ink); --body-text-color-subdued: var(--ink-soft);
  --background-fill-primary: transparent; --background-fill-secondary: transparent;
  --block-background-fill: transparent; --block-border-color: transparent;
  --border-color-primary: rgba(120,90,60,.18);
  --input-background-fill: rgba(255,255,255,.55);
  --button-secondary-background-fill: rgba(255,255,255,.55);
  --button-secondary-text-color: var(--ink);
  --button-primary-background-fill: linear-gradient(135deg, var(--accent-2), var(--accent));
  --button-primary-text-color: #fff;
  --color-accent: var(--accent);
}
.gradio-container, .gradio-container * { font-family: "Inter", "Segoe UI", system-ui, sans-serif; }

#header { text-align: center; padding: 6px 0 18px; }
#header h1 {
  font-family: Georgia, "Times New Roman", serif !important; font-size: 2.3rem; letter-spacing: -.5px;
  margin: 0; color: var(--ink);
}
#header h1 span { color: var(--accent); }
#header p { color: var(--ink-soft); margin: 6px 0 0; font-size: 1rem; }
#header .chip {
  display: inline-block; margin-top: 12px; padding: 3px 12px; font-size: .75rem; letter-spacing: .08em;
  text-transform: uppercase; border-radius: 999px; color: var(--accent);
  background: rgba(255,255,255,.55); border: 1px solid rgba(194,65,12,.25); backdrop-filter: blur(6px);
}

/* the glass panel */
#chat-glass {
  background: var(--glass) !important;
  -webkit-backdrop-filter: blur(18px) saturate(160%); backdrop-filter: blur(18px) saturate(160%);
  border: 1px solid var(--glass-edge) !important; border-radius: 22px !important; padding: 14px !important;
  box-shadow: 0 10px 40px rgba(80,50,20,.22), inset 0 1px 0 rgba(255,255,255,.8), inset 0 -1px 0 rgba(255,255,255,.2);
  position: relative; overflow: hidden;
}
#chat-glass::before { /* soft specular sheen */
  content: ""; position: absolute; inset: 0; pointer-events: none; border-radius: inherit;
  background: linear-gradient(120deg, rgba(255,255,255,.55), rgba(255,255,255,0) 32%);
}
#chat-glass > * { position: relative; }

#chat-glass, #chat-glass > div { width: 100% !important; max-width: none !important; min-width: 0 !important; }
#chat-glass .block, #chat-glass .form, #chat-glass .wrapper, #chat-glass .bubble-wrap, #chat-glass .chatbot,
#chat-glass .examples, #chat-glass .gap, #chat-glass .panel { background: transparent !important; border: none !important; box-shadow: none !important; }
#chat-glass .label-wrap, #chat-glass [data-testid="block-label"] { background: rgba(255,255,255,.6) !important; color: var(--ink) !important; border: 1px solid rgba(255,255,255,.8) !important; }
.gradio-container .examples-holder, .gradio-container .gallery { display: flex; flex-wrap: wrap; gap: 8px; }

/* messages */
#chat-glass .message, #chat-glass .bubble-wrap .message-row .message {
  border-radius: 16px !important; border: 1px solid rgba(255,255,255,.7) !important; color: var(--ink) !important;
  box-shadow: 0 2px 8px rgba(80,50,20,.08);
}
#chat-glass .user .message, #chat-glass .message.user {
  background: linear-gradient(135deg, rgba(234,88,12,.92), rgba(194,65,12,.92)) !important; color: #fff !important;
  border-color: rgba(255,255,255,.35) !important;
}
#chat-glass .bot .message, #chat-glass .message.bot { background: rgba(255,255,255,.62) !important; }
#chat-glass .user .message *, #chat-glass .message.user * { color: #fff !important; }

/* inputs, buttons, examples */
#chat-glass textarea, #chat-glass input[type=text] {
  background: rgba(255,255,255,.6) !important; border: 1px solid rgba(255,255,255,.8) !important;
  border-radius: 14px !important; color: var(--ink) !important; box-shadow: inset 0 1px 3px rgba(80,50,20,.12);
}
#chat-glass textarea:focus { outline: 2px solid rgba(234,88,12,.5) !important; }
#chat-glass button.primary, #chat-glass .submit-button { border-radius: 12px !important; }
#chat-glass button.secondary, #chat-glass .example, #chat-glass .examples button {
  background: rgba(255,255,255,.55) !important; border: 1px solid rgba(255,255,255,.85) !important;
  border-radius: 12px !important; color: var(--ink) !important; transition: transform .15s, box-shadow .15s;
}
#chat-glass button.secondary:hover, #chat-glass .examples button:hover {
  transform: translateY(-1px); box-shadow: 0 6px 16px rgba(80,50,20,.18);
}
#chat-glass .placeholder, #chat-glass .placeholder * { color: var(--ink-soft) !important; }
footer { opacity: .55; }
@media (max-width: 640px) { body::before { inset: 1vh 2vw; } body::after { display: none; } #header h1 { font-size: 1.8rem; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
""".replace("GRAIN", _GRAIN).replace("FIBRES", _FIBRES)


def respond(message, history):
    # Gradio gives history as [{"role","content"}, ...]; keep only plain-text content.
    clean = [{"role": h["role"], "content": h["content"]} for h in history if isinstance(h["content"], str)]
    try:
        text, trace = _a.ask(_gb.load(), _st.connect(), message, verbose=False)
    except BaseException as e:  # Ollama down, model missing, graph not built...
        yield f"Sorry, I couldn't answer ({type(e).__name__}: {e}). Is Ollama running and the graph built?"
        return
    sources = sorted({c.split("#")[0] for c in trace["chunk_ids"]})
    yield text + ("*Sources searched: " + ", ".join(sources) + "*" if sources else "")


with gr.Blocks(title="Bright Studio Assistant") as demo:
    gr.HTML(
        "<div id='header'><h1>Bright <span>Studio</span> Assistant</h1>"
        "<p>Ask about our services, pricing, team, and how we work.</p>"
        "<span class='chip'>Runs locally · Gemma + your documents</span></div>"
    )
    with gr.Column(elem_id="chat-glass", min_width=0):
        gr.ChatInterface(
            respond,
            examples=EXAMPLES,
            fill_width=True,
            chatbot=gr.Chatbot(height=460, placeholder="Hi! Ask me anything about Bright Studio."),
            textbox=gr.Textbox(placeholder="Type your question…", container=False, scale=7),
        )

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        # The paper look is light-only: force Gradio's light theme.
        head="<script>if(!location.search.includes('__theme=light')){location.replace(location.pathname+'?__theme=light')}</script>",
    )
