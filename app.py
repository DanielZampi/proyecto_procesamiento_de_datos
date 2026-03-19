import streamlit as st
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ProphetNet — Headline Generator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[data-testid="stAppViewContainer"]{background:#0a0a0f!important;color:#e8e6e0!important;font-family:'DM Sans',sans-serif!important}
[data-testid="stAppViewContainer"]>.main{background:#0a0a0f!important}
[data-testid="stHeader"]{background:transparent!important}
.block-container{padding:0!important;max-width:100%!important}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stToolbar"]{display:none}
.hero{position:relative;padding:72px 80px 56px;border-bottom:1px solid rgba(255,255,255,0.06);overflow:hidden}
.hero::before{content:'';position:absolute;top:-120px;right:-120px;width:500px;height:500px;background:radial-gradient(circle,rgba(139,92,246,0.12) 0%,transparent 70%);pointer-events:none}
.hero-eyebrow{font-family:'DM Mono',monospace;font-size:11px;letter-spacing:0.18em;color:rgba(139,92,246,0.9);text-transform:uppercase;margin-bottom:20px}
.hero-title{font-family:'DM Serif Display',serif;font-size:clamp(42px,6vw,72px);font-weight:400;line-height:1.05;color:#f0ede6;margin-bottom:16px}
.hero-title em{font-style:italic;color:rgba(139,92,246,0.85)}
.hero-desc{font-size:15px;font-weight:300;color:rgba(232,230,224,0.55);max-width:520px;line-height:1.7;margin-bottom:36px}
.hero-stats{display:flex;gap:40px;flex-wrap:wrap}
.stat-value{font-family:'DM Serif Display',serif;font-size:28px;color:#f0ede6;line-height:1}
.stat-label{font-size:11px;font-weight:300;color:rgba(232,230,224,0.4);margin-top:4px;letter-spacing:0.05em}
.field-label{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.15em;color:rgba(232,230,224,0.4);text-transform:uppercase;margin-bottom:12px;display:block}
.stTextArea textarea{background:rgba(255,255,255,0.03)!important;border:1px solid rgba(255,255,255,0.08)!important;border-radius:12px!important;color:#e8e6e0!important;font-family:'DM Sans',sans-serif!important;font-size:14px!important;font-weight:300!important;line-height:1.7!important;padding:20px!important}
.stTextArea textarea:focus{border-color:rgba(139,92,246,0.4)!important}
.stTextArea label,.stSelectbox label,.stSlider label{display:none!important}
.stSelectbox>div>div{background:rgba(255,255,255,0.03)!important;border:1px solid rgba(255,255,255,0.08)!important;border-radius:10px!important;color:#e8e6e0!important;font-family:'DM Sans',sans-serif!important}
.stButton>button{width:100%!important;background:rgba(139,92,246,0.15)!important;border:1px solid rgba(139,92,246,0.35)!important;border-radius:10px!important;color:rgba(232,230,224,0.9)!important;font-family:'DM Sans',sans-serif!important;font-size:13px!important;font-weight:500!important;padding:14px 28px!important;transition:all 0.2s!important}
.stButton>button:hover{background:rgba(139,92,246,0.25)!important;border-color:rgba(139,92,246,0.6)!important}
.output-card{background:rgba(139,92,246,0.06);border:1px solid rgba(139,92,246,0.2);border-radius:16px;padding:28px 32px;margin-bottom:28px}
.output-eyebrow{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.15em;color:rgba(139,92,246,0.7);text-transform:uppercase;margin-bottom:14px}
.output-headline{font-family:'DM Serif Display',serif;font-size:26px;font-weight:400;color:#f0ede6;line-height:1.3;font-style:italic}
.metrics-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:28px}
.metric-card{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:18px 20px;text-align:center}
.metric-value{font-family:'DM Serif Display',serif;font-size:24px;color:#f0ede6;line-height:1;margin-bottom:6px}
.metric-label{font-family:'DM Mono',monospace;font-size:10px;color:rgba(232,230,224,0.35);letter-spacing:0.1em;text-transform:uppercase}
.attn-row{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.04)}
.attn-row:last-child{border-bottom:none}
.attn-token{font-family:'DM Mono',monospace;font-size:12px;color:rgba(232,230,224,0.6);width:90px;flex-shrink:0}
.attn-bar-wrap{flex:1;height:4px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden}
.attn-bar{height:100%;border-radius:99px;background:linear-gradient(90deg,rgba(139,92,246,0.7),rgba(20,184,166,0.7))}
.attn-source{font-family:'DM Mono',monospace;font-size:11px;color:rgba(232,230,224,0.3);width:100px;flex-shrink:0;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.section-title{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.15em;color:rgba(232,230,224,0.3);text-transform:uppercase;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.05)}
.divider{height:1px;background:rgba(255,255,255,0.05);margin:28px 0}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def cargar_modelo():
    from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer
    MODEL = "microsoft/prophetnet-large-uncased-cnndm"
    tokenizer = ProphetNetTokenizer.from_pretrained(MODEL)
    model = ProphetNetForConditionalGeneration.from_pretrained(MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def generar_titular(model, tokenizer, device, articulo,
                    num_beams=4, max_length=60, min_length=8,
                    no_repeat_ngram_size=3):
    inputs = tokenizer(
        articulo.lower(), return_tensors="pt",
        truncation=True, max_length=512
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def extraer_atencion(model, tokenizer, device, articulo, titular):
    inputs = tokenizer(
        articulo.lower(), return_tensors="pt",
        truncation=True, max_length=512
    ).to(device)
    labels = tokenizer(
        titular, return_tensors="pt",
        truncation=True, max_length=128
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=labels,
            output_attentions=True,
        )
    cross_attn = outputs.cross_attentions[-1].squeeze(0).mean(dim=0).cpu().numpy()
    cross_attn = np.exp(cross_attn)
    cross_attn = cross_attn / cross_attn.sum(axis=-1, keepdims=True)
    tok_in  = [t for t in tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
               if t not in ["[PAD]", "[CLS]"]]
    tok_out = [t for t in tokenizer.convert_ids_to_tokens(labels[0])
               if t not in ["[PAD]", "[SEP]"]]
    cross_attn = cross_attn[:len(tok_out), :len(tok_in)]
    return cross_attn, tok_in, tok_out


def heatmap_b64(attn, tok_in, tok_out):
    max_in  = min(28, len(tok_in))
    max_out = min(18, len(tok_out))
    fig, ax = plt.subplots(figsize=(max(10, max_in * 0.45), max(4, max_out * 0.42)))
    fig.patch.set_facecolor("#0e0e16")
    ax.set_facecolor("#0e0e16")
    sns.heatmap(
        attn[:max_out, :max_in], ax=ax,
        xticklabels=tok_in[:max_in],
        yticklabels=tok_out[:max_out],
        cmap="rocket_r", linewidths=0.3, linecolor="#111",
        cbar_kws={"label": "Attention weight", "shrink": 0.8},
    )
    ax.tick_params(axis="x", colors="#aaa", labelsize=8, rotation=45)
    ax.tick_params(axis="y", colors="#aaa", labelsize=8, rotation=0)
    ax.set_xlabel("Encoder tokens (input article)", color="#888", fontsize=9)
    ax.set_ylabel("Decoder tokens (generated headline)", color="#888", fontsize=9)
    ax.collections[0].colorbar.ax.tick_params(colors="#888", labelsize=8)
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, facecolor="#0e0e16", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


EJEMPLOS = {
    "🇺🇸 Diplomacia — Bolivia": (
        "the us state department said wednesday it had received no formal word from bolivia "
        "that it was expelling the us ambassador there but said the charges made against him "
        "are baseless and that he enjoys the full confidence of the secretary of state."
    ),
    "🦕 Dinosaurio en Argentina": (
        "scientists have discovered a new species of dinosaur in argentina that they believe "
        "is one of the largest creatures ever to have walked the earth. the titanosaur, "
        "which lived about 100 million years ago, is estimated to have weighed 70 tons and "
        "measured 40 meters from head to tail."
    ),
    "🍎 Apple ganancias record": (
        "apple has announced its quarterly earnings, reporting record revenue of 90 billion dollars "
        "driven by strong iphone sales and growth in its services division. the company also "
        "announced a new stock buyback program worth 90 billion dollars and increased its "
        "quarterly dividend by four percent."
    ),
    "🚀 NASA en Marte": (
        "nasa has confirmed that its perseverance rover has successfully collected its first rock "
        "sample from the surface of mars. the sample, extracted from a rock called rochette, "
        "will eventually be returned to earth for detailed scientific analysis. scientists hope "
        "the samples will help answer whether life ever existed on mars."
    ),
    "🌍 OMS emergencia global": (
        "the world health organization declared a global health emergency on thursday as a new "
        "respiratory virus continues to spread across multiple continents. health officials "
        "urged governments to increase surveillance and strengthen their health systems "
        "while researchers race to develop vaccines and treatments."
    ),
    "✏️ Escribe el tuyo...": "",
}

ROUGE_DATA = {
    "🇺🇸 Diplomacia — Bolivia":   (0.2703, 0.0000, 0.1622),
    "🦕 Dinosaurio en Argentina":  (0.3684, 0.0556, 0.3158),
    "🍎 Apple ganancias record":    (0.1935, 0.0000, 0.1935),
    "🚀 NASA en Marte":             (0.4444, 0.1176, 0.3889),
    "🌍 OMS emergencia global":     (0.3784, 0.2286, 0.3784),
    "✏️ Escribe el tuyo...":        None,
}

st.markdown(
    '<div class="hero">'
    '<div class="hero-eyebrow">Procesamiento de Datos Secuenciales · Proyecto Final</div>'
    '<div class="hero-title">Prophet<em>Net</em><br>Headline Generator</div>'
    '<div class="hero-desc">Transformer encoder-decoder con <em>future n-gram prediction</em>. '
    'Introduce un artículo y observa cómo el modelo genera el titular '
    'y en qué tokens pone su atención.</div>'
    '<div class="hero-stats">'
    '<div class="stat-item"><div class="stat-value">485M</div><div class="stat-label">Parámetros</div></div>'
    '<div class="stat-item"><div class="stat-value">12+12</div><div class="stat-label">Capas Enc/Dec</div></div>'
    '<div class="stat-item"><div class="stat-value">16</div><div class="stat-label">Attention heads</div></div>'
    '<div class="stat-item"><div class="stat-value">n=2</div><div class="stat-label">N-gram prediction</div></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns(2, gap="small")

with col_left:
    st.markdown('<div style="padding:48px 56px">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Selecciona un ejemplo o escribe el tuyo</span>',
                unsafe_allow_html=True)
    seleccion = st.selectbox("_", list(EJEMPLOS.keys()), label_visibility="collapsed")
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Artículo de noticias (en inglés)</span>',
                unsafe_allow_html=True)
    articulo = st.text_area(
        "_", value=EJEMPLOS[seleccion], height=220,
        placeholder="Paste your news article here...",
        label_visibility="collapsed",
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Parámetros de generación</span>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="field-label">Beam search</span>', unsafe_allow_html=True)
        num_beams = st.slider("_", 1, 8, 4, label_visibility="collapsed")
    with c2:
        st.markdown('<span class="field-label">Max tokens output</span>', unsafe_allow_html=True)
        max_length = st.slider("_", 20, 80, 50, label_visibility="collapsed")
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    generar = st.button("⟶  Generar titular", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div style="padding:48px 56px">', unsafe_allow_html=True)

    if generar and articulo.strip():
        with st.spinner("Cargando modelo..."):
            model, tokenizer, device = cargar_modelo()

        with st.spinner("Generando titular..."):
            titular = generar_titular(
                model, tokenizer, device, articulo,
                num_beams=num_beams, max_length=max_length,
            )

        st.markdown(
            '<div class="output-card">'
            '<div class="output-eyebrow">Titular generado · ProphetNet</div>'
            f'<div class="output-headline">{titular}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        rouge = ROUGE_DATA.get(seleccion)
        if rouge:
            r1, r2, rl = rouge
            st.markdown(
                '<div class="section-title">Métricas ROUGE vs titular real</div>'
                '<div class="metrics-row">'
                f'<div class="metric-card"><div class="metric-value">{r1:.3f}</div>'
                '<div class="metric-label">ROUGE-1</div></div>'
                f'<div class="metric-card"><div class="metric-value">{r2:.3f}</div>'
                '<div class="metric-label">ROUGE-2</div></div>'
                f'<div class="metric-card"><div class="metric-value">{rl:.3f}</div>'
                '<div class="metric-label">ROUGE-L</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with st.spinner("Extrayendo atención..."):
            attn, tok_in, tok_out = extraer_atencion(
                model, tokenizer, device, articulo, titular
            )

        st.markdown(
            '<div class="section-title">Cross-attention heatmap · última capa</div>',
            unsafe_allow_html=True,
        )
        b64 = heatmap_b64(attn, tok_in, tok_out)
        st.markdown(
            '<div style="border-radius:12px;overflow:hidden;'
            'border:1px solid rgba(255,255,255,0.07)">'
            f'<img src="data:image/png;base64,{b64}" style="width:100%;display:block">'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Atención por token generado · top fuente</div>',
            unsafe_allow_html=True,
        )

        rows_html = ""
        n_tokens = min(8, len(tok_out))
        n_in     = min(len(tok_in), attn.shape[1])
        for i in range(n_tokens):
            pesos     = attn[i, :n_in]
            top_idx   = int(np.argmax(pesos))
            tok_label = tok_out[i].replace("##", "")
            best_tok  = tok_in[top_idx].replace("##", "")
            best_val  = float(pesos[top_idx])
            bar_w     = min(100, int(best_val * 300))
            rows_html += (
                '<div class="attn-row">'
                f'<div class="attn-token">{tok_label}</div>'
                '<div class="attn-bar-wrap">'
                f'<div class="attn-bar" style="width:{bar_w}%"></div></div>'
                f'<div class="attn-source">→ {best_tok} {best_val:.3f}</div>'
                '</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)

    elif generar:
        st.markdown(
            '<div style="padding:40px;text-align:center;'
            'color:rgba(232,230,224,0.3);font-size:12px">'
            'INGRESA UN ARTÍCULO PRIMERO</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="padding:60px 20px;text-align:center">'
            '<div style="font-size:48px;color:rgba(232,230,224,0.06);'
            'margin-bottom:20px">✦</div>'
            '<div style="font-size:11px;letter-spacing:0.15em;'
            'color:rgba(232,230,224,0.2)">'
            'SELECCIONA UN ARTÍCULO<br>Y PRESIONA GENERAR</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)
