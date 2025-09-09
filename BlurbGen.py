import os
import json
import ast
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import re

# --- Setup --- #
st.set_page_config(page_title="Blurb Generator", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("No API key found. Please set it in your local .env or in Streamlit Cloud secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4.1-mini"

# ---- session state init (add once) ----
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "retrieved_idx" not in st.session_state:
    st.session_state.retrieved_idx = []
if "context_text" not in st.session_state:
    st.session_state.context_text = ""

# --- Utils ---
def parse_embedding(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.array(x, dtype=np.float32)
    elif isinstance(x, str):
        s = x.strip()
        try:
            arr = np.array(json.loads(s), dtype=np.float32)
        except Exception:
            arr = np.array(ast.literal_eval(s), dtype=np.float32)
    else:
        arr = np.array([], dtype=np.float32)
    return arr

def embed_query(text: str) -> np.ndarray:
    text = (text or "").replace("\n", " ")
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return A_norm @ b_norm

def _dynamic_max_tokens(word_max: int, provided_max_tokens: int) -> int:
    """
    Estimate tokens from words (~1.6 tokens/word) and add ~35% buffer.
    Use whichever is larger: the provided cap or the dynamic estimate.
    """
    approx = int(max(64, round(word_max * 1.6)))     # base estimate
    with_buffer = int(round(approx * 1.35))          # headroom to avoid mid-sentence cuts
    return max(provided_max_tokens, with_buffer)

def generate_blurb(context, brand, page_type, tone, temperature=0.1, max_tokens=120, word_min=60, word_max=80):
    # Page-specific focus
    guidance = {
        "Casino Bonus": "Highlight bonuses, welcome offers, and special promotions explicitly mentioned.",
        "Casino Payment Methods": "Focus on deposit and withdrawal methods, processing speeds, and fees.",
        "Generic": "Provide a balanced overview of the brand using only factual data.",
        "Casino Games": "Highlight variety and quality of games offered.",
        "Casino Games - Blackjack": "Focus specifically on blackjack offerings and features.",
        "Casino Games - Roulette": "Focus specifically on roulette offerings and features.",
        "Casino Games - Slots": "Focus specifically on slots selection and standout titles.",
        "Mobile Casino": "Focus on mobile experience, app availability, and gameplay quality."
    }.get(page_type, "Keep it neutral and factual.")

    # Tone instructions
    tone_instructions = {
        "Neutral": "Use a clear, neutral, factual tone with no promotional language.",
        "Informative": "Use an informative, journalistic tone, like a trusted news site.",
        "Conversational": "Use a friendly, casual tone, like explaining to a friend.",
        "Persuasive": "Use persuasive marketing-style language while staying factual.",
        "SEO-optimized": "Write with keyword clarity, concise sentences, and strong structure."
    }

    example_blurb = (
        "Example Casino.org blurb:\n"
        "\"RealPrize comes out on the top of our list as the best sweepstakes casino in the US right now. "
        "Featuring a solid no deposit bonus of 100,000 GC and 2 SC just for joining, their first purchase offer "
        "elevates this bonus with a massive 2,000,000 GC, 1,000 VIP Points and 80 free SC up for grabs. "
        "Suppose their welcome offer isn't enticing enough for you. In that case, they also offer a daily login "
        "bonus of 5,000 GC and 0.3 SC, letting you experience the platform fully before making any purchases.\""
    )

    if tone == "Brand voice":
        style_instruction = f"Match the style and tone of Casino.org, as shown below:\n{example_blurb}"
    else:
        style_instruction = tone_instructions.get(tone, "Use a clear, neutral tone.")

    # Final prompt (unchanged)
    prompt = f"""
You are generating plain text only. Do NOT use markdown, HTML, LaTeX, or KaTeX.
Return only a single plain text paragraph.

Tone instructions: {style_instruction}

Write a {word_min}-{word_max} word paragraph summarising {brand} for a "{page_type}" page,
using only the provided context below. {guidance}

If the context does not contain sufficient information, respond with exactly:
INSUFFICIENT DATA

Context:
{context}
"""

    # NEW: choose a safer max_tokens (avoid mid-sentence truncation)
    effective_max_tokens = _dynamic_max_tokens(word_max, int(max_tokens))

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=float(temperature),
        max_tokens=int(effective_max_tokens),
        messages=[{"role": "user", "content": prompt}]
    )
    return (resp.choices[0].message.content or "").strip()


def enforce_word_budget(text: str, wmin: int, wmax: int) -> str:
    """
    Lightly enforce a word budget:
    - Returns text untouched if it's within range.
    - If too long, trims to the nearest sentence end *before* wmax.
    - Never truncates mid-sentence.
    - If text is empty or explicitly 'INSUFFICIENT DATA', returns as-is.
    """
    if not text or text.strip().upper() == "INSUFFICIENT DATA":
        return text

    words = text.strip().split()

    # Under budget, leave untouched
    if len(words) <= wmax:
        return text.strip()

    # Truncate safely to last sentence before wmax
    partial = " ".join(words[:wmax + 10])  # small buffer past max
    sentences = re.split(r'(?<=[.!?])\s+', partial)
    
    if sentences:
        return " ".join(s for s in sentences if s).strip()

    return partial.strip()


@st.cache_data(show_spinner=False)
def cache_merge_and_parse(df_embeddings, df_text, addr_col_embed, emb_col, addr_col_text, text_col):
    df_merged = pd.merge(
        df_embeddings,
        df_text[[addr_col_text, text_col]],
        left_on=addr_col_embed,
        right_on=addr_col_text,
        how="inner"
    ).dropna(subset=[emb_col, text_col])

    emb_list = df_merged[emb_col].apply(parse_embedding).tolist()
    dims = [len(e) for e in emb_list]
    if not dims or max(dims) == 0 or len(set(dims)) != 1:
        raise ValueError("Embeddings missing or inconsistent dimensions.")
    E = np.vstack(emb_list)
    urls = df_merged[addr_col_embed].tolist()
    texts = df_merged[text_col].tolist()
    return df_merged, E, urls, texts

# --- Sidebar controls ---
st.sidebar.header("Settings")
tone = st.sidebar.selectbox(
    "Blurb Tone",
    ["Brand voice", "Neutral", "Informative", "Conversational", "Persuasive", "SEO-optimized"],
    index=0
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
min_sim_cutoff = st.sidebar.slider("Min similarity cutoff", 0.0, 1.0, 0.20, 0.01)
word_min = st.sidebar.number_input("Min words", 20, 200, 60, 5)
word_max = st.sidebar.number_input("Max words", 30, 300, 80, 5)
brand_aliases = st.sidebar.text_input("Brand aliases (comma-separated)", help="e.g. Chumba, Chumba Casino, Chumba Games")

# --- UI ---
st.title("Casino Blurb Generator (MVP)")

# --- Step 1: Upload embeddings CSV ---
st.subheader("1) Upload embeddings CSV (URLs + embeddings)")
sf_embeddings = st.file_uploader("Upload embeddings CSV", type=["csv"], key="embeddings")
if not sf_embeddings:
    st.info("Upload the embeddings CSV to continue.")
    st.stop()

df_embeddings = pd.read_csv(sf_embeddings, keep_default_na=False)

# Column mapping
cols_embed = {c.lower(): c for c in df_embeddings.columns}
addr_col_embed = cols_embed.get("address") or cols_embed.get("url") or "Address"
emb_col = cols_embed.get("openai embeddings 1") or "OpenAI Embeddings 1"

if addr_col_embed not in df_embeddings.columns or emb_col not in df_embeddings.columns:
    st.error("Embeddings CSV must have 'Address' and 'OpenAI Embeddings 1'")
    st.stop()

# --- Step 2: Upload text extraction CSV ---
st.subheader("2) Upload text extraction CSV (URLs + Text Extract 1)")
sf_text = st.file_uploader("Upload custom_extraction_text.csv", type=["csv"], key="textcsv")
if not sf_text:
    st.info("Upload the text extraction CSV to continue.")
    st.stop()

df_text = pd.read_csv(sf_text, keep_default_na=False)
cols_text = {c.lower(): c for c in df_text.columns}
addr_col_text = cols_text.get("address") or cols_text.get("url") or "Address"
text_col = cols_text.get("text extract 1") or "Text Extract 1"

if addr_col_text not in df_text.columns or text_col not in df_text.columns:
    st.error("Text CSV must have 'Address' and 'Text Extract 1'")
    st.stop()

# --- Merge + parse (cached) ---
try:
    df_merged, E, urls, texts = cache_merge_and_parse(
        df_embeddings, df_text,
        addr_col_embed, emb_col,
        addr_col_text, text_col
    )
except Exception as e:
    st.error(f"Failed to parse data: {e}")
    st.stop()

st.success(f"Merged {len(df_merged)} rows where both embeddings and text exist.")

# --- Step 3: Retrieval & Blurb Generation (split into 2 actions) ---
st.subheader("3) Retrieve Context and Generate Blurb")

page_types = [
    "Casino Bonus", "Casino Payment Methods", "Generic", "Casino Games",
    "Casino Games - Blackjack", "Casino Games - Roulette", "Casino Games - Slots", "Mobile Casino"
]
page_type = st.selectbox("Page Type", page_types)
brand = st.text_input("Brand Name (used for retrieval & generation)")
top_k = st.slider("Top-K to retrieve", 1, 25, 8, 1)

# ---------- A) SEARCH ----------

if st.button("Search (retrieve similar URLs)"):
    if not brand.strip():
        st.error("Enter a brand name to search.")
        st.stop()

    # Build retrieval query using brand + aliases
    alias_str = ""
    if brand_aliases.strip():
        alias_str = " " + " ".join([a.strip() for a in brand_aliases.split(",") if a.strip()])
    retrieval_query = f"{brand}{alias_str}"

    # Retrieve
    q = embed_query(retrieval_query)
    sims = cosine_sim_matrix(E, q)
    order = np.argsort(-sims)
    order = [i for i in order if sims[i] >= min_sim_cutoff]

    if len(order) == 0:
        st.warning("No documents passed the similarity cutoff. Lower the cutoff or try different aliases.")
        st.stop()

    idx = order[:top_k]
    st.session_state.retrieved_idx = idx

    # Results table
    out = pd.DataFrame({
        "rank": range(1, len(idx) + 1),
        "url": [urls[i] for i in idx],
        "similarity": [float(sims[i]) for i in idx],
        "sample_text": [texts[i][:250] + ("..." if len(texts[i]) > 250 else "") for i in idx]
    })
    st.session_state.results_df = out

    # Build default context (cap by characters)
    top_context_n = min(4, len(idx))
    selected_texts = [texts[i] for i in idx[:top_context_n]]
    default_context = "\n\n---\n\n".join(selected_texts)
    st.session_state.context_text = default_context

    # Show retrieved results if available
    if st.session_state.results_df is not None:
        st.subheader("Retrieved Pages")
        st.dataframe(st.session_state.results_df, use_container_width=True)
        st.download_button(
            "Download retrieved pages CSV",
            data=st.session_state.results_df.to_csv(index=False),
            file_name="retrieval_results.csv",
            mime="text/csv"
        )

    # Editable context box
    st.subheader("Context (editable before generation)")
    st.session_state.context_text = st.text_area(
        "Context to use for generation",
        st.session_state.context_text,
        height=300
    )

# ---------- B) GENERATE ----------
if st.button("Generate blurb", type="primary"):
    ctx = (st.session_state.context_text or "").strip()
    if not ctx:
        st.error("No context available. Run Search first or paste context.")
        st.stop()

    with st.spinner("Generating..."):
        blurb = generate_blurb(
            context=ctx,
            brand=brand,
            page_type=page_type,
            temperature=temperature,
            max_tokens=500,          # your existing call – kept
            tone=tone,
            word_min=word_min,
            word_max=word_max
        )
        blurb = enforce_word_budget(blurb, word_min, word_max)

    wc = len(blurb.split()) if blurb and blurb.upper() != "INSUFFICIENT DATA" else 0
    st.subheader("Generated Blurb")
    #st.caption(f"DEBUG OUTPUT LENGTH: {wc} words / {len(blurb) if blurb else 0} characters")
    st.text_area("Blurb (plain text)", blurb or "", height=160)
    st.caption(f"Word count: {wc}")
    
    st.download_button(
        "Download blurb as .txt",
        data=blurb or "",
        file_name=f"{brand.replace(' ', '_')}_blurb.txt",
        mime="text/plain"
    )
