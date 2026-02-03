# streamlit_app.py
# Catalog Import Demo — POC
# Flow: Import + Fetch → Cleanup + Merchant Readiness → AI Category → Create Collection → Publish → Output

import json
import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd
import streamlit as st


# =============================
# Page / Theme
# =============================
st.set_page_config(page_title="Catalog Import Demo", layout="wide")

ACCENT = "rgba(59, 130, 246, 0.95)"
ACCENT_HOVER = "rgba(59, 130, 246, 1.00)"
TEXT = "#ffffff"
BG = "#0b0f14"
SURFACE = "#0f1720"
BORDER = "rgba(255,255,255,0.14)"
BORDER_SOFT = "rgba(255,255,255,0.10)"

st.markdown(
    f"""
<style>
html, body, [class*="css"] {{
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial !important;
  color: {TEXT} !important;
}}
.stApp {{
  background: {BG} !important;
  color: {TEXT} !important;
}}
section[data-testid="stSidebar"] {{
  background: {BG} !important;
}}
h1, h2, h3, h4, h5, h6,
small, .stCaption, p, label, span {{
  color: {TEXT} !important;
}}

/* Ensure checkbox labels never go dark */
div[data-testid="stCheckbox"] label {{
  color: {TEXT} !important;
}}

/* Buttons */
.stButton > button {{
  background: {ACCENT} !important;
  color: {TEXT} !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  box-shadow: 0 12px 30px rgba(59,130,246,0.25) !important;
  border-radius: 10px !important;
  padding: 0.60rem 1.00rem !important;
  transition: transform .08s ease, filter .08s ease, background .08s ease !important;
}}
.stButton > button:hover {{
  background: {ACCENT_HOVER} !important;
  transform: translateY(-1px) !important;
  filter: brightness(1.06) !important;
}}
.stButton > button:focus {{
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(59,130,246,0.28) !important;
}}

/* Inputs */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
textarea {{
  background: {SURFACE} !important;
  color: {TEXT} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 10px !important;
}}
div[data-baseweb="input"] input:hover,
div[data-baseweb="textarea"] textarea:hover {{
  border: 1px solid {ACCENT_HOVER} !important;
}}
div[data-baseweb="input"] input:focus,
div[data-baseweb="textarea"] textarea:focus {{
  border: 1px solid {ACCENT_HOVER} !important;
  box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}}
div[data-baseweb="select"] > div {{
  background: {SURFACE} !important;
  color: {TEXT} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 10px !important;
}}
div[data-baseweb="select"] > div:hover {{
  border: 1px solid {ACCENT_HOVER} !important;
}}
span[data-baseweb="tag"] {{
  background: {ACCENT} !important;
  color: {TEXT} !important;
  border: none !important;
}}

/* Dataframes */
div[data-testid="stDataFrame"] {{
  background: {SURFACE} !important;
  border-radius: 14px !important;
  border: 1px solid {BORDER_SOFT} !important;
  overflow: hidden !important;
}}

/* Alerts */
div[data-testid="stAlert"] {{
  background: {SURFACE} !important;
  color: {TEXT} !important;
  border: 1px solid {BORDER_SOFT} !important;
  border-radius: 12px !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Catalog Import Demo")
st.caption("One-run demo: Import + Fetch → Cleanup + Merchant Readiness → AI Category → Create Collection → Publish → Output")


# =============================
# Constants
# =============================
NETWORK_NAME_MAP = {
    600: "FlexOffers US",
    3: "CJ US",
    603: "Impact",
    605: "Sovrn",
    602: "Rakuten",
    601: "Partnerize US",
    606: "Pepperjam US",
    607: "AvantLink US",
    608: "Awin US",
    609: "Commission Factory",
}

IN_SCOPE_NETWORK_ID = 600
MIN_COMMISSION = 5.0

SUPPLEMENT_KWS = [
    "supplement", "supplements", "vitamin", "capsule", "capsules", "gummy", "gummies",
    "turmeric", "collagen", "probiotic", "omega", "biotin", "creatine", "whey",
    "nutrition", "multivitamin", "fish oil",
]
WELLNESS_KWS = ["massage", "massager", "theragun",
                "therabody", "recovery", "percussive"]


# =============================
# Session State
# =============================
def ss_init():
    defaults = {
        "parent_json_text": "",
        "parent_json": None,
        "run_id": None,
        "run_name": None,

        "raw_df": None,
        "scoped_df": None,
        "ai_df": None,

        # (merchant, network_id) -> profile {merchant_id, default_commission, allow_under_5, associated_sovrn}
        "merchant_profiles": {},

        # cleanup controls
        "merchant_only_enabled": True,
        "merchant_only_name": "Alo Yoga",
        "smart_cleanup_enabled": True,
        "keep_ai_categories": [],  # hard keep list

        # collection
        "collection_mode": None,   # "brand" | "category"
        "collection_name": "",
        "collection_hero_bytes": None,

        # publish / output
        "published_df": None,
        "published_at": None,
        "rules_table": [],

        # optional upload
        "uploaded_csv_bytes": None,
        "uploaded_csv_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_downstream(after: str):
    """
    Reset only what is *after* the step you just completed.
    after: "import" | "cleanup" | "ai" | "collection"
    """
    if after == "import":
        st.session_state.scoped_df = None
        st.session_state.ai_df = None
        st.session_state.collection_mode = None
        st.session_state.collection_name = ""
        st.session_state.collection_hero_bytes = None
        st.session_state.published_df = None
        st.session_state.published_at = None
        st.session_state.rules_table = []

    elif after == "cleanup":
        st.session_state.ai_df = None
        st.session_state.collection_mode = None
        st.session_state.collection_name = ""
        st.session_state.collection_hero_bytes = None
        st.session_state.published_df = None
        st.session_state.published_at = None

    elif after == "ai":
        st.session_state.collection_mode = None
        st.session_state.collection_name = ""
        st.session_state.collection_hero_bytes = None
        st.session_state.published_df = None
        st.session_state.published_at = None

    elif after == "collection":
        st.session_state.published_df = None
        st.session_state.published_at = None


ss_init()


# =============================
# Helpers
# =============================
def safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_parent_json(text: str):
    if not text or not text.strip():
        return None, "Paste a JSON object."
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None, "Parent JSON must be a single JSON object."
        return obj, None
    except Exception as e:
        return None, f"Invalid JSON: {e}"


def extract_run_id(parent_obj: dict) -> Optional[str]:
    if not isinstance(parent_obj, dict):
        return None
    return (parent_obj.get("pool_id") or "").strip() or None


def extract_search_terms(parent_obj: dict) -> List[str]:
    if not isinstance(parent_obj, dict):
        return []
    terms: List[str] = []
    search = parent_obj.get("search", [])
    if isinstance(search, list):
        for s in search:
            if not isinstance(s, dict):
                continue
            field = norm(s.get("field", ""))
            val = (s.get("value") or "").strip()
            if not val:
                continue
            if field in ("network.name", "network", "network_name"):
                continue
            terms.append(val.strip())

    seen = set()
    out: List[str] = []
    for t in terms:
        k = norm(t)
        if k and k not in seen:
            out.append(t)
            seen.add(k)
    return out


def build_run_name(parent_obj: Optional[dict]) -> str:
    if not isinstance(parent_obj, dict):
        return "demo_run"
    pool = (parent_obj.get("pool_id") or "").strip()
    terms = extract_search_terms(parent_obj)
    bits = []
    if terms:
        bits.append(terms[0])
    if pool:
        bits.append(pool[:8])
    return " • ".join([b for b in bits if b]) or "demo_run"


def show_csv_help():
    st.info(
        "CSV must include: **title, merchant, url**\n\n"
        "Optional: image_url, commission_percent, network_id, network_name, category"
    )


def normalize_products_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    for col in ["title", "merchant", "url"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    for col in ["image_url", "commission_percent", "network_id", "network_name", "category"]:
        if col not in df.columns:
            df[col] = ""

    for c in ["title", "merchant", "url", "image_url", "commission_percent", "network_id", "network_name", "category"]:
        df[c] = df[c].fillna("").astype(str)

    def infer_network_id(row) -> int:
        n = safe_int(row.get("network_id"))
        if n is not None:
            return n
        m = norm(row.get("merchant", ""))
        if "alo" in m:
            return 600
        return 3

    df["network_id"] = df.apply(infer_network_id, axis=1).astype(int)
    df["network_name"] = df["network_id"].apply(
        lambda x: NETWORK_NAME_MAP.get(int(x), f"Network {int(x)}"))

    if "product_id" not in df.columns:
        df.insert(0, "product_id", [f"row_{i+1}" for i in range(len(df))])

    return df


def load_products_csv(uploaded_bytes: Optional[bytes]) -> pd.DataFrame:
    if uploaded_bytes:
        df = pd.read_csv(pd.io.common.BytesIO(uploaded_bytes)).copy()
        return normalize_products_df(df)

    csv_path = Path(__file__).parent / "demo_products.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing demo_products.csv next to streamlit_app.py: {csv_path}")
    df = pd.read_csv(csv_path).copy()
    return normalize_products_df(df)


def ensure_demo_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps the demo reliable:
      - enough 600 rows
      - a duplicate URL (dedupe)
      - a missing URL row (blocked)
      - a low-commission row (blocked unless override)
    """
    out = df.copy()

    # Ensure enough 600 rows
    m600 = out["network_id"].astype(int) == 600
    ct600 = int(m600.sum())
    if ct600 < 6:
        seed = (
            out[m600].iloc[0].to_dict()
            if ct600 > 0
            else {
                "product_id": "row_seed",
                "title": "ALO | Airbrush Legging",
                "merchant": "Alo Yoga",
                "url": "https://example.com/alo_seed",
                "image_url": "",
                "commission_percent": "8",
                "network_id": 600,
                "network_name": "FlexOffers US",
                "category": "Apparel > Pants",
            }
        )
        needed = 6 - ct600
        variants = ["Black", "Navy", "Anthracite",
                    "Espresso", "Stone", "Olive"]
        sizes = ["XS", "S", "M", "L", "XL"]
        synth = []
        for i in range(needed):
            row = dict(seed)
            row["product_id"] = f"row_600_{i+1}"
            v = variants[i % len(variants)]
            sz = sizes[i % len(sizes)]
            row["title"] = f"ALO | Airbrush High-Waist Legging ({v}) Size {sz}"
            row["url"] = f"https://example.com/alo_{i+1}"
            row["commission_percent"] = row.get("commission_percent") or "8"
            row["network_id"] = 600
            row["network_name"] = "FlexOffers US"
            synth.append(row)
        out = pd.concat([out, pd.DataFrame(synth)], ignore_index=True)

    # Duplicate URL row -> dedupe trigger
    if out["url"].astype(str).str.strip().ne("").any():
        first_url = out["url"].astype(
            str).str.strip().loc[lambda s: s.ne("")].iloc[0]
        dup_row = out.iloc[0].to_dict()
        dup_row["product_id"] = "row_dup_url"
        dup_row["url"] = first_url
        dup_row["title"] = str(dup_row.get("title", "")) + " (Duplicate URL)"
        out = pd.concat([out, pd.DataFrame([dup_row])], ignore_index=True)

    # Missing URL -> blocked
    miss_url = {
        "product_id": "row_missing_url",
        "title": "Alo Yoga | Demo Item (Missing URL)",
        "merchant": "Alo Yoga",
        "url": "",
        "image_url": "",
        "commission_percent": "8",
        "network_id": 600,
        "network_name": "FlexOffers US",
        "category": "Apparel",
    }
    out = pd.concat([out, pd.DataFrame([miss_url])], ignore_index=True)

    # Low commission -> blocked unless override
    low_comm = {
        "product_id": "row_low_comm",
        "title": "Alo Yoga | Demo Item (Low Commission)",
        "merchant": "Alo Yoga",
        "url": "https://example.com/alo_low_comm",
        "image_url": "",
        "commission_percent": "2.5",
        "network_id": 600,
        "network_name": "FlexOffers US",
        "category": "Apparel",
    }
    out = pd.concat([out, pd.DataFrame([low_comm])], ignore_index=True)

    return normalize_products_df(out)


def prime_demo_profile_for_600():
    key = ("Alo Yoga", 600)
    if key not in st.session_state.merchant_profiles:
        st.session_state.merchant_profiles[key] = {
            "merchant_id": "49651",
            "default_commission": 8.0,
            "allow_under_5": False,
            "associated_sovrn": False,  # NEW
        }


def networks_summary(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["network_id", "network_name"]
                   ).size().reset_index(name="products")
    return t.sort_values(by="products", ascending=False).reset_index(drop=True)


def ai_categorize_row(title: str, category_hint: str) -> Tuple[str, str]:
    t = norm(title)
    c = norm(category_hint)

    # brand (demo)
    if "alo" in t:
        brand = "ALO"
    elif "skims" in t:
        brand = "SKIMS"
    elif "nike" in t:
        brand = "Nike"
    elif "sephora" in t:
        brand = "Sephora"
    else:
        brand = ""

    # AI category (demo normalization)
    if any(k in t for k in ["legging", "leggings"]):
        cat = "Apparel — Leggings"
    elif any(k in t for k in ["bra", "bralette"]):
        cat = "Apparel — Intimates"
    elif any(k in t for k in ["tank", "tee", "tshirt", "t shirt", "top", "hoodie", "shirt", "sweatshirt"]):
        cat = "Apparel — Tops"
    elif "skincare" in c or any(k in t for k in ["serum", "cleanser", "moisturizer", "spf", "sunscreen"]):
        cat = "Beauty — Skincare"
    else:
        cat = "Other"

    return brand, cat


def add_ai_preview(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cats = []
    for _, r in out.iterrows():
        _, cat = ai_categorize_row(r.get("title", ""), r.get("category", ""))
        cats.append(cat)
    out["ai_category_preview"] = cats
    return out


def apply_cleanup(
    df: pd.DataFrame,
    keep_categories: List[str],
    merchant_only_enabled: bool,
    merchant_only_name: str,
    smart_cleanup_enabled: bool,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Hard enforcement:
      1) keep ONLY network 600
      2) hard-keep selected AI categories (if provided)
      3) optional merchant-only filter
      4) optional smart cleanup (noise)
      5) dedupe
    """
    stats = {
        "start": len(df),
        "removed_out_of_scope_network": 0,
        "removed_category_scope": 0,
        "removed_merchant_only": 0,
        "removed_smart_cleanup": 0,
        "removed_dedupe": 0,
        "end": 0,
    }

    out = df.copy()

    # (1) enforce network scope
    before = len(out)
    out = out[out["network_id"].astype(int) == IN_SCOPE_NETWORK_ID].copy()
    stats["removed_out_of_scope_network"] = before - len(out)

    # add AI category preview for scope filtering
    out = add_ai_preview(out)

    # (2) hard keep selected categories
    if keep_categories:
        before = len(out)
        out = out[out["ai_category_preview"].isin(keep_categories)].copy()
        stats["removed_category_scope"] = before - len(out)

    # (3) merchant-only
    if merchant_only_enabled:
        before = len(out)
        out = out[out["merchant"].astype(str).str.strip() == str(
            merchant_only_name).strip()].copy()
        stats["removed_merchant_only"] = before - len(out)

    # (4) smart cleanup
    if smart_cleanup_enabled:
        before = len(out)

        def is_noise(r) -> bool:
            t = norm(r.get("title", ""))
            c = norm(r.get("category", ""))
            if any(k in t for k in SUPPLEMENT_KWS) or any(k in c for k in SUPPLEMENT_KWS):
                return True
            if any(k in t for k in WELLNESS_KWS) or any(k in c for k in WELLNESS_KWS):
                return True
            return False

        out = out[~out.apply(is_noise, axis=1)].copy()
        stats["removed_smart_cleanup"] = before - len(out)

    # (5) dedupe
    before = len(out)
    out["_dedupe_key"] = out["url"].astype(str).str.strip()
    out.loc[out["_dedupe_key"] == "", "_dedupe_key"] = (
        out["merchant"].astype(str).str.strip() + "||" +
        out["title"].astype(str).str.strip()
    )
    out = out.drop_duplicates(subset=["_dedupe_key"], keep="first").copy()
    out.drop(columns=["_dedupe_key"], inplace=True)
    stats["removed_dedupe"] = before - len(out)

    stats["end"] = len(out)
    return out.reset_index(drop=True), stats


def apply_quality_gates_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixable issues only:
      - missing merchant_id
      - missing commission details (row or default)
      - commission below 5% (unless allow override)
      - missing direct URL
    """
    out = df.copy()

    brands, cats = [], []
    for _, r in out.iterrows():
        b, c = ai_categorize_row(r.get("title", ""), r.get("category", ""))
        brands.append(b)
        cats.append(c)
    out["brand_ai"] = brands
    out["ai_category"] = cats

    # resolve commission: row commission, else default
    resolved = []
    for _, r in out.iterrows():
        row_comm = safe_float(r.get("commission_percent"))
        if row_comm is not None:
            resolved.append(row_comm)
            continue
        key = (str(r.get("merchant", "")).strip(), int(r.get("network_id")))
        prof = st.session_state.merchant_profiles.get(key, {}) or {}
        resolved.append(safe_float(prof.get("default_commission")))
    out["commission_resolved"] = resolved

    status, reason = [], []
    for _, r in out.iterrows():
        merchant = str(r.get("merchant", "")).strip()
        net_id = int(r.get("network_id"))
        url = str(r.get("url", "")).strip()

        prof = st.session_state.merchant_profiles.get((merchant, net_id))
        if not prof or not str(prof.get("merchant_id", "")).strip():
            status.append("BLOCKED")
            reason.append("missing_merchant_id")
            continue

        if not url:
            status.append("BLOCKED")
            reason.append("missing_direct_url")
            continue

        comm = r.get("commission_resolved")
        if comm is None:
            status.append("BLOCKED")
            reason.append("missing_commission")
            continue

        allow_under_5 = bool(prof.get("allow_under_5", False))
        if float(comm) < MIN_COMMISSION and not allow_under_5:
            status.append("BLOCKED")
            reason.append("commission_below_5")
            continue

        status.append("READY")
        reason.append("ok")

    out["status"] = status
    out["reason"] = reason

    # simple ranking
    scores = []
    for _, r in out.iterrows():
        s = 0.0
        comm = r.get("commission_resolved")
        if comm is not None:
            s += float(comm)
        if r.get("brand_ai") == "ALO":
            s += 2.0
        scores.append(s)
    out["rank_score"] = scores

    out = out.sort_values(by=["status", "rank_score"], ascending=[
                          False, False]).reset_index(drop=True)
    return out


def format_filter_summary() -> str:
    cats = st.session_state.keep_ai_categories or []
    merchant_only = st.session_state.merchant_only_name if st.session_state.merchant_only_enabled else "N/A"
    return (
        f"Network: {NETWORK_NAME_MAP.get(IN_SCOPE_NETWORK_ID, str(IN_SCOPE_NETWORK_ID))}  |  "
        f"MerchantOnly: {merchant_only}  |  "
        f"AI Category Scope: {', '.join(cats) if cats else 'N/A'}"
    )


# =============================
# Optional CSV Upload
# =============================
with st.expander("Optional: Upload a CSV instead of using demo_products.csv", expanded=False):
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        st.session_state.uploaded_csv_bytes = up.getvalue()
        st.session_state.uploaded_csv_name = up.name
        st.success(f"Loaded upload: {up.name}")


# =============================
# 1) Import + Fetch
# =============================
st.subheader("1) Import + Fetch")

parent_text = st.text_area(
    "Paste parent JSON",
    value=st.session_state.parent_json_text,
    height=120,
    placeholder='{"pool_id":"...","network_ids":[600],"merchant_ids":[49651],"search":[{"field":"any","value":"legging","operator":"like"}]}',
)

col_a, col_b = st.columns([1, 6])
with col_a:
    if st.button("Import + Fetch", type="primary"):
        obj, err = parse_parent_json(parent_text)
        if err:
            st.error(err)
        else:
            with st.spinner("Loading queue…"):
                st.session_state.parent_json_text = parent_text
                st.session_state.parent_json = obj
                st.session_state.run_id = extract_run_id(obj)
                st.session_state.run_name = build_run_name(obj)

                try:
                    df = load_products_csv(st.session_state.uploaded_csv_bytes)
                except FileNotFoundError as e:
                    st.error(str(e))
                    show_csv_help()
                    st.stop()
                except ValueError as e:
                    st.error(str(e))
                    show_csv_help()
                    st.stop()
                except Exception as e:
                    st.error(f"Failed to load CSV: {e}")
                    show_csv_help()
                    st.stop()

                df = ensure_demo_triggers(df)
                st.session_state.raw_df = df
                prime_demo_profile_for_600()
                reset_downstream(after="import")

            st.success("Loaded queue.")
            st.rerun()

with col_b:
    if st.session_state.parent_json:
        run_id = st.session_state.run_id or "(missing pool_id)"
        run_name = st.session_state.run_name or "demo_run"
        st.caption(
            f"run_id: {run_id}  |  run_name: {run_name}  |  in-scope network: {IN_SCOPE_NETWORK_ID}")

st.divider()

if st.session_state.raw_df is None:
    st.info("Paste parent JSON, then click **Import + Fetch**.")
    st.stop()

raw_df = st.session_state.raw_df.copy()


# =============================
# 2) Cleanup + Merchant Readiness
# =============================
st.subheader("2) Cleanup + Merchant Readiness")

net_ids_found = sorted(set(raw_df["network_id"].astype(int).tolist()))
out_of_scope_ct = int(
    (raw_df["network_id"].astype(int) != IN_SCOPE_NETWORK_ID).sum())

st.caption(
    f"Summary: Total products: {len(raw_df)} • Networks found: {len(net_ids_found)} • "
    f"In-scope: {IN_SCOPE_NETWORK_ID} • Out-of-scope: {out_of_scope_ct}"
)

with st.expander("Network breakdown (raw queue)", expanded=False):
    st.dataframe(networks_summary(raw_df),
                 use_container_width=True, hide_index=True)

# AI category scope options (computed on 600 rows)
preview_600 = raw_df[raw_df["network_id"].astype(
    int) == IN_SCOPE_NETWORK_ID].copy()
preview_600 = add_ai_preview(preview_600)
cats_present = sorted(
    preview_600["ai_category_preview"].dropna().astype(str).unique().tolist())

if cats_present:
    default_keep = st.session_state.keep_ai_categories or cats_present
    keep_cats = st.multiselect(
        "Keep only these AI categories (hard remove everything else)",
        options=cats_present,
        default=default_keep,
    )
    st.session_state.keep_ai_categories = list(keep_cats)
else:
    st.session_state.keep_ai_categories = []

c1, c2 = st.columns([2, 3])
with c1:
    st.session_state.merchant_only_enabled = st.checkbox(
        "Apply merchant-only policy (Network 600)",
        value=bool(st.session_state.merchant_only_enabled),
    )
with c2:
    st.session_state.merchant_only_name = st.text_input(
        "Merchant (when merchant-only is enabled)",
        value=str(st.session_state.merchant_only_name),
    )

st.session_state.smart_cleanup_enabled = st.checkbox(
    "Enable smart cleanup",
    value=bool(st.session_state.smart_cleanup_enabled),
)

if st.button("Cleanup", type="primary"):
    with st.spinner("Running cleanup…"):
        cleaned, stats = apply_cleanup(
            raw_df,
            keep_categories=st.session_state.keep_ai_categories,
            merchant_only_enabled=bool(st.session_state.merchant_only_enabled),
            merchant_only_name=str(st.session_state.merchant_only_name),
            smart_cleanup_enabled=bool(st.session_state.smart_cleanup_enabled),
        )
        st.session_state.scoped_df = cleaned
        reset_downstream(after="cleanup")

    st.success(
        f"Cleaned queue: {stats['start']} → {stats['end']} "
        f"(removed: out_of_scope {stats['removed_out_of_scope_network']}, "
        f"category_scope {stats['removed_category_scope']}, "
        f"merchant_only {stats['removed_merchant_only']}, "
        f"smart_cleanup {stats['removed_smart_cleanup']}, "
        f"dedupe {stats['removed_dedupe']})"
    )
    st.rerun()

if st.session_state.scoped_df is None:
    st.info("Click **Cleanup** to continue.")
    st.stop()

scoped = st.session_state.scoped_df.copy()
st.caption(f"Queue after cleanup: {len(scoped)} items")

# No image column
st.dataframe(
    scoped[["product_id", "title", "merchant", "network_id",
            "network_name", "commission_percent", "url"]],
    use_container_width=True,
    hide_index=True,
    column_config={"url": st.column_config.LinkColumn("URL")},
)

st.divider()

st.subheader("Merchant Readiness")
st.caption(
    f"Fixable issues only: missing merchant ID, missing commission details, commission below {MIN_COMMISSION:g}%, "
    f"or missing direct URL. (Associated with Sovrn is stored for the run.)"
)

merchants_in_scope = (
    scoped[["merchant", "network_id", "network_name"]]
    .drop_duplicates()
    .sort_values(by=["network_id", "merchant"])
    .reset_index(drop=True)
)

for _, row in merchants_in_scope.iterrows():
    merchant = str(row["merchant"]).strip()
    net_id = int(row["network_id"])
    net_name = str(row["network_name"]).strip()
    key = (merchant, net_id)

    existing = st.session_state.merchant_profiles.get(key, {}) or {}
    with st.expander(f"Merchant: {merchant}  |  Network: {net_id} — {net_name}", expanded=False):
        a, b, c = st.columns([2, 2, 1])

with a:
    merch_id = st.text_input(
        "Merchant ID",
        value=str(existing.get("merchant_id", "") or ""),
        key=f"mid_{merchant}_{net_id}",
    )

with b:
    default_comm = st.text_input(
        "Default commission % (used when row commission missing)",
        value="" if existing.get("default_commission") is None else str(
            existing.get("default_commission")),
        key=f"comm_{merchant}_{net_id}",
    )

with c:
    allow_under_5 = st.checkbox(
        f"Allow < {MIN_COMMISSION:g}% import",
        value=bool(existing.get("allow_under_5", False)),
        key=f"allow_{merchant}_{net_id}",
    )
    associated_sovrn = st.checkbox(
        "Associated with Sovrn",
        value=bool(existing.get("associated_sovrn", False)),
        key=f"sovrn_{merchant}_{net_id}",
    )

    if st.button(f"Save profile for {merchant}", key=f"save_{merchant}_{net_id}"):
        st.session_state.merchant_profiles[key] = {
            "merchant_id": str(merch_id).strip(),
            "default_commission": safe_float(default_comm),
            "allow_under_5": bool(allow_under_5),
            "associated_sovrn": bool(associated_sovrn),  # NEW
        }
        st.success("Saved.")
        st.rerun()

st.divider()


# =============================
# 3) AI Category
# =============================
st.subheader("3) AI Category (quality gates + rank)")

if st.button("Run AI Category", type="primary"):
    with st.spinner("Running category + gates + ranking…"):
        st.session_state.ai_df = apply_quality_gates_and_rank(scoped)
        reset_downstream(after="ai")
    st.success("AI Category complete.")
    st.rerun()

if st.session_state.ai_df is None:
    st.info("Click **Run AI Category** to continue.")
    st.stop()

ai = st.session_state.ai_df.copy()
ready = ai[ai["status"] == "READY"].copy()
blocked = ai[ai["status"] == "BLOCKED"].copy()

st.caption(
    f"READY vs BLOCKED: {len(ready)}/{len(ai)} READY • {len(blocked)}/{len(ai)} BLOCKED")

st.dataframe(
    ai[[
        "product_id", "title", "merchant", "network_id",
        "brand_ai", "ai_category", "commission_resolved",
        "status", "reason", "url"
    ]],
    use_container_width=True,
    hide_index=True,
    column_config={"url": st.column_config.LinkColumn("URL")},
)

st.divider()


# =============================
# 4) Create Collection (one collection)
# =============================
st.subheader("4) Create Collection")

if ready.empty:
    st.error("No READY items to publish. Fix merchant readiness (merchant ID / commission) or adjust policy override.")
    st.stop()

# Defaults for naming
unique_merchants = sorted(ready["merchant"].astype(
    str).str.strip().unique().tolist())
default_merchant_name = unique_merchants[0] if len(
    unique_merchants) == 1 else "Merchant Collection"

cats_ready = sorted(ready["ai_category"].astype(str).unique().tolist())
default_category_name = cats_ready[0] if len(
    cats_ready) == 1 else "Category Collection"

st.caption("Choose one collection style for this run (no sub-collections).")

b1, b2 = st.columns([1, 1])
with b1:
    if st.button("Group everything under Brand (Merchant)", type="primary"):
        st.session_state.collection_mode = "brand"
        st.session_state.collection_name = default_merchant_name
        st.session_state.collection_hero_bytes = None
        reset_downstream(after="collection")
        st.rerun()

with b2:
    if st.button("Group everything under Category (AI)", type="primary"):
        st.session_state.collection_mode = "category"
        st.session_state.collection_name = default_category_name
        st.session_state.collection_hero_bytes = None
        reset_downstream(after="collection")
        st.rerun()

if st.session_state.collection_mode is None:
    st.info("Pick a grouping option to continue.")
    st.stop()

st.divider()

st.caption("Collection details")
st.session_state.collection_name = st.text_input(
    "Collection name (editable)",
    value=st.session_state.collection_name or "Collection",
)

hero = st.file_uploader("Upload hero image (optional)",
                        type=["png", "jpg", "jpeg", "webp"])
if hero is not None:
    st.session_state.collection_hero_bytes = hero.getvalue()

st.caption(f"Included products: {len(ready)}")
st.dataframe(
    ready[["product_id", "title", "merchant", "ai_category",
           "commission_resolved", "rank_score", "url"]],
    use_container_width=True,
    hide_index=True,
    column_config={"url": st.column_config.LinkColumn("URL")},
)

st.divider()


# =============================
# 5) Publish + Output
# =============================
st.subheader("5) Publish + Output")

default_rule = (st.session_state.run_name or "demo_run").replace(
    "•", "-").replace(" ", "").strip("-")
rule_name = st.text_input(
    "Rule name (for Product Import Automation output)", value=default_rule or "Alo_test")

if st.button("Publish", type="primary"):
    with st.spinner("Publishing…"):
        pub = ready.copy()
        pub["collection_name_final"] = (
            st.session_state.collection_name or "Collection").strip()
        st.session_state.published_df = pub.reset_index(drop=True)
        st.session_state.published_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        created = date.today().strftime("%d %b %Y")
        pia_row = {
            "Rule Name": (rule_name.strip() or "Untitled"),
            "Filter Summary": format_filter_summary(),
            "Max Products": int(len(pub)),
            "Created / Last Modified date": created,
            "Status": "ACTIVE",
            "Actions": "…",
        }
        st.session_state.rules_table.append(pia_row)

    st.success(f"Published {len(st.session_state.published_df)} products.")
    st.rerun()

if st.session_state.published_df is None:
    st.info("Publish becomes available after the collection is created.")
    st.stop()

published = st.session_state.published_df.copy()
st.caption(
    f"Published products: {len(published)} • Collection: {published['collection_name_final'].iloc[0]}")

st.dataframe(
    published[[
        "product_id", "title", "merchant", "network_id",
        "ai_category", "commission_resolved", "collection_name_final", "url"
    ]],
    use_container_width=True,
    hide_index=True,
    column_config={"url": st.column_config.LinkColumn("URL")},
)

st.divider()

st.subheader("Output → Product Import Automation")

rules_df = pd.DataFrame(st.session_state.rules_table)
if rules_df.empty:
    st.info("No rules yet.")
else:
    st.dataframe(rules_df, use_container_width=True, hide_index=True)
