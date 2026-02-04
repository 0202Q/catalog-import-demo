# streamlit_app.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# =============================
# Config
# =============================
APP_TITLE = "Catalog Import Demo"
IN_SCOPE_NETWORK_ID = 600
IN_SCOPE_NETWORK_NAME = "FlexOffers US"

DEFAULT_KEEP_LABELS = [
    "Apparel — Leggings",
    "Apparel — Intimates",
    "Apparel — Tops",
    "Other / Needs Review",
]

DEMO_CSV_PATH = "demo_products.csv"


# =============================
# Small helpers
# =============================
def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_int(x) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None


def _safe_float(x) -> Optional[float]:
    try:
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _top_level_category(label: str) -> str:
    # "Apparel — Leggings" -> "Apparel"
    if not isinstance(label, str) or label.strip() == "":
        return "Unknown"
    parts = [p.strip() for p in label.split("—")]
    return parts[0] if parts else label.strip()


def _dedupe_keep_first(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    if key_col not in df.columns:
        return df
    return df.drop_duplicates(subset=[key_col], keep="first").reset_index(drop=True)


def _unique_cols(cols: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _profile_key(merchant_name: str) -> str:
    return str(merchant_name or "").strip().lower()


# =============================
# Demo data (fallback)
# =============================
def _demo_fallback_df() -> pd.DataFrame:
    # NOTE: product_id should NOT encode reasons like "demo_low_commission".
    rows = [
        # Alo Yoga - good items
        (
            "8496513547418731001",
            "ALO | Airbrush High-Waist Heart Throb Legging in Anthracite/White Grey, Size: Large",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            8.0,
            "https://example.com/alo600/1",
        ),
        (
            "8496513547418731002",
            "ALO | Airlift High-Waist Elongated Legging in Black, Size: 2XS",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            8.0,
            "https://example.com/alo600/2",
        ),
        (
            "8496513547418731003",
            "ALO | Airlift High-Waist Legging in Navy, Size: Small",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            8.0,
            "https://example.com/alo600/3",
        ),
        (
            "8496513547418731004",
            "ALO | Airbrush High-Waist 7/8 Legging in Espresso, Size: Medium",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            8.0,
            "https://example.com/alo600/4",
        ),
        (
            "8496513547418731005",
            "ALO | Airlift High-Waist 7/8 Line Up Legging in Gravel, Size: Large",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            8.0,
            "https://example.com/alo600/5",
        ),
        (
            "8496513547418731006",
            "ALO | Airbrush High-Waist Legging - Sherpa Stitch, Size: Medium",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            8.0,
            "https://example.com/alo600/6",
        ),
        (
            "8496513547418731007",
            "ALO | Airlift High-Waist Legging - Micro Grid, Size: Small",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            8.0,
            "https://example.com/alo600/7",
        ),
        (
            "8496513547418731008",
            "ALO | Airbrush Real Bra in Black, Size: Medium",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            8.0,
            "https://example.com/alo600/8",
        ),
        (
            "8496513547418731009",
            "ALO | Seamless Ribbed Tank in White, Size: Small",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            8.0,
            "https://example.com/alo600/9",
        ),
        # Duplicate (dedupe should remove)
        (
            "8496513547418731096",
            "ALO | Airbrush High-Waist Heart Throb Legging in Anthracite/White Grey, Size: Large (Duplicate)",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            8.0,
            "https://example.com/alo600/1",
        ),
        # Missing commission percent (forces default commission to matter)
        (
            "8496513547418731100",
            "ALO | Demo Item (Missing Commission Percent)",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            None,
            "https://example.com/alo600/missing_commission",
        ),
        # Low commission (tests <5% rule)
        (
            "8496513547418731101",
            "ALO | Demo Item (Low Commission)",
            "Alo Yoga",
            600,
            IN_SCOPE_NETWORK_NAME,
            2.5,
            "https://example.com/alo600/low_commission",
        ),
        # Other merchants (should be removed for Alo demo)
        (
            "78720",
            "SKIMS T-Shirt Demi Bra | Deep Neutral | 34DDD | Wireless Form",
            "SKIMS",
            603,
            "Impact US",
            5.0,
            "https://example.com/skims603/1",
        ),
        (
            "78719",
            "SKIMS T-Shirt Demi Bra | Black | 40C | Wireless Form",
            "SKIMS",
            603,
            "Impact US",
            5.0,
            "https://example.com/skims603/2",
        ),
        ("90001", "REVOLVE | Something", "REVOLVE", 605,
         "Sovrn", 6.0, "https://example.com/revolve605/1"),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "product_id",
            "title",
            "merchant",
            "network_id",
            "network_name",
            "commission_percent",
            "deep_link",
        ],
    )


def load_demo_products() -> pd.DataFrame:
    if os.path.exists(DEMO_CSV_PATH):
        try:
            df = pd.read_csv(DEMO_CSV_PATH)
            # Standardize expected columns (ignore extras)
            col_map = {c.strip(): c for c in df.columns}

            def pick(*names):
                for n in names:
                    if n in col_map:
                        return col_map[n]
                return None

            p_id = pick("product_id", "Product ID", "id")
            title = pick("title", "name", "product_name")
            merchant = pick("merchant", "brand", "Merchant")
            n_id = pick("network_id", "network", "Network ID")
            n_name = pick("network_name", "Network Name")
            comm = pick("commission_percent", "commission", "Commission")
            deep = pick("deep_link", "url", "URL", "Direct URL")

            out = pd.DataFrame()
            out["product_id"] = df[p_id].astype(
                str) if p_id else df.index.astype(str)
            out["title"] = df[title].astype(str) if title else ""
            out["merchant"] = df[merchant].astype(str) if merchant else ""
            out["network_id"] = df[n_id] if n_id else IN_SCOPE_NETWORK_ID
            out["network_name"] = df[n_name].astype(
                str) if n_name else IN_SCOPE_NETWORK_NAME
            out["commission_percent"] = df[comm] if comm else None
            out["deep_link"] = df[deep].astype(str) if deep else ""
            return out
        except Exception:
            return _demo_fallback_df()
    return _demo_fallback_df()


# =============================
# AI classifier (mock)
# =============================
def ai_classify_taxonomy(title: str) -> Tuple[str, str, float, str]:
    """
    Returns: (taxonomy_key, taxonomy_label, confidence, reason)
    Mock "AI classifier" using simple keyword rules for demo.
    """
    t = (title or "").lower()

    if "legging" in t:
        return ("apparel_leggings", "Apparel — Leggings", 0.92, "matched keyword: legging")
    if any(k in t for k in ["bra", "brief", "underwear", "intimate", "thong"]):
        return ("apparel_intimates", "Apparel — Intimates", 0.88, "matched keyword: bra/intimates")
    if any(k in t for k in ["tank", "top", "tee", "t-shirt", "shirt"]):
        return ("apparel_tops", "Apparel — Tops", 0.82, "matched keyword: top/tank")
    return ("other_needs_review", "Other / Needs Review", 0.60, "fallback category")


# =============================
# Merchant profile
# =============================
@dataclass
class MerchantProfile:
    merchant_name: str
    merchant_id: Optional[str]
    network_id: int
    network_name: Optional[str]
    default_commission: Optional[float]
    allow_under_5: bool
    associated_with_sovrn: bool


def _get_profile(merchant_name: str) -> MerchantProfile:
    profiles: Dict[str, dict] = st.session_state.get("profiles", {})
    k = _profile_key(merchant_name)
    if k not in profiles:
        profiles[k] = {
            "merchant_name": merchant_name,
            "merchant_id": None,
            "network_id": IN_SCOPE_NETWORK_ID,
            "network_name": IN_SCOPE_NETWORK_NAME,
            "default_commission": None,
            "allow_under_5": False,
            "associated_with_sovrn": False,
        }
        st.session_state["profiles"] = profiles

    d = profiles[k]
    return MerchantProfile(
        merchant_name=d.get("merchant_name") or merchant_name,
        merchant_id=d.get("merchant_id"),
        network_id=_safe_int(d.get("network_id")) or IN_SCOPE_NETWORK_ID,
        network_name=d.get("network_name"),
        default_commission=_safe_float(d.get("default_commission")),
        allow_under_5=bool(d.get("allow_under_5", False)),
        associated_with_sovrn=bool(d.get("associated_with_sovrn", False)),
    )


def _save_profile(p: MerchantProfile) -> None:
    profiles: Dict[str, dict] = st.session_state.get("profiles", {})
    profiles[_profile_key(p.merchant_name)] = {
        "merchant_name": p.merchant_name,
        "merchant_id": p.merchant_id,
        "network_id": p.network_id,
        "network_name": p.network_name,
        "default_commission": p.default_commission,
        "allow_under_5": p.allow_under_5,
        "associated_with_sovrn": p.associated_with_sovrn,
    }
    st.session_state["profiles"] = profiles


def _profile_missing(p: MerchantProfile) -> bool:
    if p.merchant_id is None or str(p.merchant_id).strip() == "":
        return True
    if p.default_commission is None:
        return True
    if p.network_name is None or str(p.network_name).strip() == "":
        return True
    return False


# =============================
# Pipeline steps
# =============================
def step_import_fetch() -> None:
    st.header("1) Import + Fetch")

    parent_json = st.text_area(
        "Paste parent JSON (optional)",
        height=110,
        placeholder='{"page":1,"per_page":100,"network_ids":[3,600,603],"merchant_ids":[49651],"search":[{"field":"any","operator":"like","value":"legging"}]}',
        key="parent_json_text",
    )

    if st.button("Import + Fetch", key="btn_fetch"):
        if parent_json.strip():
            try:
                json.loads(parent_json)
            except Exception:
                st.warning("Parent JSON is not valid JSON (ignored for demo).")

        df = load_demo_products()
        df["network_id"] = df["network_id"].apply(lambda x: _safe_int(x) or x)
        df["commission_percent"] = df["commission_percent"].apply(_safe_float)
        df["deep_link"] = df["deep_link"].astype(str)

        st.session_state["raw_df"] = df
        st.session_state["clean_df"] = None
        st.session_state["ai_df"] = None
        st.session_state["published"] = None

    raw_df = st.session_state.get("raw_df")
    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        networks_found = raw_df["network_id"].dropna().astype(
            int).nunique() if "network_id" in raw_df.columns else 0
        st.caption(
            f"Fetched: {len(raw_df)} products • Networks: {networks_found} • In-scope network: {IN_SCOPE_NETWORK_ID}")

        with st.expander("View raw queue (before cleanup)", expanded=False):
            display_cols = _unique_cols(
                ["product_id", "title", "merchant", "network_id",
                    "network_name", "commission_percent", "deep_link"]
            )
            st.dataframe(raw_df[display_cols],
                         use_container_width=True, hide_index=True)


def step_cleanup() -> None:
    st.header("2) Cleanup (Policy + Dedupe)")

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.info("Run Import + Fetch first.")
        return

    merchants = sorted(
        raw_df["merchant"].dropna().astype(str).unique().tolist())
    if not merchants:
        st.warning("No merchants found in fetched data.")
        return

    if "selected_merchant" not in st.session_state:
        st.session_state["selected_merchant"] = "Alo Yoga" if "Alo Yoga" in merchants else merchants[0]

    left, right = st.columns([2, 1], gap="large")

    with left:
        selected_merchant = st.selectbox(
            "Merchant to import (demo)",
            options=merchants,
            key="selected_merchant",
            help="For this demo, we keep only the selected merchant.",
        )

        keep_labels = st.multiselect(
            "Keep only these taxonomy labels (hard remove everything else)",
            options=DEFAULT_KEEP_LABELS,
            default=DEFAULT_KEEP_LABELS,
            key="keep_taxonomy_labels",
        )

    with right:
        st.markdown("**Fixed policy**")
        st.caption(f"• Keep Network {IN_SCOPE_NETWORK_ID} only")
        st.caption(f"• Keep Merchant “{selected_merchant}” only")

        st.session_state.setdefault("enable_smart_cleanup", True)

        enable_smart_cleanup = st.checkbox(
            "Enable smart cleanup",
            key="enable_smart_cleanup",
        )

    if st.button("Cleanup", key="btn_cleanup"):
        df = raw_df.copy()

        df = df[df["network_id"].astype(int) == IN_SCOPE_NETWORK_ID].copy()
        df = df[df["merchant"].astype(str) == str(selected_merchant)].copy()

        if enable_smart_cleanup:
            df = _dedupe_keep_first(df, "deep_link")

        tax = df["title"].apply(ai_classify_taxonomy)
        df["taxonomy_key"] = tax.apply(lambda x: x[0])
        df["taxonomy_label"] = tax.apply(lambda x: x[1])
        df["ai_confidence"] = tax.apply(lambda x: x[2])
        df["ai_reason"] = tax.apply(lambda x: x[3])

        if keep_labels:
            df = df[df["taxonomy_label"].isin(keep_labels)].copy()

        df = df.reset_index(drop=True)

        st.session_state["clean_df"] = df
        st.session_state["ai_df"] = None
        st.session_state["published"] = None

    clean_df = st.session_state.get("clean_df")
    if isinstance(clean_df, pd.DataFrame):
        st.caption(f"Queue after cleanup: {len(clean_df)} items")
        display_cols = _unique_cols(
            ["product_id", "title", "merchant", "network_id",
                "network_name", "commission_percent", "deep_link"]
        )
        st.dataframe(clean_df[display_cols],
                     use_container_width=True, hide_index=True)


def _apply_gating(ai_df: pd.DataFrame, profile: MerchantProfile) -> pd.DataFrame:
    out = ai_df.copy()

    resolved: List[Optional[float]] = []
    status: List[str] = []
    reasons: List[str] = []

    for _, row in out.iterrows():
        row_comm = _safe_float(row.get("commission_percent"))
        use_comm = row_comm if row_comm is not None else profile.default_commission
        resolved.append(use_comm)

        missing_profile = _profile_missing(profile)
        below_5 = (use_comm is not None) and (
            use_comm < 5.0) and (not profile.allow_under_5)

        if missing_profile:
            status.append("BLOCKED")
            reasons.append("missing_profile")
        elif below_5:
            status.append("BLOCKED")
            reasons.append("commission_below_5")
        else:
            status.append("READY")
            reasons.append("ok")

    out["commission_resolved"] = resolved
    out["status"] = status
    out["gate_reason"] = reasons
    return out


def step_ai_classify() -> None:
    st.header("3) AI Classify (Taxonomy for grouping)")

    clean_df = st.session_state.get("clean_df")
    if clean_df is None:
        st.info("Run Cleanup first.")
        return

    selected_merchant = st.session_state.get("selected_merchant")
    if not selected_merchant:
        st.warning("Select a merchant in Cleanup first.")
        return

    prof = _get_profile(selected_merchant)

    if st.button("Run AI Classify", key="btn_ai"):
        ai_df = clean_df.copy()

        if "taxonomy_label" not in ai_df.columns:
            tax = ai_df["title"].apply(ai_classify_taxonomy)
            ai_df["taxonomy_key"] = tax.apply(lambda x: x[0])
            ai_df["taxonomy_label"] = tax.apply(lambda x: x[1])
            ai_df["ai_confidence"] = tax.apply(lambda x: x[2])
            ai_df["ai_reason"] = tax.apply(lambda x: x[3])

        ai_df = _apply_gating(ai_df, prof)

        st.session_state["ai_df"] = ai_df
        st.session_state["published"] = None

    ai_df = st.session_state.get("ai_df")
    if not isinstance(ai_df, pd.DataFrame):
        st.caption(
            "AI Classify assigns canonical taxonomy labels used later for grouping.")
        return

    ready = int((ai_df["status"] == "READY").sum())
    blocked = int((ai_df["status"] == "BLOCKED").sum())

    c1, c2, c3 = st.columns([1, 1, 2], gap="large")
    c1.metric("READY", ready)
    c2.metric("BLOCKED", blocked)

    # ---- FIXED: no duplicate 'count' columns across pandas versions ----
    reason_counts = (
        ai_df.loc[ai_df["status"] == "BLOCKED", "gate_reason"]
        .value_counts(dropna=False)
        .rename_axis("gate_reason")
        .reset_index(name="count")
    )

    with c3:
        if len(reason_counts) == 0:
            st.caption("No blocked items.")
        else:
            st.dataframe(reason_counts, use_container_width=True,
                         hide_index=True)

    display_cols = _unique_cols(
        [
            "product_id",
            "title",
            "merchant",
            "network_id",
            "taxonomy_label",
            "ai_confidence",
            "commission_resolved",
            "status",
            "gate_reason",
            "deep_link",
        ]
    )
    st.dataframe(ai_df[display_cols],
                 use_container_width=True, hide_index=True)


def step_merchant_profile() -> None:
    st.header("4) Merchant Profile (only if needed)")

    ai_df = st.session_state.get("ai_df")
    if ai_df is None:
        st.info("Run AI Classify first.")
        return

    selected_merchant = st.session_state.get("selected_merchant")
    if not selected_merchant:
        st.warning("Select a merchant in Cleanup first.")
        return

    prof = _get_profile(selected_merchant)

    needs_profile = bool((ai_df["gate_reason"] == "missing_profile").any())
    if needs_profile:
        st.warning("Merchant Profile required → complete this step to continue.")
    else:
        st.caption("No profile updates required for this run.")

    with st.expander(
        f"Merchant: {selected_merchant} | Network: {prof.network_id} — {prof.network_name or ''}",
        expanded=True,
    ):
        with st.form(key=f"profile_form_{_profile_key(selected_merchant)}"):
            merchant_id = st.text_input(
                "Merchant ID",
                value="" if prof.merchant_id is None else str(
                    prof.merchant_id),
                placeholder="e.g., 49651",
            )

            n1, n2 = st.columns([1, 2], gap="large")
            with n1:
                network_id = st.number_input(
                    "Network ID",
                    min_value=0,
                    value=int(prof.network_id),
                    step=1,
                )
            with n2:
                network_name = st.text_input(
                    "Network",
                    value="" if prof.network_name is None else str(
                        prof.network_name),
                    placeholder="e.g., FlexOffers US",
                )

            default_commission = st.text_input(
                "Default commission %",
                value="" if prof.default_commission is None else str(
                    prof.default_commission),
                placeholder="e.g., 8.0",
            )

            allow_under_5 = st.checkbox(
                "Allow <5% import (policy override)",
                value=bool(prof.allow_under_5),
            )
            associated = st.checkbox(
                "Associated with Sovrn",
                value=bool(prof.associated_with_sovrn),
            )

            saved = st.form_submit_button("Save profile")

        if saved:
            mid_val = merchant_id.strip() or None
            net_name_val = network_name.strip() or None
            comm_val = _safe_float(default_commission)

            new_prof = MerchantProfile(
                merchant_name=selected_merchant,
                merchant_id=mid_val,
                network_id=int(network_id),
                network_name=net_name_val,
                default_commission=comm_val,
                allow_under_5=bool(allow_under_5),
                associated_with_sovrn=bool(associated),
            )
            _save_profile(new_prof)

            st.session_state["ai_df"] = _apply_gating(
                st.session_state["ai_df"], new_prof)

            st.success("Profile saved.")
            st.rerun()


def step_create_collection() -> None:
    st.header("5) Create Collection")

    ai_df = st.session_state.get("ai_df")
    if ai_df is None:
        st.info("Run AI Classify first.")
        return

    selected_merchant = st.session_state.get("selected_merchant") or "Merchant"
    ready_df = ai_df[ai_df["status"] == "READY"].copy()

    if ready_df.empty:
        st.warning(
            "No READY products yet. If items are blocked due to missing_profile, complete Step 4.")
        return

    st.session_state.setdefault("grouping_mode", "brand")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        if st.button("Group everything under Brand (Merchant)", key="btn_group_brand"):
            st.session_state["grouping_mode"] = "brand"
            st.session_state["collection_name"] = selected_merchant
    with c2:
        if st.button("Group everything under Category (AI)", key="btn_group_cat"):
            st.session_state["grouping_mode"] = "category"

    mode = st.session_state.get("grouping_mode", "brand")

    top_levels = ready_df["taxonomy_label"].apply(_top_level_category)
    collection_category = "Unknown" if top_levels.empty else str(
        top_levels.value_counts().index[0])

    st.session_state.setdefault("collection_name", selected_merchant)

    default_name = selected_merchant if mode in (
        "brand", "category") else selected_merchant
    if not st.session_state.get("collection_name"):
        st.session_state["collection_name"] = default_name

    st.text_input("Collection name (editable)", key="collection_name")

    st.file_uploader(
        "Upload hero image (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        key="hero_image",
        help="Optional for demo; not used for any processing.",
    )

    st.caption(
        f"READY products: {len(ready_df)} • Category (top-level): {collection_category}")

    st.session_state["collection_category"] = collection_category


def step_publish_output() -> None:
    st.header("6) Publish → Output")

    ai_df = st.session_state.get("ai_df")
    if ai_df is None:
        st.info("Run AI Classify first.")
        return

    selected_merchant = st.session_state.get("selected_merchant") or "Merchant"
    collection_name = st.session_state.get(
        "collection_name") or selected_merchant
    collection_category = st.session_state.get(
        "collection_category") or "Unknown"

    ready_df = ai_df[ai_df["status"] == "READY"].copy()
    blocked_df = ai_df[ai_df["status"] == "BLOCKED"].copy()

    if not blocked_df.empty:
        st.warning(
            "Some items are still BLOCKED. Complete Step 4 (Merchant Profile) or adjust the <5% policy override.")

    rule_default = f"{_profile_key(selected_merchant).replace(' ', '-')}-{IN_SCOPE_NETWORK_ID}"
    st.session_state.setdefault("rule_name", rule_default)

    st.text_input(
        "Rule name (for Product Import Automation output)", key="rule_name")

    if st.button("Publish", key="btn_publish"):
        published_at = _now_iso()

        overview = {
            "merchant": selected_merchant,
            "collection_name": collection_name,
            "collection_category": collection_category,
            "products": int(len(ready_df)),
            "rule": st.session_state.get("rule_name") or rule_default,
            "published_at": published_at,
        }

        st.session_state["published"] = {
            "overview": overview,
            "items": ready_df.copy(),
        }

    published = st.session_state.get("published")
    if not published:
        return

    ov = published["overview"]

    a, b, c, d = st.columns(4, gap="large")
    a.metric("Merchant", ov["merchant"])
    b.metric("Collection", ov["collection_name"])
    c.metric("Category", ov["collection_category"])
    d.metric("Products", ov["products"])

    st.caption(f"Rule: {ov['rule']} • Published at: {ov['published_at']}")

    st.subheader("Output (what was imported + rule created)")

    out_items: pd.DataFrame = published["items"]

    out = out_items.copy()
    out["collection_name_final"] = ov["collection_name"]
    out["collection_category_final"] = ov["collection_category"]

    display_cols = _unique_cols(
        [
            "product_id",
            "title",
            "merchant",
            "network_id",
            "taxonomy_label",
            "commission_resolved",
            "collection_name_final",
            "collection_category_final",
            "deep_link",
        ]
    )
    st.dataframe(out[display_cols], use_container_width=True, hide_index=True)


# =============================
# App init
# =============================
def init_state() -> None:
    st.session_state.setdefault("raw_df", None)
    st.session_state.setdefault("clean_df", None)
    st.session_state.setdefault("ai_df", None)
    st.session_state.setdefault("published", None)
    st.session_state.setdefault("profiles", {})


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    init_state()

    st.title(APP_TITLE)
    st.caption(
        "Import + Fetch → Cleanup (Policy + Dedupe) → AI Classify → Merchant Profile (if needed) → Create Collection → Publish → Output"
    )

    step_import_fetch()
    st.divider()

    step_cleanup()
    st.divider()

    step_ai_classify()
    st.divider()

    step_merchant_profile()
    st.divider()

    step_create_collection()
    st.divider()

    step_publish_output()


if __name__ == "__main__":
    main()
