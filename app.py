# ============================================================
# PHISHING URL BEHAVIOUR CLASSIFICATION
# Advanced Cybersecurity Streamlit Application
# ============================================================

import os
import re
from urllib.parse import urlparse

import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CyberGuard // Phishing URL Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CYBERSECURITY CSS STYLING
# ============================================================

st.markdown(
    """
    <style>
    /* Dark Cybersecurity Theme Colors */
    :root {
        --bg-main: #0b0f19;
        --bg-card: #131b2e;
        --accent-cyan: #00f0ff;
        --accent-green: #00e676;
        --accent-red: #ff1744;
        --accent-yellow: #ffc107;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --border-color: #1e293b;
    }

    /* Main background */
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Custom Glassmorphism Containers */
    .cyber-card {
        background: rgba(19, 27, 46, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 240, 255, 0.15);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .cyber-card-danger {
        border-left: 5px solid var(--accent-red);
        background: linear-gradient(135deg, rgba(255, 23, 68, 0.08) 0%, rgba(19, 27, 46, 0.8) 100%);
    }

    .cyber-card-safe {
        border-left: 5px solid var(--accent-green);
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.08) 0%, rgba(19, 27, 46, 0.8) 100%);
    }

    /* Custom Hero Header */
    .hero-container {
        text-align: center;
        padding: 10px 0 5px 0;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #00f0ff 0%, #7000ff 50%, #00e676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: var(--text-secondary);
        margin-bottom: 15px;
    }

    /* Metric Badges */
    .status-badge-phishing {
        display: inline-block;
        padding: 6px 16px;
        background-color: rgba(255, 23, 68, 0.18);
        color: #ff5252;
        border: 1px solid var(--accent-red);
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .status-badge-safe {
        display: inline-block;
        padding: 6px 16px;
        background-color: rgba(0, 230, 118, 0.18);
        color: #69f0ae;
        border: 1px solid var(--accent-green);
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* Custom Stat Cards */
    .stat-card {
        background: #101726;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--accent-cyan);
    }
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Threat Indicator Tag */
    .threat-tag {
        display: inline-block;
        background: rgba(255, 23, 68, 0.12);
        color: #ff8a80;
        border: 1px solid rgba(255, 23, 68, 0.3);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.85rem;
        margin: 3px;
    }
    .safe-tag {
        display: inline-block;
        background: rgba(0, 230, 118, 0.12);
        color: #b9f6ca;
        border: 1px solid rgba(0, 230, 118, 0.3);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.85rem;
        margin: 3px;
    }

    /* Sidebar Custom Styling */
    [data-testid="stSidebar"] {
        background-color: #0e1422;
        border-right: 1px solid #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/phishing_url_model.pkl"
ALT_MODEL_PATH = "models/phishing_url_model_evaluated.pkl"

@st.cache_resource
def load_phishing_model():
    for path in [MODEL_PATH, ALT_MODEL_PATH]:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model
            except Exception:
                pass
    return None

model = load_phishing_model()

if model is None:
    st.error(
        f"🚨 **Unable to load trained model** from `{MODEL_PATH}` or `{ALT_MODEL_PATH}`. "
        "Please ensure `analysis.py` has been executed to generate the model file."
    )
    st.stop()

# ============================================================
# URL FEATURE EXTRACTION FUNCTION
# Must strictly match analysis.py feature schema
# ============================================================

def extract_url_features(url):
    url = str(url).strip()

    # Add scheme if missing
    if re.match(r"^[a-zA-Z]+://", url):
        parsed_url = urlparse(url)
    else:
        parsed_url = urlparse("http://" + url)

    hostname = parsed_url.hostname or ""
    path = parsed_url.path or ""

    # Basic URL measurements
    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)

    # Character counts
    digit_count = sum(char.isdigit() for char in url)
    special_char_count = sum(not char.isalnum() for char in url)

    dot_count = url.count(".")
    hyphen_count = url.count("-")
    slash_count = url.count("/")
    at_count = url.count("@")
    question_count = url.count("?")
    equal_count = url.count("=")
    ampersand_count = url.count("&")
    percent_count = url.count("%")

    # Ratios
    digit_ratio = digit_count / url_length if url_length > 0 else 0
    special_char_ratio = special_char_count / url_length if url_length > 0 else 0

    # Subdomain count
    hostname_parts = [part for part in hostname.split(".") if part]
    if len(hostname_parts) > 2:
        subdomain_count = len(hostname_parts) - 2
    else:
        subdomain_count = 0

    # HTTPS check
    has_https = int(parsed_url.scheme.lower() == "https")

    # IP address detection
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    has_ip_address = int(bool(re.match(ip_pattern, hostname)))

    # Suspicious words
    suspicious_words = [
        "login", "signin", "verify", "verification", "secure",
        "account", "update", "confirm", "password", "bank",
        "billing", "payment", "authenticate", "wallet", "credential"
    ]
    url_lower = url.lower()
    suspicious_word_count = sum(word in url_lower for word in suspicious_words)

    # Punycode
    has_punycode = int("xn--" in hostname.lower())

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "digit_count": digit_count,
        "digit_ratio": digit_ratio,
        "special_char_count": special_char_count,
        "special_char_ratio": special_char_ratio,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "slash_count": slash_count,
        "at_count": at_count,
        "question_count": question_count,
        "equal_count": equal_count,
        "ampersand_count": ampersand_count,
        "percent_count": percent_count,
        "subdomain_count": subdomain_count,
        "has_https": has_https,
        "has_ip_address": has_ip_address,
        "suspicious_word_count": suspicious_word_count,
        "has_punycode": has_punycode
    }

EXPECTED_FEATURES = [
    "url_length", "hostname_length", "path_length", "digit_count",
    "digit_ratio", "special_char_count", "special_char_ratio",
    "dot_count", "hyphen_count", "slash_count", "at_count",
    "question_count", "equal_count", "ampersand_count", "percent_count",
    "subdomain_count", "has_https", "has_ip_address",
    "suspicious_word_count", "has_punycode"
]

def normalize_user_url(url_str):
    url_str = str(url_str).strip()
    if not re.match(r"^[a-zA-Z]+://", url_str):
        url_str = "https://" + url_str
    
    parsed = urlparse(url_str)
    hostname = parsed.hostname or ""
    is_ip = bool(re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", hostname))
    
    if not is_ip and hostname.count(".") == 1 and not hostname.startswith("www."):
        new_hostname = "www." + hostname
        url_str = url_str.replace(hostname, new_hostname, 1)
        
    return url_str

def validate_url(url):
    url = url.strip()
    if not url:
        return False, "Please enter a URL to analyze."
    if len(url) < 4:
        return False, "URL string is too short."
    test_url = url if re.match(r"^[a-zA-Z]+://", url) else "https://" + url
    try:
        parsed = urlparse(test_url)
        if not parsed.netloc:
            return False, "Invalid URL structure. Missing network domain."
        return True, ""
    except Exception:
        return False, "Malformed URL format."

def analyze_single_url(url):
    raw_url = url.strip()
    normalized_url = normalize_user_url(raw_url)
    features = extract_url_features(normalized_url)
    feature_df = pd.DataFrame([features])[EXPECTED_FEATURES]

    prediction = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]

    return {
        "url": raw_url,
        "normalized_url": normalized_url,
        "features": features,
        "feature_df": feature_df,
        "prediction": prediction,
        "legitimate_prob": probabilities[0],
        "phishing_prob": probabilities[1]
    }

# ============================================================
# SIDEBAR NAVIGATION & MODES
# ============================================================

with st.sidebar:
    st.markdown("## 🛡️ CyberGuard")
    st.markdown("### Threat Intelligence Engine")
    
    app_mode = st.radio(
        "Select Inspection State:",
        [
            "🛡️ Single URL Inspector",
            "📁 Batch Scanner"
        ],
        index=0
    )

# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🛡️ CYBERGUARD // PHISHING URL DETECTOR</div>
        <div class="hero-subtitle">
            Zero-Cost Machine Learning Threat Classifier • Structural & Lexical URL Behaviour Intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# STATE 1: SINGLE URL INSPECTOR
# ============================================================

if app_mode == "🛡️ Single URL Inspector":
    st.markdown("### 🔎 Analyze Web Address")

    url_input = st.text_input(
        "Target URL",
        value="",
        placeholder="e.g. https://secure-bank.example.com/login",
        help="Enter any complete or standard web address to extract structural indicators."
    )

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        analyze_click = st.button("⚡ Run Threat Inspection", type="primary", use_container_width=True)

    if url_input.strip():
        valid, err_msg = validate_url(url_input)

        if not valid:
            st.error(f"❌ {err_msg}")
        else:
            res = analyze_single_url(url_input)
            features = res["features"]
            phish_prob = res["phishing_prob"]
            legit_prob = res["legitimate_prob"]
            is_phishing = res["prediction"] == 1

            st.markdown("<br>", unsafe_allow_html=True)

            # High Level Result Card
            if is_phishing:
                st.markdown(
                    f"""
                    <div class="cyber-card cyber-card-danger">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="status-badge-phishing">🚨 POTENTIALLY PHISHING URL</span>
                                <h2 style="color: #ff5252; margin-top: 10px; margin-bottom: 5px;">High Risk Assessment Detected</h2>
                                <p style="color: #cbd5e1; margin-bottom: 0;">This URL exhibits structural patterns strongly aligned with malicious phishing targets.</p>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 2.5rem; font-weight: 800; color: #ff5252;">{phish_prob*100:.1f}%</div>
                                <div style="font-size: 0.85rem; color: #94a3b8;">Threat Confidence</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="cyber-card cyber-card-safe">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="status-badge-safe">✅ LIKELY LEGITIMATE URL</span>
                                <h2 style="color: #69f0ae; margin-top: 10px; margin-bottom: 5px;">Clean Structural Evaluation</h2>
                                <p style="color: #cbd5e1; margin-bottom: 0;">This URL matches typical structural characteristics of legitimate internet domains.</p>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 2.5rem; font-weight: 800; color: #69f0ae;">{legit_prob*100:.1f}%</div>
                                <div style="font-size: 0.85rem; color: #94a3b8;">Legitimate Confidence</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Visual Risk Score Gauge Bar
            st.markdown("#### 🎯 Threat Risk Meter")
            st.progress(float(phish_prob))

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.caption(f"**Legitimate Score:** {legit_prob*100:.2f}%")
            with col_g2:
                st.caption(f"**Phishing Risk Score:** {phish_prob*100:.2f}%")

            # Threat Signals & Red Flags
            st.markdown("#### 🚩 Structural Feature Signal Analysis")
            
            threat_flags = []
            safe_flags = []

            if features["has_ip_address"]:
                threat_flags.append("Raw IP Hostname detected (bypasses domain name system)")
            else:
                safe_flags.append("Standard domain hostname (No IP address)")

            if features["subdomain_count"] > 2:
                threat_flags.append(f"Excessive subdomains ({features['subdomain_count']} detected)")
            else:
                safe_flags.append(f"Normal subdomain depth ({features['subdomain_count']})")

            if features["suspicious_word_count"] > 0:
                threat_flags.append(f"Contains security/sensitive keywords ({features['suspicious_word_count']} matched)")
            else:
                safe_flags.append("No common phishing keywords detected")

            if features["at_count"] > 0:
                threat_flags.append("Contains '@' symbol (URL user-info redirection technique)")

            if features["has_punycode"]:
                threat_flags.append("Punycode homograph attack pattern detected ('xn--')")

            if features["url_length"] > 75:
                threat_flags.append(f"Elevated URL length ({features['url_length']} characters)")
            elif features["url_length"] < 40:
                safe_flags.append(f"Compact URL length ({features['url_length']} characters)")

            if features["digit_ratio"] > 0.15:
                threat_flags.append(f"High digit ratio ({features['digit_ratio']*100:.1f}% digits)")

            col_fl1, col_fl2 = st.columns(2)
            with col_fl1:
                st.markdown("**🚨 Suspicious Signals:**")
                if threat_flags:
                    for tf in threat_flags:
                        st.markdown(f'<span class="threat-tag">⚠️ {tf}</span>', unsafe_allow_html=True)
                else:
                    st.info("No high-risk structural anomalies detected.")

            with col_fl2:
                st.markdown("**🛡️ Reassuring Indicators:**")
                if safe_flags:
                    for sf in safe_flags:
                        st.markdown(f'<span class="safe-tag">✓ {sf}</span>', unsafe_allow_html=True)
                else:
                    st.caption("Multiple structural risk factors observed.")

            # Comprehensive Feature Breakdown Table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Extracted URL Feature Vector")
            
            feat_table = pd.DataFrame({
                "Feature Characteristic": [
                    "URL Total Length", "Hostname Length", "Path Length",
                    "Digit Count", "Digit Ratio", "Special Character Count",
                    "Special Character Ratio", "Dot (.) Count", "Hyphen (-) Count",
                    "Slash (/) Count", "At Symbol (@) Count", "Question (?) Count",
                    "Equals (=) Count", "Ampersand (&) Count", "Percent (%) Count",
                    "Subdomain Count", "HTTPS Protocol", "IP Hostname",
                    "Suspicious Word Matches", "Punycode ('xn--')"
                ],
                "Extracted Value": [
                    str(features["url_length"]), str(features["hostname_length"]), str(features["path_length"]),
                    str(features["digit_count"]), f"{features['digit_ratio']:.4f}", str(features["special_char_count"]),
                    f"{features['special_char_ratio']:.4f}", str(features["dot_count"]), str(features["hyphen_count"]),
                    str(features["slash_count"]), str(features["at_count"]), str(features["question_count"]),
                    str(features["equal_count"]), str(features["ampersand_count"]), str(features["percent_count"]),
                    str(features["subdomain_count"]), "Yes (1)" if features["has_https"] else "No (0)",
                    "Yes (1)" if features["has_ip_address"] else "No (0)",
                    str(features["suspicious_word_count"]), "Yes (1)" if features["has_punycode"] else "No (0)"
                ]
            })

            st.dataframe(feat_table, use_container_width=True, hide_index=True)

# ============================================================
# STATE 2: BATCH URL SCANNER
# ============================================================

elif app_mode == "📁 Batch Scanner":
    st.markdown("### 📁 Batch URL Threat Scanner")
    st.markdown("Scan multiple URLs simultaneously by pasting a list or uploading a text/CSV file.")

    batch_input = st.text_area(
        "Paste URLs (one per line)",
        height=150,
        placeholder="https://google.com\nhttp://192.168.1.1/login\nhttps://github.com\nhttp://paypal.verify.account-security.info"
    )

    uploaded_file = st.file_uploader("Or upload CSV / TXT file with URLs", type=["csv", "txt"])

    urls_to_scan = []
    if batch_input.strip():
        urls_to_scan.extend([line.strip() for line in batch_input.splitlines() if line.strip()])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded_file)
                col_name = [c for c in df_up.columns if "url" in c.lower()]
                if col_name:
                    urls_to_scan.extend(df_up[col_name[0]].astype(str).tolist())
                else:
                    urls_to_scan.extend(df_up.iloc[:, 0].astype(str).tolist())
            else:
                content = uploaded_file.read().decode("utf-8")
                urls_to_scan.extend([line.strip() for line in content.splitlines() if line.strip()])
        except Exception as e:
            st.error(f"Error parsing file: {e}")

    if urls_to_scan:
        st.markdown(f"**Found {len(urls_to_scan)} URLs to analyze.**")
        if st.button("🚀 Execute Batch Scan", type="primary"):
            scan_results = []
            progress_bar = st.progress(0.0)

            for i, target_url in enumerate(urls_to_scan):
                valid, _ = validate_url(target_url)
                if not valid:
                    scan_results.append({
                        "URL": target_url,
                        "Status": "Invalid URL",
                        "Phishing Probability": "N/A",
                        "Risk Category": "Error"
                    })
                else:
                    res = analyze_single_url(target_url)
                    p_prob = res["phishing_prob"]
                    is_p = res["prediction"] == 1
                    scan_results.append({
                        "URL": target_url,
                        "Status": "🚨 Phishing" if is_p else "✅ Legitimate",
                        "Phishing Probability": f"{p_prob*100:.2f}%",
                        "Risk Category": "HIGH RISK" if p_prob > 0.7 else ("MODERATE RISK" if p_prob > 0.3 else "SAFE")
                    })
                progress_bar.progress((i + 1) / len(urls_to_scan))

            res_df = pd.DataFrame(scan_results)
            st.markdown("#### 📋 Batch Inspection Results")
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            phish_cnt = sum(1 for r in scan_results if "Phishing" in r["Status"])
            legit_cnt = sum(1 for r in scan_results if "Legitimate" in r["Status"])
            
            c_b1, c_b2, c_b3 = st.columns(3)
            c_b1.metric("Total Scanned", len(scan_results))
            c_b2.metric("Legitimate URLs", legit_cnt)
            c_b3.metric("Phishing Flagged", phish_cnt)

            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV Report",
                data=csv,
                file_name="phishing_batch_scan_results.csv",
                mime="text/csv"
            )