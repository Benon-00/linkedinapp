import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import warnings
from datetime import datetime, time, timedelta

# --- 1. SUPPRESS WARNINGS ---
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Use relative path for cloud deployment
# Ensure the 'Saved_Models' folder is in the same directory as this script
MODEL_DIR = "Saved_Models"

# List of all targets and their VIRAL THRESHOLDS
THRESHOLDS = {
    'Video Views': 1000,
    'Watch Time (s)': 3600,
    'Average Watch Time (s)': 60,
    'Impression': 1000,
    'Members Reached': 700,
    'Page Viewers from post': 20,
    'Followers Gained From Post': 10,
    'Engagement Clicks': 200,
    'Engagement Rate': 0.12, # 12%
    'Clicks': 50,
    'Click Through Rate': 0.13, # 13%
    'Reactions': 150,
    'Comments': 50,
    'Reposts': 10
}

TARGET_COLS = list(THRESHOLDS.keys())

# Dropdown Options
CONTENT_TYPES = [
    "Text Only", "9:16 video + text", "16:9 video + text", "Text + Chart",
    "Carousel + text", "Poll + text", "Text + image", "Text + Image (Quote)", "Real-Photo"
]

# --- CACHED MODEL LOADER ---
@st.cache_resource
def load_all_models():
    """Loads all models into memory once at startup."""
    models = {}
    if not os.path.exists(MODEL_DIR):
        st.error(f"⚠️ Model directory '{MODEL_DIR}' not found. Please upload the 'Saved_Models' folder to your GitHub repository.")
        return {}

    for target in TARGET_COLS:
        safe_name = target.replace(' ', '_').replace('/', '')
        model_name = f"model_{safe_name}.pkl"
        path = os.path.join(MODEL_DIR, model_name)
        
        if os.path.exists(path):
            try:
                models[target] = joblib.load(path)
            except Exception: pass
    return models

# --- FEATURE ENGINE ---
def get_feature_dict(date_obj, day_str, time_obj, content_type, text_content):
    # 1. TIME FEATURES
    try:
        date_val = date_obj.toordinal()
        is_weekend = 1 if day_str in ['Saturday', 'Sunday'] else 0
        hour = time_obj.hour
        minute = time_obj.minute
        if hour < 12: phase = "Morning"
        elif hour < 18: phase = "Afternoon"
        else: phase = "Evening"
        time_numeric = hour + (minute/60.0)
    except:
        date_val, is_weekend, time_numeric, phase = 0, 0, 12.0, "Afternoon"

    # 2. CONTENT TYPE MAPPING
    type_flags = {
        'Is_Text': 0, 'Is_9:16 Video': 0, 'Is_16:9 Video': 0, 'Is_Chart': 0,
        'Is_Carousel': 0, 'Is_Poll': 0, 'Is_Image': 0, 'Is_Image (Quote)': 0, 'Is_Real-Photo': 0
    }
    mapping = {
        "Text Only": 'Is_Text', "9:16 video + text": 'Is_9:16 Video',
        "16:9 video + text": 'Is_16:9 Video', "Text + Chart": 'Is_Chart',
        "Carousel + text": 'Is_Carousel', "Poll + text": 'Is_Poll',
        "Text + image": 'Is_Image', "Text + Image (Quote)": 'Is_Image (Quote)',
        "Real-Photo": 'Is_Real-Photo'
    }
    if content_type in mapping: type_flags[mapping[content_type]] = 1

    # 3. NLP FEATURES
    text = str(text_content).strip()
    text_lower = text.lower()
    
    words = re.findall(r'\b\w+\b', text)
    sentences = [s for s in re.split(r'[.?!]+', text) if s.strip()]
    paragraphs = [p for p in text.split('\n') if p.strip()]
    
    word_count = len(words)
    sent_count = len(sentences)
    avg_wps = word_count / sent_count if sent_count else 0
    para_count = len(paragraphs)
    comp_mentions = sum(text_lower.count(c) for c in ['promptbi', 'delta 40', 'google'])
    q_count = text.count('?')
    
    has_named = 1 if any(n in text_lower for n in ['bertha', 'kasiera', 'benon']) else 0
    has_job = 1 if any(t in text_lower for t in ['ceo', 'founder', 'manager']) else 0
    has_cta = 1 if any(k in text_lower for k in ['link', 'bio', 'sign up', 'dm']) else 0
    has_trans = 1 if any(w in text_lower for w in ['before', 'after', 'transform']) else 0
    has_jargon = 1 if any(j in text_lower for j in ['sql', 'roi', 'kpi', 'dashboard']) else 0
    high_space = 1 if (para_count > 0 and (word_count/para_count) < 25) else 0
    
    first_sent = sentences[0].lower() if sentences else ""
    hook_quote = 1 if '"' in first_sent or '“' in first_sent else 0
    hook_prov = 1 if any(first_sent.startswith(w) for w in ['stop', 'don\'t', 'never']) else 0
    hook_cont = 1 if 'unpopular opinion' in first_sent else 0
    hook_story = 1 if any(w in first_sent for w in ['yesterday', 'i remember', 'once']) else 0
    hook_q = 1 if '?' in first_sent else 0
    
    p1 = 1 if re.search(r'\b(i|my|we)\b', text_lower) else 0
    p2 = 1 if re.search(r'\b(you|your)\b', text_lower) else 0
    p3 = 1 if re.search(r'\b(he|she|they)\b', text_lower) else 0
    p_mixed = 1 if (p1 + p2 + p3) > 1 else 0
    
    nums = len(re.findall(r'\d+', text))
    caps = len(re.findall(r'(?<!^)(?<!\. )\b[A-Z][a-z]+\b', text))
    spec_score = nums + caps
    spec_vague = 1 if spec_score < 2 else 0
    spec_mod = 1 if 2 <= spec_score <= 5 else 0
    spec_high = 1 if spec_score > 5 else 0

    return {
        'Date': date_val, 'Is_Weekend': is_weekend, 'Time': time_numeric, 'Time-Phases': phase,
        **type_flags,
        'Word_Count': word_count, 'Sentence_Count': sent_count,
        'Avg_Words_Per_Sentence': avg_wps,
        'Company_Mention_Count': comp_mentions,
        'Question_Count': q_count, 'Paragraph_Count': para_count,
        'Has_Named_Entity': has_named, 'Has_Job_Title': has_job,
        'Has_CTA': has_cta, 'Has_Before_After': has_trans,
        'Has_Jargon': has_jargon, 'High_Whitespace': high_space,
        'Hook_Is_Quote': hook_quote, 'Hook_Is_Provocative': hook_prov,
        'Hook_Is_Controversial': hook_cont, 'Hook_Is_Story': hook_story,
        'Hook_Is_Question': hook_q,
        'perspective_first_person': p1, 'perspective_second_person': p2,
        'perspective_third_person': p3, 'perspective_mixed': p_mixed,
        'vague': spec_vague, 'moderate': spec_mod, 'high_specificity': spec_high
    }

# --- BATCH PREDICTION ---
def predict_batch(input_df, models):
    results = {}
    if input_df.empty: return results

    FEATURE_COLS = [
        'Date', 'Is_Weekend', 'Time', 'Time-Phases',
        'Is_Text', 'Is_9:16 Video', 'Is_16:9 Video', 'Is_Chart', 
        'Is_Carousel', 'Is_Poll', 'Is_Image', 'Is_Image (Quote)', 'Is_Real-Photo',
        'Word_Count', 'Sentence_Count', 'Avg_Words_Per_Sentence', 
        'Company_Mention_Count', 'Question_Count', 'Paragraph_Count',
        'Has_Named_Entity', 'Has_Job_Title', 'Has_CTA', 
        'Has_Before_After', 'Has_Jargon', 'High_Whitespace',
        'Hook_Is_Quote', 'Hook_Is_Provocative', 'Hook_Is_Controversial', 
        'Hook_Is_Story', 'Hook_Is_Question',
        'perspective_first_person', 'perspective_second_person', 
        'perspective_third_person', 'perspective_mixed',
        'vague', 'moderate', 'high_specificity'
    ]
    
    input_df = input_df.reindex(columns=FEATURE_COLS, fill_value=0)

    for target, model in models.items():
        try:
            pred_log = model.predict(input_df)
            if len(pred_log) == 1:
                 results[target] = max(0, np.expm1(pred_log[0]))
            else:
                 results[target] = np.maximum(0, np.expm1(pred_log))
        except:
            if len(input_df) == 1: results[target] = 0
            else: results[target] = np.zeros(len(input_df))
            
    return results

# --- CONTENT MUTATION ENGINE ---
def optimize_content_and_time(base_date, base_day, base_time, base_type, base_content, current_results, models):
    
    best_config = None
    best_results = current_results
    baseline_met = sum(1 for k, v in THRESHOLDS.items() if current_results.get(k, 0) >= v)
    max_goals_met = baseline_met

    search_times = [9, 13, 17] 
    mutations = [
        ("", "Keep Current Content"),
        ("Stop scrolling! ", "Add Provocative Hook ('Stop...')"),
        ("Did you know? ", "Add Question Hook"),
        ("Yesterday, I realized... ", "Add Story Hook"),
        ("\"Success is not final.\" ", "Add Quote Hook"),
        (" Link in bio.", "Add CTA"),
        (" Before vs After. ", "Add Transformation"),
        (" ROI KPI SQL. ", "Add Industry Jargon"),
        ("\n\n\n", "Increase Whitespace"),
        (" I believe... ", "Shift to First Person"),
        (" You need this. ", "Shift to Second Person"),
        (" " + base_content, "Double Word Count")
    ]
    
    scenarios = [] 
    data_dicts = [] 
    
    for t_hour in search_times:
        for tweak_txt, tweak_label in mutations:
            if "Hook" in tweak_label: new_content = tweak_txt + base_content
            else: new_content = base_content + tweak_txt
            
            new_time = base_time.replace(hour=t_hour, minute=0)
            
            scenarios.append({
                'Time': f"{t_hour}:00",
                'Action': tweak_label,
                'Modified Content': new_content
            })
            d = get_feature_dict(base_date, base_day, new_time, base_type, new_content)
            data_dicts.append(d)

    if not data_dicts: return None, current_results
    
    batch_df = pd.DataFrame(data_dicts)
    batch_preds = predict_batch(batch_df, models)
    results_df = pd.DataFrame(batch_preds)
    
    goals_met_scores = np.zeros(len(results_df))
    for k, v in THRESHOLDS.items():
        if k in results_df.columns:
            goals_met_scores += (results_df[k] >= v).astype(int)
            
    best_idx = np.argmax(goals_met_scores)
    best_score = goals_met_scores[best_idx]
    
    if best_score > max_goals_met:
        max_goals_met = best_score
        is_better = True
    elif best_score == max_goals_met and max_goals_met > 0:
        curr_score = (current_results.get('Engagement Rate', 0) * 1000) + current_results.get('Impression', 0)
        new_score = (results_df.iloc[best_idx].get('Engagement Rate', 0) * 1000) + results_df.iloc[best_idx].get('Impression', 0)
        is_better = new_score > curr_score
    else:
        is_better = False

    if is_better:
        best_config = scenarios[best_idx]
        best_config['Goals Met'] = int(max_goals_met)
        best_config['Baseline Goals'] = baseline_met
        best_results = results_df.iloc[best_idx].to_dict()

    return best_config, best_results

# --- UI LAYOUT ---
st.set_page_config(page_title="LinkedIn Viral Predictor", layout="wide")

try:
    models = load_all_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    models = {}

st.title("🚀 LinkedIn Viral Predictor")
st.markdown("Optimization targets **Time** and **Content Structure** to hit **14 Success Thresholds**.")

with st.sidebar:
    st.header("1. Fixed Variables")
    d_date = st.date_input("Date", datetime.today())
    d_day = d_date.strftime("%A")
    st.info(f"Day: {d_day}")
    d_type = st.selectbox("Content Type", CONTENT_TYPES)
    
    st.header("2. Variables to Optimize")
    d_time = st.time_input("Time", time(9, 0))
    
    if "video" in d_type.lower():
        st.caption("📹 Video Content Detected")
        d_caption = st.text_area("Post Caption", height=100, placeholder="Text accompanying the video...")
        d_script = st.text_area("Video Script", height=150, placeholder="Spoken words or text overlays...")
        d_content = f"{d_caption}\n\n{d_script}"
    else:
        d_content = st.text_area("Actual Content", height=200, placeholder="Paste your post content here...")
    
    predict_btn = st.button("🔮 Predict Performance", type="primary")

if predict_btn:
    if not models:
        st.error("❌ Models not found! Please ensure the 'Saved_Models' folder is uploaded to your repository.")
    else:
        feat_dict = get_feature_dict(d_date, d_day, d_time, d_type, d_content)
        input_df = pd.DataFrame([feat_dict])
        results = predict_batch(input_df, models)
        
        st.subheader("🎯 Goal Status (Current Input)")
        met_count = 0
        cols = st.columns(4)
        col_idx = 0
        for metric, threshold in THRESHOLDS.items():
            val = results.get(metric, 0)
            is_met = val >= threshold
            if is_met: met_count += 1
            
            if 'Rate' in metric or 'Click Through' in metric:
                val_fmt = f"{val:.1%}"
                target_fmt = f"{threshold:.1%}"
            else:
                val_fmt = f"{val:,.0f}"
                target_fmt = f"{threshold:,.0f}"
                
            color = "green" if is_met else "red"
            icon = "✅" if is_met else "❌"
            
            with cols[col_idx % 4]:
                st.markdown(f"**{metric}**")
                st.markdown(f":{color}[{val_fmt}] / {target_fmt} {icon}")
            col_idx += 1
            
        st.progress(met_count / len(THRESHOLDS))

        st.markdown("---")
        st.subheader("🧪 Content Re-Writing Engine")
        
        with st.spinner("Simulating content mutations..."):
            config, opt_results = optimize_content_and_time(d_date, d_day, d_time, d_type, d_content, results, models)
        
        if config and config['Goals Met'] > config['Baseline Goals']:
            st.success(f"🎉 FOUND A BETTER CONFIGURATION! (Hits {config['Goals Met']}/14 Goals)")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("### 📝 Recommended Changes")
                st.write(f"**1. Time Shift:** {d_time.strftime('%H:%M')} ➝ **{config['Time']}**")
                st.write(f"**2. Content Tweak:** {config['Action']}")
                
                with st.expander("👀 Preview Re-Written Content Strategy"):
                    st.info(config['Modified Content'])
                    
            with c2:
                st.markdown("### 📈 Projected Impact")
                key_metrics = ['Video Views', 'Impression', 'Engagement Rate', 'Clicks']
                kc1, kc2 = st.columns(2)
                for idx, m in enumerate(key_metrics):
                    old = results.get(m, 0)
                    new = opt_results.get(m, 0)
                    delta = (new - old) / old if old > 0 else 0
                    with [kc1, kc2][idx % 2]:
                        if 'Rate' in m: st.metric(m, f"{new:.1%}", f"{delta:+.1%}")
                        else: st.metric(m, f"{new:,.0f}", f"{delta:+.1%}")
        elif met_count == len(THRESHOLDS):
            st.balloons()
            st.success("🌟 PERFECT SCORE! No changes needed.")
        else:
            st.warning("Current content is optimal for this Date/Format. Try changing the Format if possible.")
