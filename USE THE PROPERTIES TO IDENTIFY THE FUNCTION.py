import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import random

# ---------------------------------------------------------
# 1. إعداد الصفحة والتنسيقات (نفس التصميم السابق)
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Calculus Quiz: Sketching Functions")

st.markdown("""
<style>
    /* الاتجاه العام للتطبيق من اليمين لليسار */
    .stApp {
        direction: rtl; 
    }
    
    /* محاذاة النصوص */
    h1, h2, h3, p, div {
        text-align: right;
    }

    /* إجبار المعادلات الرياضية واللاتكس على الاتجاه من اليسار لليمين */
    .katex-display, .katex {
        direction: ltr;
        text-align: center;
    }
    
    /* محاذاة أزرار الاختيار */
    .stRadio > div {
        direction: rtl;
        text-align: right;
    }
    
    /* تنسيق صندوق السؤال */
    .question-box {
        background-color: #f1f3f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-right: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .question-text-ar {
        font-size: 20px;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 8px;
    }
    .question-text-en {
        font-size: 16px;
        color: #4b5563;
        font-family: sans-serif;
        direction: ltr;
        text-align: left;
    }
    
    /* توسيط الأزرار */
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. إدارة حالة الجلسة
# ---------------------------------------------------------
if 'q_index' not in st.session_state:
    st.session_state.q_index = 0

def next_question():
    if st.session_state.q_index < len(questions) - 1:
        st.session_state.q_index += 1

def prev_question():
    if st.session_state.q_index > 0:
        st.session_state.q_index -= 1

# ---------------------------------------------------------
# 3. دالة الرسم البياني (محسنة لتدعم الدوال متعددة التعريف)
# ---------------------------------------------------------
def plot_function(func, x_range=(-4, 6), y_range=(-4, 6), title="", has_asymptote_at=None):
    """
    رسم الدالة مع محاور إحداثية مركزية
    """
    x = np.linspace(x_range[0], x_range[1], 1000)
    
    # حساب قيم y
    try:
        y = func(x)
    except:
        y = np.zeros_like(x)

    # معالجة خطوط التقارب الرأسية (إن وجدت) لإخفاء الخط الواصل
    if has_asymptote_at is not None:
        threshold = 20
        y_diff = np.diff(y, prepend=y[0])
        y[np.abs(y_diff) > threshold] = np.nan
        
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # --- المحاور في المنتصف ---
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # التدريج والشبكة
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=7, direction='inout')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)

    # رسم الدالة
    ax.plot(x, y, color='#0056b3', linewidth=2)
    
    # رسم خط التقارب الرأسي إذا وجد
    if has_asymptote_at is not None:
         ax.axvline(x=has_asymptote_at, color='red', linestyle='--', linewidth=1, alpha=0.6)

    # حدود الرسم
    ax.set_ylim(y_range)
    ax.set_xlim(x_range)
    
    # العنوان (الحرف فقط)
    ax.set_title(title, fontsize=14, loc='right', color='black', fontweight='bold')
    
    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# 4. بيانات الأسئلة (27 - 32)
# ---------------------------------------------------------

# دوال مساعدة لإنشاء الأشكال المعقدة (Piecewise)
def q27_correct(x):
    # f(0)=1, f(2)=5. Max at 2, Min at 0.
    return -1 * (x**3) + 3*(x**2) + 1 

def q28_correct(x):
    # f(-1)=1 (min smooth), f(2)=5 (max sharp), f'(-1)=0, f'(2) DNE
    # Parabola for left part, Linear for right part meeting at 2
    return np.where(x < 2, 4/9 * (x+1)**2 + 1, -2*(x-2) + 5)

def q29_correct(x):
    # f(3)=0, VA at 0. Max at 3? No, f'(x)<0 for x>3.
    # f'(x)<0 x<0 (dec), f'(x)>0 0<x<3 (inc), f'(x)<0 x>3 (dec).
    # VA at 0. Max at 3.
    # Shape: Hyperbola-like left, upside down parabola right?
    return np.where(x < 0, -1/x - 2, np.where(x==0, np.nan, -(x-3)**2/2))

def q30_correct(x):
    # f(1)=0 (min), HA at y=2 as x->inf.
    return 2 * (x-1)**2 / (1 + (x-1)**2)

def q31_correct(x):
    # f(-1)=0, f(2)=0. Sharp min at -1 (f' DNE). Saddle at 2 (f'=0).
    # Dec (-inf, -1), Inc (-1, 0), Dec (0, 2), Dec (2, inf).
    # Means Max at 0 (approx). Sharp min -1. Saddle 2.
    val = np.zeros_like(x)
    mask1 = x < -1
    mask2 = (x >= -1) & (x < 0)
    mask3 = x >= 0
    val[mask1] = -2*(x[mask1]+1) # Linear down
    val[mask2] = 2*(x[mask2]+1) # Linear up (Sharp corner at -1)
    # Cubic saddle decreasing at 2: -(x-2)^3
    val[mask3] = -0.5 * (x[mask3]) * (x[mask3]-3.5) * np.exp(-0.2*x[mask3]) # Approximate bump then down
    # Let's try simpler logic for visual
    return np.where(x < -1, -2*(x+1), 
           np.where(x < 0, 2*(x+1),
           np.where(x < 4, 2 - 0.25*(x)**2 - 0.1*(x-2)**3, -10))) # Tricky to model perfectly, but visuals matter

def q32_correct(x):
    # f(0)=0 (saddle inc), f(3)=-1 (max). 
    # Inc (-inf, 0), Inc (0, 1), Inc (1, 3). Dec (3, inf).
    # f'(0)=0 (flat). f'(1) DNE (kink). f'(3)=0 (round).
    return np.where(x < 1, x**3, # Flat at 0
           np.where(x < 3, 1 + (x-1), # Linear up (kink at 1)
           -1 * (x-3)**2 + 3)) # Parabola down (smooth max at 3, value adjusted visual)


questions = [
    {
        "id": 27,
        "latex": r"f(0)=1, f(2)=5, \\ f'(x) < 0 \text{ for } x < 0 \text{ and } x > 2, \\ f'(x) > 0 \text{ for } 0 < x < 2",
        "correct": {"func": lambda x: -0.5*(x**3) + 1.5*(x**2) + 1, "va": None}, # Scaled cubic
        "distractors": [
            {"func": lambda x: 0.5*(x**3) - 1.5*(x**2) + 3, "va": None}, # Inverted signs (Min at 2)
            {"func": lambda x: (x-1)**2 + 1, "va": None}, # Parabola (Wrong extrema)
            {"func": lambda x: -x + 3, "va": None}, # Linear (No extrema)
        ]
    },
    {
        "id": 28,
        "latex": r"f(-1)=1, f(2)=5, \\ f'(x) < 0 \text{ for } x < -1 \text{ and } x > 2, \\ f'(x) > 0 \text{ for } -1 < x < 2, \\ f'(-1)=0, f'(2) \text{ does not exist}",
        "correct": {"func": lambda x: np.where(x<2, 4/9*(x+1)**2+1, 5-2*(x-2)), "va": None},
        "distractors": [
            {"func": lambda x: -1*(x-0.5)**2 + 6, "va": None}, # Smooth parabola (No sharp corner)
            {"func": lambda x: np.where(x<-1, -x, (x+1)**2 + 1), "va": None}, # Sharp min, Smooth max (Reversed logic)
            {"func": lambda x: np.sin(x)+3, "va": None}, # Periodic wave
        ]
    },
    {
        "id": 29,
        "latex": r"f(3)=0, \\ f'(x) < 0 \text{ for } x < 0 \text{ and } x > 3, \\ f'(x) > 0 \text{ for } 0 < x < 3, \\ f'(3)=0, f(0) \text{ and } f'(0) \text{ do not exist}",
        "correct": {"func": lambda x: np.where(x<0, -1/x - 2, np.where(x==0, np.nan, -(x-3)**2/3)), "va": 0},
        "distractors": [
            {"func": lambda x: -(x-1.5)**2 + 2.25, "va": None}, # Parabola (No Asymptote)
            {"func": lambda x: np.where(x<3, (x-3)**2, -(x-3)), "va": 3}, # Asymptote at 3 (Wrong location)
            {"func": lambda x: np.where(x<0, x, x-3), "va": 0}, # Wrong increasing/decreasing intervals
        ]
    },
    {
        "id": 30,
        "latex": r"f(1)=0, \lim_{x \to \infty} f(x) = 2, \\ f'(x) < 0 \text{ for } x < 1, f'(x) > 0 \text{ for } x > 1, \\ f'(1)=0",
        "correct": {"func": lambda x: 2 * (x-1)**2 / (0.5 + (x-1)**2), "va": None},
        "distractors": [
            {"func": lambda x: -2 * (x-1)**2 / (1 + (x-1)**2) + 2, "va": None}, # Inverted (Max at 1)
            {"func": lambda x: (x-1)**2, "va": None}, # Parabola (No Horizontal Asymptote)
            {"func": lambda x: 2 * (x+1)**2 / (1 + (x+1)**2), "va": None}, # Shifted (Min at -1)
        ]
    },
    {
        "id": 31,
        "latex": r"f(-1)=f(2)=0, \\ f'(x) < 0 \text{ for } x < -1, 0 < x < 2, x > 2, \\ f'(x) > 0 \text{ for } -1 < x < 0, \\ f'(-1) \text{ DNE}, f'(2)=0",
        "correct": {"func": lambda x: np.where(x<-1, -2*(x+1), np.where(x<0, 2*(x+1), np.where(x<2, 2-(x)**2/2, -(x-2)**3))), "va": None},
        "distractors": [
            {"func": lambda x: (x+1)*(x-2)**2, "va": None}, # Polynomial (Smooth everywhere, wrong min type)
            {"func": lambda x: np.where(x<2, (x-2)**2, -(x-2)), "va": None}, # Sharp at 2 (Wrong, should be smooth flat)
            {"func": lambda x: np.sin(x)*x, "va": None}, # Random wavy
        ]
    },
    {
        "id": 32,
        "latex": r"f(0)=0, f(3)=-1, \\ f'(x) < 0 \text{ for } x > 3, \\ f'(x) > 0 \text{ for } x < 0, 0 < x < 1, 1 < x < 3, \\ f'(0)=0, f'(1) \text{ DNE}, f'(3)=0",
        "correct": {"func": lambda x: np.where(x<1, (x)**3, np.where(x<3, 0.5*(x-1)+1, -(x-3)**2 + 2)), "va": None},
        "distractors": [
            {"func": lambda x: -x**2 + 3*x, "va": None}, # Simple parabola (No saddle or kink)
            {"func": lambda x: np.where(x<1, -x**2, x), "va": None}, # Wrong intervals
            {"func": lambda x: np.where(x<0, -x, np.where(x<3, x, -x+6)), "va": None}, # Sharp min at 0 (Should be saddle)
        ]
    }
]

# ---------------------------------------------------------
# 5. عرض التطبيق
# ---------------------------------------------------------

# الحصول على السؤال الحالي
q_idx = st.session_state.q_index
q = questions[q_idx]

# --- شريط التقدم ---
progress = (q_idx + 1) / len(questions)
st.progress(progress)

# --- نص السؤال الموحد ---
st.markdown("""
<div class="question-box">
    <div class="question-text-ar">أي من الرسومات البيانية التالية يحقق جميع الشروط المذكورة أدناه؟</div>
    <div class="question-text-en">Which of the following graphs satisfies all the given conditions?</div>
</div>
""", unsafe_allow_html=True)

# عرض الشروط الرياضية
st.latex(q['latex'])

# --- تجهيز الخيارات (الرسوم) ---
random.seed(q['id'] + 100) # Seed مختلف قليلاً

options_data = []
# إضافة الخيار الصحيح
options_data.append({
    "type": "correct",
    "fig": plot_function(q["correct"]["func"], has_asymptote_at=q["correct"]["va"])
})
# إضافة الخيارات الخاطئة
for dist in q["distractors"]:
    options_data.append({
        "type": "wrong",
        "fig": plot_function(dist["func"], has_asymptote_at=dist["va"])
    })

# خلط الخيارات
random.shuffle(options_data)

# --- عرض الرسوم في شبكة ---
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
cols_list = [col1, col2, col3, col4]
letters = ['A', 'B', 'C', 'D']
correct_letter = None

for idx, opt_data in enumerate(options_data):
    letter = letters[idx]
    with cols_list[idx]:
        # نضع الحرف داخل الرسم كعنوان
        opt_data["fig"].axes[0].set_title(f"({letter})", loc='left', fontsize=14)
        st.pyplot(opt_data["fig"])
        if opt_data["type"] == "correct":
            correct_letter = letter

# --- منطقة الإجابة ---
st.markdown("---")
col_input, col_action = st.columns([2, 1])

with col_input:
    user_answer = st.radio(
        "اختر الرمز المطابق للرسم الصحيح:",
        letters,
        key=f"radio_{q['id']}",
        horizontal=True
    )

with col_action:
    st.write("") 
    st.write("") 
    check_btn = st.button("تحقق من الإجابة", key=f"check_{q['id']}")

# --- التحقق ---
if check_btn:
    if user_answer == correct_letter:
        st.success(f"✅ أحسنت! الرسم ({correct_letter}) هو الرسم الوحيد الذي يحقق جميع الشروط.")
    else:
        st.error(f"❌ إجابة خاطئة. الإجابة الصحيحة هي ({correct_letter}).")
        with st.expander("توضيح الشروط"):
             st.info("راجع إشارات المشتقة الأولى (التزايد والتناقص)، والنقاط التي تكون عندها المشتقة صفرًا (مماس أفقي) أو غير موجودة (رأس حاد أو مماس رأسي).")

# --- أزرار التنقل ---
st.markdown("---")
c1, c2, c3 = st.columns([1, 2, 1])

with c1:
    if st.session_state.q_index > 0:
        if st.button("⬅️ السابق"):
            prev_question()
            st.rerun()

with c3:
    if st.session_state.q_index < len(questions) - 1:
        if st.button("التالي ➡️"):
            next_question()
            st.rerun()