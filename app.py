import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import io
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict · Previsão de Desempenho Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d0f1a;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #12152a !important;
    border-right: 1px solid #1e2340;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.5px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1d35 0%, #12152a 100%);
    border: 1px solid #2a2f55;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.metric-card h4 {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #6b7db3;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0 0 0.4rem 0;
}

.metric-card .value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
}

/* Prediction result */
.result-box {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}

.result-aprovado {
    background: linear-gradient(135deg, #0f3d2e 0%, #0a2e22 100%);
    border: 2px solid #22c55e;
}

.result-reprovado {
    background: linear-gradient(135deg, #3d1a0f 0%, #2e120a 100%);
    border: 2px solid #ef4444;
}

.result-box h2 {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem;
    margin: 0;
}

.result-box p {
    color: #a0aec0;
    margin: 0.5rem 0 0 0;
}

/* Tag badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.badge-green { background: #14532d; color: #86efac; border: 1px solid #22c55e; }
.badge-red   { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }
.badge-blue  { background: #1e3a5f; color: #93c5fd; border: 1px solid #3b82f6; }

/* Divider */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4a5580;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin: 2rem 0 1rem 0;
    border-bottom: 1px solid #1e2340;
    padding-bottom: 0.5rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Sliders & inputs */
[data-testid="stSlider"] > div > div { color: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA GENERATION (simula o dataset do Kaggle)
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=395):
    """Gera dataset sintético baseado na estrutura real do Student Alcohol Consumption."""
    np.random.seed(42)
    data = {
        "age":        np.random.randint(15, 22, n),
        "Medu":       np.random.randint(0, 5, n),
        "Fedu":       np.random.randint(0, 5, n),
        "traveltime": np.random.randint(1, 5, n),
        "studytime":  np.random.randint(1, 5, n),
        "failures":   np.random.choice([0,1,2,3], n, p=[0.67,0.20,0.08,0.05]),
        "famrel":     np.random.randint(1, 6, n),
        "freetime":   np.random.randint(1, 6, n),
        "goout":      np.random.randint(1, 6, n),
        "Dalc":       np.random.randint(1, 6, n),
        "Walc":       np.random.randint(1, 6, n),
        "health":     np.random.randint(1, 6, n),
        "absences":   np.random.randint(0, 40, n),
        "G1":         np.random.randint(0, 20, n),
        "G2":         np.random.randint(0, 20, n),
    }
    df = pd.DataFrame(data)
    # G3 correlacionado com G1, G2 e failures
    g3_raw = (0.4*df["G1"] + 0.4*df["G2"]
              - 3*df["failures"]
              + 0.3*df["studytime"]
              - 0.2*df["absences"]/10
              + np.random.normal(0, 1.5, n))
    df["G3"] = np.clip(g3_raw.round(), 0, 20).astype(int)
    df["aprovado"] = (df["G3"] >= 10).astype(int)
    return df


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = generate_dataset()
    features = ["age","Medu","Fedu","traveltime","studytime","failures",
                 "famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2"]
    X = df[features].values
    y = df["aprovado"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, features, acc, report, cm, X_test_s, y_test, y_pred, df


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 2rem 0;'>
        <div style='font-size:2.5rem;'>🎓</div>
        <div style='font-family: Space Mono, monospace; font-size:1.1rem; color:#e8eaf0; font-weight:700;'>EduPredict</div>
        <div style='font-size:0.72rem; color:#4a5580; letter-spacing:2px; text-transform:uppercase;'>Sistema de Previsão Escolar</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navegação",
        ["🔮 Previsão Individual", "📊 Dashboard do Modelo", "📂 Dataset"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#4a5580; line-height:1.8;'>
    <b style='color:#6b7db3;'>Modelo</b><br>
    MLP · 3 camadas ocultas<br>
    64 → 32 → 16 neurônios<br>
    Ativação: ReLU · Otimizador: Adam<br><br>
    <b style='color:#6b7db3;'>Dataset</b><br>
    Student Alcohol Consumption<br>
    UCI / Kaggle<br><br>
    <b style='color:#6b7db3;'>PBL 3 · UNDB 2026.1</b>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("Treinando modelo de rede neural..."):
    model, scaler, features, acc, report, cm, X_test_s, y_test, y_pred, df = train_model()


# ═══════════════════════════════════════════════
# PAGE 1 — PREVISÃO INDIVIDUAL
# ═══════════════════════════════════════════════
if page == "🔮 Previsão Individual":
    st.markdown("""
    <h1 style='font-size:1.8rem; color:#e8eaf0; margin-bottom:0;'>Previsão de Desempenho</h1>
    <p style='color:#4a5580; margin-top:0.3rem;'>Insira os dados do aluno para gerar uma previsão de risco escolar.</p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dados Acadêmicos</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        G1 = st.slider("Nota 1º Período (G1)", 0, 20, 10)
        G2 = st.slider("Nota 2º Período (G2)", 0, 20, 10)
        failures = st.selectbox("Reprovações anteriores", [0,1,2,3])
    with col2:
        absences = st.slider("Faltas", 0, 40, 5)
        studytime = st.select_slider("Tempo de estudo (h/sem)",
                                     options=["<2h","2–5h","5–10h",">10h"],
                                     value="2–5h")
        studytime_val = ["<2h","2–5h","5–10h",">10h"].index(studytime) + 1
    with col3:
        age = st.slider("Idade", 15, 22, 16)
        traveltime = st.select_slider("Tempo de deslocamento",
                                      options=["<15min","15–30min","30min–1h",">1h"],
                                      value="15–30min")
        traveltime_val = ["<15min","15–30min","30min–1h",">1h"].index(traveltime) + 1

    st.markdown('<div class="section-title">Contexto Familiar e Social</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        Medu = st.slider("Escolaridade da mãe (0–4)", 0, 4, 2)
        Fedu = st.slider("Escolaridade do pai (0–4)", 0, 4, 2)
        famrel = st.slider("Relação familiar (1–5)", 1, 5, 3)
    with col5:
        goout = st.slider("Saídas com amigos (1–5)", 1, 5, 3)
        freetime = st.slider("Tempo livre (1–5)", 1, 5, 3)
        health = st.slider("Saúde (1–5)", 1, 5, 3)
    with col6:
        Dalc = st.slider("Consumo álcool (dias úteis) (1–5)", 1, 5, 1)
        Walc = st.slider("Consumo álcool (fim de semana) (1–5)", 1, 5, 1)

    st.markdown("")
    predict_btn = st.button("🔮  GERAR PREVISÃO", use_container_width=True)

    if predict_btn:
        input_data = np.array([[age, Medu, Fedu, traveltime_val, studytime_val,
                                  failures, famrel, freetime, goout,
                                  Dalc, Walc, health, absences, G1, G2]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]

        if pred == 1:
            st.markdown(f"""
            <div class="result-box result-aprovado">
                <span class="badge badge-green">✓ APROVADO</span>
                <h2 style='color:#22c55e; margin-top:1rem;'>Aluno com bom desempenho</h2>
                <p>Probabilidade de aprovação: <b style='color:#22c55e'>{prob[1]*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-reprovado">
                <span class="badge badge-red">⚠ RISCO DE REPROVAÇÃO</span>
                <h2 style='color:#ef4444; margin-top:1rem;'>Aluno em situação de risco</h2>
                <p>Probabilidade de reprovação: <b style='color:#ef4444'>{prob[0]*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

        # Probability bar
        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(5, 1.2))
            fig.patch.set_facecolor("#0d0f1a")
            ax.set_facecolor("#0d0f1a")
            ax.barh(["Reprovação", "Aprovação"],
                    [prob[0], prob[1]],
                    color=["#ef4444","#22c55e"],
                    height=0.5)
            ax.set_xlim(0, 1)
            ax.tick_params(colors="#6b7db3", labelsize=9)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.xaxis.set_visible(False)
            for i, v in enumerate([prob[0], prob[1]]):
                ax.text(v + 0.02, i, f"{v*100:.1f}%", va="center",
                        color="#e8eaf0", fontsize=9, fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Risk factors
        with col_b:
            risks = []
            if failures > 0:  risks.append(f"⚠ {failures} reprovação(ões) anterior(es)")
            if absences > 15: risks.append(f"⚠ Muitas faltas ({absences})")
            if G1 < 8:        risks.append(f"⚠ Nota G1 baixa ({G1})")
            if G2 < 8:        risks.append(f"⚠ Nota G2 baixa ({G2})")
            if studytime_val == 1: risks.append("⚠ Pouco tempo de estudo")
            if Walc >= 4:     risks.append(f"⚠ Alto consumo de álcool no FDS")

            if risks:
                st.markdown("**Fatores de risco identificados:**")
                for r in risks:
                    st.markdown(f"<span style='color:#fca5a5; font-size:0.85rem;'>{r}</span>",
                                unsafe_allow_html=True)
            else:
                st.markdown("**✅ Nenhum fator de risco crítico identificado.**")


# ═══════════════════════════════════════════════
# PAGE 2 — DASHBOARD DO MODELO
# ═══════════════════════════════════════════════
elif page == "📊 Dashboard do Modelo":
    st.markdown("""
    <h1 style='font-size:1.8rem; color:#e8eaf0; margin-bottom:0;'>Dashboard do Modelo</h1>
    <p style='color:#4a5580; margin-top:0.3rem;'>Métricas de desempenho da rede neural treinada.</p>
    """, unsafe_allow_html=True)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    precision_1 = report["1"]["precision"]
    recall_1    = report["1"]["recall"]
    f1_1        = report["1"]["f1-score"]

    for col, label, value, color in [
        (col1, "Acurácia",  f"{acc*100:.1f}%",          "#3b82f6"),
        (col2, "Precisão",  f"{precision_1*100:.1f}%",  "#22c55e"),
        (col3, "Recall",    f"{recall_1*100:.1f}%",     "#a855f7"),
        (col4, "F1-Score",  f"{f1_1*100:.1f}%",         "#f59e0b"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{label}</h4>
                <div class="value" style="color:{color}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Matriz de Confusão · Curva de Loss</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0d0f1a")
        ax.set_facecolor("#12152a")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Reprovado","Aprovado"],
                    yticklabels=["Reprovado","Aprovado"],
                    ax=ax, linewidths=1, linecolor="#0d0f1a",
                    annot_kws={"size":14, "weight":"bold", "color":"white"})
        ax.tick_params(colors="#6b7db3", labelsize=10)
        ax.set_xlabel("Previsto", color="#6b7db3", fontsize=10)
        ax.set_ylabel("Real", color="#6b7db3", fontsize=10)
        ax.set_title("Matriz de Confusão", color="#e8eaf0", fontsize=11,
                     fontfamily="monospace", pad=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        loss_curve = model.loss_curve_
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0d0f1a")
        ax.set_facecolor("#12152a")
        ax.plot(loss_curve, color="#3b82f6", linewidth=2, label="Loss de treino")
        ax.set_xlabel("Épocas", color="#6b7db3", fontsize=10)
        ax.set_ylabel("Loss", color="#6b7db3", fontsize=10)
        ax.set_title("Curva de Aprendizado", color="#e8eaf0", fontsize=11,
                     fontfamily="monospace", pad=12)
        ax.tick_params(colors="#6b7db3")
        for spine in ax.spines.values(): spine.set_color("#1e2340")
        ax.legend(facecolor="#12152a", edgecolor="#1e2340", labelcolor="#a0aec0")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-title">Importância das Variáveis (approx.)</div>',
                unsafe_allow_html=True)
    # Approximate feature importance via weight magnitude
    weights = np.abs(model.coefs_[0]).mean(axis=1)
    feat_imp = pd.Series(weights, index=features).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0d0f1a")
    ax.set_facecolor("#12152a")
    colors = ["#3b82f6" if v > feat_imp.median() else "#1e3a5f" for v in feat_imp]
    feat_imp.plot(kind="barh", ax=ax, color=colors)
    ax.tick_params(colors="#6b7db3", labelsize=9)
    ax.set_xlabel("Magnitude média dos pesos", color="#6b7db3")
    ax.set_title("Contribuição das Variáveis (1ª camada)", color="#e8eaf0",
                 fontfamily="monospace")
    for spine in ax.spines.values(): spine.set_color("#1e2340")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-title">Arquitetura da Rede Neural</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#12152a; border:1px solid #1e2340; border-radius:12px; padding:1.5rem;
                font-family: Space Mono, monospace; font-size:0.82rem; color:#6b7db3; line-height:2;'>
    <span style='color:#3b82f6;'>INPUT</span>  → 15 neurônios (features do aluno)<br>
    <span style='color:#a855f7;'>HIDDEN 1</span> → 64 neurônios · Ativação: ReLU<br>
    <span style='color:#a855f7;'>HIDDEN 2</span> → 32 neurônios · Ativação: ReLU<br>
    <span style='color:#a855f7;'>HIDDEN 3</span> → 16 neurônios · Ativação: ReLU<br>
    <span style='color:#22c55e;'>OUTPUT</span>  → 2 classes (aprovado / reprovado) · Softmax<br><br>
    <span style='color:#f59e0b;'>Otimizador:</span> Adam &nbsp;|&nbsp;
    <span style='color:#f59e0b;'>Early Stopping:</span> Ativo &nbsp;|&nbsp;
    <span style='color:#f59e0b;'>Max iterações:</span> 500
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 3 — DATASET
# ═══════════════════════════════════════════════
elif page == "📂 Dataset":
    st.markdown("""
    <h1 style='font-size:1.8rem; color:#e8eaf0; margin-bottom:0;'>Dataset</h1>
    <p style='color:#4a5580; margin-top:0.3rem;'>Student Alcohol Consumption · UCI / Kaggle</p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h4>Total de alunos</h4>
            <div class="value" style="color:#3b82f6">{len(df)}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        aprov = df["aprovado"].sum()
        st.markdown(f"""<div class="metric-card">
            <h4>Aprovados</h4>
            <div class="value" style="color:#22c55e">{aprov} <span style='font-size:1rem'>({aprov/len(df)*100:.0f}%)</span></div>
        </div>""", unsafe_allow_html=True)
    with col3:
        reprov = len(df) - aprov
        st.markdown(f"""<div class="metric-card">
            <h4>Reprovados</h4>
            <div class="value" style="color:#ef4444">{reprov} <span style='font-size:1rem'>({reprov/len(df)*100:.0f}%)</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribuição de Notas Finais (G3)</div>',
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("#0d0f1a")
    for ax in axes: ax.set_facecolor("#12152a")

    axes[0].hist(df["G3"], bins=20, color="#3b82f6", edgecolor="#0d0f1a", linewidth=0.5)
    axes[0].axvline(10, color="#ef4444", linestyle="--", linewidth=1.5, label="Mínimo aprovação (10)")
    axes[0].set_title("Distribuição de G3", color="#e8eaf0", fontfamily="monospace")
    axes[0].tick_params(colors="#6b7db3")
    axes[0].legend(facecolor="#12152a", edgecolor="#1e2340", labelcolor="#a0aec0", fontsize=8)
    for spine in axes[0].spines.values(): spine.set_color("#1e2340")

    counts = df["aprovado"].value_counts()
    axes[1].pie(counts, labels=["Reprovado","Aprovado"],
                colors=["#ef4444","#22c55e"],
                autopct="%1.1f%%", startangle=90,
                textprops={"color":"#e8eaf0", "fontsize":10},
                wedgeprops={"edgecolor":"#0d0f1a", "linewidth":2})
    axes[1].set_title("Proporção Aprovação", color="#e8eaf0", fontfamily="monospace")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-title">Amostra dos Dados</div>', unsafe_allow_html=True)
    st.dataframe(
        df.head(20).style.applymap(
            lambda v: "color: #22c55e" if v == 1 else ("color: #ef4444" if v == 0 else ""),
            subset=["aprovado"]
        ),
        use_container_width=True,
        height=350
    )

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Baixar Dataset CSV", csv, "student_data.csv", "text/csv")
