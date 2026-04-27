import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="EduPredict · Previsão de Desempenho Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0f1a; color: #e8eaf0; }
[data-testid="stSidebar"] { background: #12152a !important; border-right: 1px solid #1e2340; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.metric-card { background: linear-gradient(135deg,#1a1d35,#12152a); border:1px solid #2a2f55; border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem; }
.metric-card h4 { font-family:'Space Mono',monospace; font-size:0.7rem; color:#6b7db3; text-transform:uppercase; letter-spacing:2px; margin:0 0 0.4rem 0; }
.metric-card .value { font-size:2.2rem; font-weight:700; font-family:'Space Mono',monospace; }
.result-box { border-radius:16px; padding:2rem; text-align:center; margin:1.5rem 0; }
.result-aprovado { background:linear-gradient(135deg,#0f3d2e,#0a2e22); border:2px solid #22c55e; }
.result-reprovado { background:linear-gradient(135deg,#3d1a0f,#2e120a); border:2px solid #ef4444; }
.result-box h2 { font-family:'Space Mono',monospace !important; font-size:1.8rem; margin:0; }
.badge { display:inline-block; padding:0.25rem 0.8rem; border-radius:999px; font-size:0.75rem; font-weight:600; font-family:'Space Mono',monospace; text-transform:uppercase; letter-spacing:1px; }
.badge-green { background:#14532d; color:#86efac; border:1px solid #22c55e; }
.badge-red { background:#7f1d1d; color:#fca5a5; border:1px solid #ef4444; }
.badge-yellow { background:#78350f; color:#fde68a; border:1px solid #f59e0b; }
.badge-purple { background:#4c1d95; color:#c4b5fd; border:1px solid #a855f7; }
.section-title { font-family:'Space Mono',monospace; font-size:0.65rem; color:#4a5580; text-transform:uppercase; letter-spacing:3px; margin:2rem 0 1rem 0; border-bottom:1px solid #1e2340; padding-bottom:0.5rem; }
.stButton > button { background:linear-gradient(135deg,#3b82f6,#6366f1) !important; color:white !important; border:none !important; border-radius:8px !important; font-family:'Space Mono',monospace !important; font-weight:700 !important; letter-spacing:1px !important; padding:0.6rem 1.5rem !important; }
.perfil-card { background:#12152a; border:1px solid #2a2f55; border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.8rem; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = "models"

PERFIS = {
    "Ausente Crônico": {"cor":"#ef4444","descricao":"Alto número de faltas, baixa frequência","intervencao":"Contato com responsáveis · Encaminhamento a assistente social","emoji":"⚠️"},
    "Dificuldade Acadêmica": {"cor":"#f59e0b","descricao":"Reprovações anteriores, pouco estudo","intervencao":"Reforço pedagógico · Acompanhamento individual","emoji":"📚"},
    "Vulnerabilidade Social": {"cor":"#a855f7","descricao":"Baixa escolaridade dos pais, sem suporte familiar","intervencao":"Apoio psicossocial · Inclusão em programas de assistência","emoji":"🏠"},
    "Risco Comportamental": {"cor":"#3b82f6","descricao":"Alto consumo de álcool, saídas frequentes","intervencao":"Orientação · Programas de conscientização","emoji":"🔵"},
}

@st.cache_resource
def load_models():
    m, errs = {}, []
    paths = {
        "mlp":            ("mlp_risco.keras",       "keras"),
        "kmeans":         ("kmeans_perfis.joblib",   "joblib"),
        "scaler":         ("scaler.joblib",          "joblib"),
        "scaler_cluster": ("scaler_cluster.joblib",  "joblib"),
        "label_encoders": ("label_encoders.joblib",  "joblib"),
        "cluster_labels": ("cluster_labels.json",    "json"),
        "feature_columns":("feature_columns.json",   "json"),
    }
    for key, (fname, ftype) in paths.items():
        p = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(p):
            errs.append(f"{fname} não encontrado")
            continue
        try:
            if ftype == "keras" and KERAS_AVAILABLE:
                m[key] = keras.models.load_model(p)
            elif ftype == "joblib":
                m[key] = joblib.load(p)
            elif ftype == "json":
                with open(p) as f:
                    m[key] = json.load(f)
        except Exception as e:
            errs.append(f"{fname}: {e}")
    return m, errs

models, load_errors = load_models()
mlp_loaded = "mlp" in models

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 2rem 0;'>
        <div style='font-size:2.5rem;'>🎓</div>
        <div style='font-family:Space Mono,monospace;font-size:1.1rem;color:#e8eaf0;font-weight:700;'>EduPredict</div>
        <div style='font-size:0.72rem;color:#4a5580;letter-spacing:2px;text-transform:uppercase;'>Sistema de Previsão Escolar</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Nav", ["🔮 Previsão Individual","📊 Dashboard do Modelo","ℹ️ Sobre o Projeto"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;color:#4a5580;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;'>Modelos carregados</div>", unsafe_allow_html=True)
    for k, label in [("mlp","MLP keras"),("kmeans","K-Means"),("scaler","Scaler MLP"),("scaler_cluster","Scaler Cluster")]:
        icon = "🟢" if k in models else "🔴"
        st.markdown(f"<div style='font-size:0.75rem;color:#6b7db3;'>{icon} {label}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;color:#4a5580;line-height:1.8;'><b style='color:#6b7db3;'>Grupo</b> EdgeTech<br><b style='color:#6b7db3;'>UNDB</b> 1º/2026.1<br><b style='color:#6b7db3;'>Prof.</b> Wesleson Souza Silva</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 1 - PREVISÃO
# ══════════════════════════════════════════════
if page == "🔮 Previsão Individual":
    st.markdown("<h1 style='font-size:1.8rem;color:#e8eaf0;margin-bottom:0;'>Previsão de Desempenho</h1><p style='color:#4a5580;margin-top:0.3rem;'>Insira os dados do aluno para gerar uma análise preditiva de risco escolar.</p>", unsafe_allow_html=True)

    if not mlp_loaded:
        st.warning("⚠️ Modelo MLP não carregado. Verifique se os arquivos da pasta `models/` estão no repositório.")

    st.markdown('<div class="section-title">Dados Acadêmicos</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        G1 = st.slider("Nota 1º Período (G1)", 0, 20, 10)
        failures = st.selectbox("Reprovações anteriores", [0,1,2,3])
        absences = st.slider("Número de faltas", 0, 93, 5)
    with c2:
        studytime_lbl = st.select_slider("Tempo de estudo (h/sem)", ["<2h","2-5h","5-10h",">10h"], value="2-5h")
        studytime = ["<2h","2-5h","5-10h",">10h"].index(studytime_lbl)+1
        schoolsup = st.selectbox("Apoio educacional extra", ["no","yes"])
        age = st.slider("Idade", 15, 22, 16)
    with c3:
        tt_lbl = st.select_slider("Tempo de deslocamento", ["<15min","15-30min","30min-1h",">1h"], value="15-30min")
        traveltime = ["<15min","15-30min","30min-1h",">1h"].index(tt_lbl)+1
        sex = st.selectbox("Sexo", ["F","M"])
        school = st.selectbox("Escola", ["GP","MS"])

    st.markdown('<div class="section-title">Contexto Familiar e Social</div>', unsafe_allow_html=True)
    c4,c5,c6 = st.columns(3)
    with c4:
        Medu = st.slider("Escolaridade da mãe (0-4)",0,4,2)
        Fedu = st.slider("Escolaridade do pai (0-4)",0,4,2)
        famrel = st.slider("Relação familiar (1-5)",1,5,3)
    with c5:
        goout = st.slider("Saídas com amigos (1-5)",1,5,3)
        freetime = st.slider("Tempo livre (1-5)",1,5,3)
        health = st.slider("Saúde (1-5)",1,5,3)
    with c6:
        Dalc = st.slider("Álcool dias úteis (1-5)",1,5,1)
        Walc = st.slider("Álcool fim de semana (1-5)",1,5,1)
        Mjob = st.selectbox("Profissão da mãe",["teacher","health","services","at_home","other"])
        Fjob = st.selectbox("Profissão do pai",["teacher","health","services","at_home","other"])

    if st.button("🔮  GERAR PREVISÃO", use_container_width=True):
        alcool_total = Dalc + Walc
        edu_pais_media = (Medu+Fedu)/2
        evolucao_nota = G1 - 10

        use_real = False
        if mlp_loaded:
            try:
                raw = {"age":age,"Medu":Medu,"Fedu":Fedu,"traveltime":traveltime,"studytime":studytime,
                       "failures":failures,"famrel":famrel,"freetime":freetime,"goout":goout,
                       "Dalc":Dalc,"Walc":Walc,"health":health,"absences":absences,"G1":G1,
                       "alcool_total":alcool_total,"edu_pais_media":edu_pais_media,"evolucao_nota":evolucao_nota,
                       "school_MS":int(school=="MS"),"sex_M":int(sex=="M"),"schoolsup_yes":int(schoolsup=="yes"),
                       "Mjob_health":int(Mjob=="health"),"Mjob_other":int(Mjob=="other"),
                       "Mjob_services":int(Mjob=="services"),"Mjob_teacher":int(Mjob=="teacher"),
                       "Fjob_health":int(Fjob=="health"),"Fjob_other":int(Fjob=="other"),
                       "Fjob_services":int(Fjob=="services"),"Fjob_teacher":int(Fjob=="teacher")}
                fc = models.get("feature_columns", list(raw.keys()))
                cols = fc if isinstance(fc, list) else list(raw.keys())
                inp = np.array([[raw.get(c,0) for c in cols]])
                if "scaler" in models:
                    inp = models["scaler"].transform(inp)
                prob_raw = models["mlp"].predict(inp, verbose=0)[0]
                prob_risco = float(prob_raw[1]) if len(prob_raw)>1 else float(prob_raw[0])
                prob_ok = 1-prob_risco
                pred = int(prob_risco >= 0.5)
                use_real = True
            except Exception as e:
                st.error(f"Erro no modelo MLP: {e}")
                pred = int(G1<10 or failures>0)
                prob_risco = 0.7 if pred else 0.3
                prob_ok = 1-prob_risco
        else:
            pred = int(G1<10 or failures>0)
            prob_risco = 0.7 if pred else 0.3
            prob_ok = 1-prob_risco

        tag = "&nbsp;·&nbsp;<span style='font-size:0.8rem;'>modelo real</span>" if use_real else ""
        if pred==0:
            st.markdown(f"<div class='result-box result-aprovado'><span class='badge badge-green'>✓ BAIXO RISCO</span><h2 style='color:#22c55e;margin-top:1rem;'>Aluno com bom desempenho</h2><p>Probabilidade de aprovação: <b style='color:#22c55e'>{prob_ok*100:.1f}%</b>{tag}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box result-reprovado'><span class='badge badge-red'>⚠ ALTO RISCO</span><h2 style='color:#ef4444;margin-top:1rem;'>Aluno em situação de risco</h2><p>Probabilidade de risco: <b style='color:#ef4444'>{prob_risco*100:.1f}%</b>{tag}</p></div>", unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            fig,ax = plt.subplots(figsize=(5,1.4)); fig.patch.set_facecolor("#0d0f1a"); ax.set_facecolor("#0d0f1a")
            ax.barh(["Risco","Aprovação"],[prob_risco,prob_ok],color=["#ef4444","#22c55e"],height=0.5)
            ax.set_xlim(0,1); ax.tick_params(colors="#6b7db3",labelsize=9)
            [s.set_visible(False) for s in ax.spines.values()]; ax.xaxis.set_visible(False)
            for i,v in enumerate([prob_risco,prob_ok]): ax.text(v+0.02,i,f"{v*100:.1f}%",va="center",color="#e8eaf0",fontsize=9,fontweight="bold")
            st.pyplot(fig,use_container_width=True); plt.close()
        with cb:
            risks=[]
            if failures>0: risks.append(f"⚠ {failures} reprovação(ões) anterior(es)")
            if absences>15: risks.append(f"⚠ Muitas faltas ({absences})")
            if G1<8: risks.append(f"⚠ Nota G1 baixa ({G1})")
            if studytime==1: risks.append("⚠ Pouco tempo de estudo")
            if Walc>=4: risks.append("⚠ Alto consumo de álcool no FDS")
            if edu_pais_media<1.5: risks.append("⚠ Baixa escolaridade dos pais")
            if risks:
                st.markdown("**Fatores de risco:**")
                for r in risks: st.markdown(f"<span style='color:#fca5a5;font-size:0.85rem;'>{r}</span>",unsafe_allow_html=True)
            else: st.success("Nenhum fator de risco crítico identificado.")

        if "kmeans" in models and "scaler_cluster" in models:
            try:
                ci = np.array([[absences,failures,studytime,famrel,goout,Dalc,Walc,edu_pais_media,alcool_total]])
                cs = models["scaler_cluster"].transform(ci)
                cid = int(models["kmeans"].predict(cs)[0])
                cl = models.get("cluster_labels",{})
                pnome = cl.get(str(cid), list(PERFIS.keys())[cid % len(PERFIS)])
                perf = PERFIS.get(pnome, list(PERFIS.values())[0])
                st.markdown('<div class="section-title">Perfil de Vulnerabilidade (K-Means)</div>', unsafe_allow_html=True)
                st.markdown(f"<div class='perfil-card'><div style='font-size:1.5rem;margin-bottom:0.5rem;'>{perf['emoji']}</div><div style='font-family:Space Mono,monospace;font-size:1rem;color:{perf['cor']};font-weight:700;'>{pnome}</div><div style='font-size:0.82rem;color:#6b7db3;margin:0.3rem 0;'>{perf['descricao']}</div><div style='font-size:0.82rem;color:#a0aec0;'><b style='color:#4a5580;'>Intervenção sugerida:</b><br>{perf['intervencao']}</div></div>", unsafe_allow_html=True)
            except Exception as e:
                st.info(f"Perfil K-Means indisponível: {e}")

# ══════════════════════════════════════════════
# PAGE 2 - DASHBOARD
# ══════════════════════════════════════════════
elif page == "📊 Dashboard do Modelo":
    st.markdown("<h1 style='font-size:1.8rem;color:#e8eaf0;margin-bottom:0;'>Dashboard do Modelo</h1><p style='color:#4a5580;margin-top:0.3rem;'>Métricas e arquitetura da rede neural treinada pela equipe.</p>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,label,val,color in [
        (c1,"F1-Score","0.81","#f59e0b"),
        (c2,"Recall","Alto ✓","#22c55e"),
        (c3,"Balanceamento","78% / 22%","#3b82f6"),
        (c4,"Acurácia","N/A *","#6b7db3"),
    ]:
        with col:
            st.markdown(f"<div class='metric-card'><h4>{label}</h4><div class='value' style='color:{color};font-size:1.6rem;'>{val}</div></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#12152a;border:1px solid #2a2f55;border-radius:10px;padding:1rem 1.2rem;
                font-size:0.82rem;color:#6b7db3;line-height:1.8;margin-bottom:1.5rem;'>
    <b style='color:#f59e0b;'>* Por que a acurácia não é a métrica principal?</b><br>
    O dataset é desbalanceado: ~78% dos alunos sem risco e apenas ~22% em risco.
    Um modelo que sempre prevê "sem risco" teria 78% de acurácia mas seria inútil.
    Por isso o foco é no <b style='color:#f59e0b;'>F1-Score (0.81)</b> e no
    <b style='color:#22c55e;'>Recall</b>, que penalizam erros na identificação de alunos em risco.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Arquitetura · MLP (Keras/TensorFlow)</div>', unsafe_allow_html=True)
    st.markdown("<div style='background:#12152a;border:1px solid #1e2340;border-radius:12px;padding:1.5rem;font-family:Space Mono,monospace;font-size:0.82rem;color:#6b7db3;line-height:2;'><span style='color:#3b82f6;'>INPUT</span> → features do aluno (dados até G1)<br><span style='color:#a855f7;'>HIDDEN</span> → camadas densas · Ativação: ReLU<br><span style='color:#22c55e;'>OUTPUT</span> → risco de reprovação (sigmoid)<br><br><span style='color:#f59e0b;'>Otimizador:</span> Adam &nbsp;|&nbsp;<span style='color:#f59e0b;'>Early Stopping:</span> Ativo &nbsp;|&nbsp;<span style='color:#f59e0b;'>Arquivo:</span> mlp_risco.keras</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Perfis K-Means</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Perfil":list(PERFIS.keys()),"Características":[p["descricao"] for p in PERFIS.values()],"Intervenção":[p["intervencao"] for p in PERFIS.values()]}), use_container_width=True, hide_index=True)

# PAGE 3 - SOBRE
# ══════════════════════════════════════════════
elif page == "ℹ️ Sobre o Projeto":
    st.markdown("<h1 style='font-size:1.8rem;color:#e8eaf0;margin-bottom:0;'>Sobre o Projeto</h1><p style='color:#4a5580;margin-top:0.3rem;'>PBL 3 · Sistema Inteligente de Previsão de Desempenho Escolar</p>", unsafe_allow_html=True)
    st.markdown("<div style='background:#12152a;border:1px solid #1e2340;border-radius:12px;padding:1.5rem;line-height:1.9;font-size:0.9rem;color:#a0aec0;'>Este sistema combina dois modelos complementares:<br><br><b style='color:#3b82f6;'>Rede Neural MLP</b> - Prevê probabilidade de risco individual sem usar G2 (evitando data leakage).<br><br><b style='color:#a855f7;'>K-Means</b> - Segmenta alunos em perfis de vulnerabilidade, entregando causa provável e intervenção específica ao gestor.</div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Equipe EdgeTech</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Membro":["Fernando","[Nome]","[Nome]","[Nome]","[Nome]"],"Responsabilidade":["Interface web + Deploy","Modelo MLP (Keras)","ETL + EDA","K-Means + Perfis","Documentação"]}), use_container_width=True, hide_index=True)
