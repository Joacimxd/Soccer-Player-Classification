import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def hex_rgba(hex_color: str, alpha: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'rgba({r},{g},{b},{alpha})'

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FIFA Scout AI - Copa del Mundo 2026",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  (dark stadium night theme)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .stApp {
        background: #0b1a3e;
        color: #e2e8f0;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="collapsedControl"] { 
        display: flex !important; 
        width: 180px !important;
        height: 45px !important;
        background-color: #1a6dff !important; 
        border-radius: 8px !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        position: fixed !important;
        top: 15px !important;
        left: 15px !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
    }
    [data-testid="collapsedControl"] svg { display: none !important; }
    [data-testid="collapsedControl"]::after {
        content: "ABRIR NAVBAR >" !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        letter-spacing: 1px !important;
    }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stSidebar"] {
        background: #081230;
        border-right: 1px solid rgba(26,109,255,.12);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 2px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: flex !important;
        width: 100% !important;
        box-sizing: border-box !important;
        padding: .5rem .75rem !important;
        margin: 0 !important;
        background: transparent !important;
        border-left: 3px solid transparent;
        border-radius: 0 8px 8px 0 !important;
        transition: all .2s ease !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(0,212,255,.05) !important;
        border-left-color: #1a6dff55 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: rgba(0,212,255,.1) !important;
        border-left: 3px solid #00d4ff !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: .78rem !important;
        font-weight: 500 !important;
        letter-spacing: .8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color: #00d4ff !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div[data-testid="stMarkdownContainer"] {
        width: 100% !important;
    }

    .card {
        background: #0e1f4a;
        border: 1px solid rgba(26,109,255,.15);
        border-radius: 14px;
        padding: 1.5rem;
        margin: .75rem 0;
    }
    .hero {
        background: #0a1538;
        border: 1px solid rgba(0,212,255,.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 40px rgba(0,212,255,.06);
    }
    .box-gold {
        background: rgba(0,212,255,.06);
        border-left: 4px solid #00d4ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .box-green {
        background: rgba(26,109,255,.08);
        border-left: 4px solid #1a6dff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    [data-testid="stMetric"] {
        background: rgba(26,109,255,.08) !important;
        border: 1px solid rgba(26,109,255,.15) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
    [data-testid="stMetricValue"] { color: #00d4ff !important; }
    [data-testid="stMetricLabel"] { color: #8ba3cf !important; }
    h1, h2, h3 { color: #ffffff !important; }
    h4, h5    { color: #00d4ff !important; }
    hr { border-color: rgba(26,109,255,.12) !important; }
    .stButton > button {
        background: #1a4fc9 !important;
        color: white !important;
        border: 1px solid rgba(26,109,255,.4) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
    }
    .stButton > button:hover {
        background: #2563eb !important;
        box-shadow: 0 0 15px rgba(26,109,255,.35) !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

POSITION_MAP = {
    'GK':  'GK',
    'CB':  'DEF', 'LB':  'DEF', 'RB':  'DEF', 'LWB': 'DEF', 'RWB': 'DEF',
    'CDM': 'MID', 'CM':  'MID', 'CAM': 'MID', 'LM':  'MID', 'RM':  'MID',
    'ST':  'FWD', 'CF':  'FWD', 'LW':  'FWD', 'RW':  'FWD',
}

FEATURES = [
    'age', 'height_cm', 'weight_kgs',
    'crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys',
    'dribbling', 'curve', 'freekick_accuracy', 'long_passing', 'ball_control',
    'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance',
    'shot_power', 'jumping', 'stamina', 'strength', 'long_shots',
    'aggression', 'interceptions', 'positioning', 'vision', 'penalties',
    'composure', 'marking', 'standing_tackle', 'sliding_tackle',
    'preferred_foot', 'skill_moves(1-5)',
]

EXPO_FEATURES = [
    'age', 'height_cm', 'weight_kgs',
    'sprint_speed', 'jumping', 'shot_power',
    'stamina', 'strength'
]

POS_COLOR = {'GK': '#f59e0b', 'DEF': '#1a6dff', 'MID': '#00d4ff', 'FWD': '#ef4444'}
POS_EMOJI = {'GK': '', 'DEF': '', 'MID': '', 'FWD': ''}
CLASSES   = ['GK', 'DEF', 'MID', 'FWD']

BASE = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='#e2e8f0',
    margin=dict(l=10, r=10, t=30, b=10),
)
GRID = dict(gridcolor='rgba(26,109,255,0.1)')

# ═══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE  (cached as singleton — runs once per server process)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Cargando plantilla y entrenando modelos...")
def load_pipeline():
    # 1 — raw data
    df_raw = pd.read_csv("data/fifa_players.csv")

    # 2 — map positions
    def map_pos(s):
        if pd.isna(s):
            return None
        return POSITION_MAP.get(str(s).split(',')[0].strip(), None)

    df = df_raw.copy()
    df['position_class'] = df['positions'].apply(map_pos)
    df = df.dropna(subset=['position_class'])

    # 3 — encode categorical
    df['preferred_foot'] = (df['preferred_foot'] == 'Right').astype(int)

    # 4 — select features
    avail = [f for f in FEATURES if f in df.columns]
    df_clean = df[avail + ['position_class', 'name']].dropna().reset_index(drop=True)

    # 5 — outlier detection
    skip = {'preferred_foot', 'skill_moves(1-5)'}
    num_feats = [f for f in avail if f not in skip]
    outlier_counts = {}
    for col in num_feats:
        Q1, Q3 = df_clean[col].quantile([.25, .75])
        IQR = Q3 - Q1
        n = int(((df_clean[col] < Q1 - 1.5*IQR) | (df_clean[col] > Q3 + 1.5*IQR)).sum())
        if n:
            outlier_counts[col] = n

    # 6 — IQR capping
    df_capped = df_clean.copy()
    for col in num_feats:
        Q1, Q3 = df_capped[col].quantile([.25, .75])
        IQR = Q3 - Q1
        df_capped[col] = df_capped[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    # 7 — scale + PCA
    X = df_capped[avail].values
    y = df_capped['position_class'].values
    names = df_capped['name'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_ev = PCA(n_components=min(20, len(avail)))
    pca_ev.fit(X_scaled)
    cum_var = np.cumsum(pca_ev.explained_variance_ratio_)

    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_scaled)

    pca_95 = PCA(n_components=0.95, random_state=42)
    X_pca = pca_95.fit_transform(X_scaled)

    # 8 — split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    # 9 — train
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)

    svm = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced',
              random_state=42, probability=True)
    svm.fit(X_train, y_train)

    dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
    dt.fit(X_train, y_train)

    # 10 — evaluate
    results = {}
    for mname, model in [('KNN', knn), ('SVM', svm), (u'\u00c1rbol de Decisi\u00f3n', dt)]:
        y_pred = model.predict(X_test)
        results[mname] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'cm':       confusion_matrix(y_test, y_pred, labels=CLASSES),
            'report':   classification_report(y_test, y_pred, labels=CLASSES, output_dict=True),
        }

    P = {
        'df_raw': df_raw, 'df_clean': df_clean, 'df_capped': df_capped,
        'features': avail, 'num_feats': num_feats, 'outlier_counts': outlier_counts,
        'X_scaled': X_scaled, 'X_2d': X_2d, 'X_pca': X_pca,
        'y': y, 'names': names, 'cum_var': cum_var,
        'scaler': scaler, 'pca_95': pca_95,
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'models': {'KNN': knn, 'SVM': svm, '\u00c1rbol de Decisi\u00f3n': dt},
        'results': results,
    }

    # Expo-specific pipeline (no PCA needed for 8 features)
    if all(f in df_capped.columns for f in EXPO_FEATURES):
        X_expo = df_capped[EXPO_FEATURES].values
        expo_scaler = StandardScaler()
        X_expo_scaled = expo_scaler.fit_transform(X_expo)
        expo_svm = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced',
                       random_state=42, probability=True)
        expo_svm.fit(X_expo_scaled, y)
        P['expo_scaler'] = expo_scaler
        P['expo_svm'] = expo_svm
        P['expo_features'] = EXPO_FEATURES

    return P


# ═══════════════════════════════════════════════════════════════════════════════
#  NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

STEPS = [
    ("", "INICIO",          "Vista General del Dataset"),
    ("", "EXPLORACION",     "Analisis Exploratorio"),
    ("", "LIMPIEZA",        "Limpieza de Datos"),
    ("", "ENTRENAMIENTO",   "Ingenieria de Caracteristicas"),
    ("", "TACTICAS",        "Escalado y PCA"),
    ("", "DIA DE PARTIDO",  "Entrenamiento de Modelos"),
    ("", "PITAZO FINAL",    "Resultados y Comparacion"),
    ("", "SCOUT ESTELAR",   "Predictor en Vivo"),
]

STEP_COLORS = [
    '#00d4ff',  # cyan
    '#1a6dff',  # royal blue
    '#00bfa5',  # teal
    '#5b8def',  # sky blue
    '#00e5c3',  # mint
    '#3d7aff',  # blue
    '#00c8e8',  # bright cyan
    '#6dacff',  # light blue
]

with st.sidebar:
    st.image('fifa.jpg', use_container_width=True)
    st.markdown("""
    <div style='text-align:center;padding:.5rem 0 .75rem'>
        <div style='font-size:1.4rem;font-weight:900;letter-spacing:3px;color:#ffffff;
                    text-shadow:0 0 20px rgba(0,212,255,.3)'>FIFA SCOUT AI</div>
        <div style='font-size:.65rem;letter-spacing:2px;color:#5b8def;margin-top:.2rem'>CLASIFICACION DE JUGADORES</div>
    </div>
    """, unsafe_allow_html=True)

    step = st.radio(
        "Pipeline",
        options=range(len(STEPS)),
        format_func=lambda i: f"{STEPS[i][1]}",
        label_visibility="collapsed",
    )



# ─── hero banner ─────────────────────────────────────────────────────────────
icon, title, subtitle = STEPS[step]
active_color = STEP_COLORS[step]
dots = "".join(
    f'<span style="width:9px;height:9px;border-radius:50%;'
    f'background:{STEP_COLORS[i] if i == step else "#ffffff12"};'
    f'display:inline-block;margin:0 3px"></span>'
    for i in range(len(STEPS))
)
st.markdown(f"""
<div class='hero'>
    <div style='font-size:2.8rem;line-height:1;margin-bottom:.5rem'>{icon}</div>
    <h1 style='color:{active_color} !important;margin:0;font-size:2.2rem;letter-spacing:4px;font-weight:900'>
        {title}
    </h1>
    <p style='color:{active_color};margin:.25rem 0 .75rem;font-size:1.05rem;letter-spacing:1px;opacity:.7'>
        {subtitle}
    </p>
    <div>{dots}</div>
</div>
""", unsafe_allow_html=True)

P = load_pipeline()   # singleton — instant after first load

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 0 · KICKOFF  — Dataset Overview
# ═══════════════════════════════════════════════════════════════════════════════
if step == 0:
    df = P['df_raw']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jugadores",        f"{len(df):,}")
    c2.metric("Atributos",        str(df.shape[1]))
    c3.metric("Nacionalidades",   str(df['nationality'].nunique()))
    c4.metric("Valoracion Prom.", f"{df['overall_rating'].mean():.1f}")

    st.markdown("---")
    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("#### Top 20 Jugadores por Valoracion General")
        top = (df.nlargest(20, 'overall_rating')
                 [['name', 'nationality', 'overall_rating', 'positions', 'age']]
                 .sort_values('overall_rating'))
        fig = px.bar(
            top, x='overall_rating', y='name', orientation='h',
            color='overall_rating',
            color_continuous_scale=['#0a1538', '#00d4ff', '#fbbf24'],
            text='overall_rating',
            hover_data=['nationality', 'positions', 'age'],
        )
        fig.update_layout(**BASE, height=520, coloraxis_showscale=False,
                          xaxis=dict(**GRID), yaxis_title='')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Distribucion de Valoraciones")
        fig = px.histogram(df, x='overall_rating', nbins=30,
                           color_discrete_sequence=['#00d4ff'])
        fig.update_layout(**BASE, height=200, bargap=.05,
                          xaxis=dict(**GRID), yaxis=dict(**GRID))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top 10 Nacionalidades")
        nat = df['nationality'].value_counts().head(10).reset_index()
        nat.columns = ['country', 'count']
        fig = px.bar(
            nat.sort_values('count'), x='count', y='country', orientation='h',
            color='count', color_continuous_scale=['#081230', '#00d4ff'],
        )
        fig.update_layout(**BASE, height=310, coloraxis_showscale=False,
                          xaxis=dict(**GRID), yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='box-gold'>
        <strong>Resumen de la Mision:</strong> 17,954 jugadores de la base de datos oficial de FIFA, cada uno con
        50 atributos que van desde centros hasta compostura. Nuestro objetivo:
        <strong>entrenar una IA para predecir la posicion tactica de cualquier jugador solo con sus estadisticas</strong>,
        tal como un scout de la Copa del Mundo perfilando a un jugador desconocido.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 · SCOUTING  — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif step == 1:
    df = P['df_clean']

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown("#### Composicion del Plantel")
        pos_dist = df['position_class'].value_counts().reset_index()
        pos_dist.columns = ['pos', 'n']
        fig = go.Figure(go.Pie(
            labels=[p for p in pos_dist['pos']],
            values=pos_dist['n'],
            hole=.52,
            marker_colors=[POS_COLOR[p] for p in pos_dist['pos']],
            textinfo='label+percent',
            textfont_size=13,
        ))
        fig.update_layout(
            **BASE, height=360, showlegend=False,
            annotations=[dict(text='17,954<br>jugadores', x=.5, y=.5,
                              font_size=13, showarrow=False, font_color='#fbbf24')],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='box-green' style='font-size:.85rem'>
            MID - 6,668  (37 %)<br>
            DEF - 5,883  (33 %)<br>
            FWD - 3,338  (19 %)<br>
            GK  - 2,065  (11 %)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Huellas Tacticas por Posicion")
        radar_cols = ['crossing', 'finishing', 'dribbling', 'stamina',
                      'aggression', 'composure', 'marking', 'jumping']
        avg = df.groupby('position_class')[radar_cols].mean()
        fig = go.Figure()
        for pos in CLASSES:
            if pos in avg.index:
                r_vals = avg.loc[pos].tolist()
                fig.add_trace(go.Scatterpolar(
                    r=r_vals + [r_vals[0]],
                    theta=radar_cols + [radar_cols[0]],
                    fill='toself', name=pos,
                    line_color=POS_COLOR[pos], opacity=.75,
                ))
        fig.update_layout(
            **BASE, height=380,
            polar=dict(bgcolor='rgba(0,0,0,0)',
                       radialaxis=dict(visible=True, range=[0, 80], color='#475569')),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    sample = df.sample(min(3000, len(df)), random_state=1)

    with col1:
        st.markdown("#### Altura vs Salto -- Podemos Separar Posiciones?")
        fig = px.scatter(sample, x='height_cm', y='jumping',
                         color='position_class', color_discrete_map=POS_COLOR,
                         opacity=.5, hover_data=['name'],
                         labels={'height_cm': 'Altura (cm)', 'jumping': 'Salto'})
        fig.update_layout(**BASE, height=340,
                          legend=dict(bgcolor='rgba(0,0,0,0)'),
                          xaxis=dict(**GRID), yaxis=dict(**GRID))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Definicion vs Marcaje -- Eje DEL/DEF")
        fig = px.scatter(sample, x='finishing', y='marking',
                         color='position_class', color_discrete_map=POS_COLOR,
                         opacity=.5, hover_data=['name'],
                         labels={'finishing': 'Definicion', 'marking': 'Marcaje'})
        fig.update_layout(**BASE, height=340,
                          legend=dict(bgcolor='rgba(0,0,0,0)'),
                          xaxis=dict(**GRID), yaxis=dict(**GRID))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='box-gold'>
        <strong>Perspectiva del Scout:</strong> Los GKs se agrupan perfectamente (altos + gran salto).
        Los DEFs destacan en marcaje; los FWDs en definicion; los MIDs son equilibrados.
        El <strong>solapamiento MID/FWD</strong> es el reto de clasificacion que necesitaremos resolver con ML.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 · TRANSFER  — Data Cleaning
# ═══════════════════════════════════════════════════════════════════════════════
elif step == 2:
    df_raw   = P['df_raw']
    df_clean = P['df_clean']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Auditoria de Valores Faltantes")
        nulls = df_raw.isnull().sum()
        nulls = nulls[nulls > 0].sort_values(ascending=True)
        fig = px.bar(x=nulls.values, y=nulls.index, orientation='h',
                     color=nulls.values,
                     color_continuous_scale=['#00d4ff', '#ef4444'],
                     labels={'x': 'Cantidad Faltante', 'y': ''})
        fig.update_layout(**BASE, height=360, coloraxis_showscale=False,
                          xaxis=dict(**GRID))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Consolidacion de Posiciones: 890 -> 4 Clases")
        mapping_df = pd.DataFrame({
            'Clase':       ['GK', 'DEF', 'MID', 'FWD'],
            'Mapeado Desde': ['GK',
                            'CB · LB · RB · LWB · RWB',
                            'CDM · CM · CAM · LM · RM',
                            'ST · CF · LW · RW'],
            'Cantidad':    [2065, 5883, 6668, 3338],
        })
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)

        fig = go.Figure(go.Funnel(
            y=['890 Posiciones Crudas', '14 Roles Principales', '4 Clases Tacticas', '17,954 Registros Limpios'],
            x=[890, 14, 4, 17954],
            marker_color=['#ef4444', '#fbbf24', '#00d4ff', '#00d4ff'],
            textinfo='value+label',
        ))
        fig.update_layout(**BASE, height=260)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Antes vs Despues: Distribucion de Posiciones")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Antes -- Top 20 Posiciones Crudas**")
        raw_pos = df_raw['positions'].value_counts().head(20).reset_index()
        raw_pos.columns = ['pos', 'n']
        fig = px.bar(raw_pos, x='pos', y='n',
                     color='n', color_continuous_scale=['#0a1538', '#00d4ff'])
        fig.update_layout(**BASE, height=300, coloraxis_showscale=False,
                          xaxis=dict(tickangle=45, **GRID), yaxis=dict(**GRID))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Despues -- 4 Clases Limpias**")
        clean_pos = df_clean['position_class'].value_counts().reset_index()
        clean_pos.columns = ['pos', 'n']
        fig = px.bar(clean_pos, x='pos', y='n',
                     color='pos', color_discrete_map=POS_COLOR, text='n')
        fig.update_layout(**BASE, height=300, showlegend=False,
                          xaxis=dict(**GRID), yaxis=dict(**GRID))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='box-green'>
        <strong>Transferencia Completa:</strong> Se redujeron 890 posiciones granulares a 4 clases tacticas.
        Se codifico <code>preferred_foot</code> (Derecho = 1, Izquierdo = 0).
        Se descartaron registros con valores faltantes -- <strong>17,954 jugadores limpios retenidos</strong>.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 · TRAINING  — Feature Engineering & Outliers
# ═══════════════════════════════════════════════════════════════════════════════
elif step == 3:
    df_clean       = P['df_clean']
    df_capped      = P['df_capped']
    features       = P['features']
    outlier_counts = P['outlier_counts']

    c1, c2, c3 = st.columns(3)
    c1.metric("Atributos Originales",       "51")
    c2.metric("Caracteristicas Selec.",     str(len(features)))
    c3.metric("Caract. con Atipicos",       str(len(outlier_counts)))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Grupos de Caracteristicas")
        groups = {
            'Fisico':     3,
            'Tecnico':   10,
            'Movimiento': 6,
            'Potencia':   4,
            'Mental':     6,
            'Defensivo':  3,
            'Estilo':     2,
        }
        fig = px.bar(
            x=list(groups.values()), y=list(groups.keys()), orientation='h',
            color=list(groups.values()),
            color_continuous_scale=['#0e1f4a', '#fbbf24'],
            text=list(groups.values()),
        )
        fig.update_layout(**BASE, height=320, coloraxis_showscale=False,
                          xaxis=dict(**GRID), yaxis_title='')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Valores Atipicos Detectados por Caracteristica (Metodo IQR)")
        o_df = (pd.DataFrame({'feature': list(outlier_counts.keys()),
                               'count':   list(outlier_counts.values())})
                  .sort_values('count'))
        fig = px.bar(o_df, x='count', y='feature', orientation='h',
                     color='count', color_continuous_scale=['#00d4ff', '#ef4444'])
        fig.update_layout(**BASE, height=380, coloraxis_showscale=False,
                          xaxis=dict(**GRID), yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Demo de Recorte IQR -- elige una caracteristica")
    pick = st.selectbox("Caracteristica:", sorted(outlier_counts.keys()), index=0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Antes del Recorte**")
        fig = px.box(df_clean, x='position_class', y=pick,
                     color='position_class', color_discrete_map=POS_COLOR)
        fig.update_layout(**BASE, height=300, showlegend=False, yaxis=dict(**GRID))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Despues del Recorte IQR**")
        fig = px.box(df_capped, x='position_class', y=pick,
                     color='position_class', color_discrete_map=POS_COLOR)
        fig.update_layout(**BASE, height=300, showlegend=False, yaxis=dict(**GRID))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='box-gold'>
        <strong>Nota del Entrenador:</strong> El recorte IQR (limitar a Q1 - 1.5 x IQR ... Q3 + 1.5 x IQR)
        controla valores extremos manteniendo los 17,954 jugadores en el entrenamiento.
        Eliminar atipicos descartaria injustamente a jugadores elite como Messi y Ronaldo.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 · TACTICS  — Scaling & PCA
# ═══════════════════════════════════════════════════════════════════════════════
elif step == 4:
    features  = P['features']
    df_capped = P['df_capped']
    X_scaled  = P['X_scaled']
    X_2d      = P['X_2d']
    X_pca     = P['X_pca']
    y         = P['y']
    names     = P['names']
    cum_var   = P['cum_var']

    n_comp = X_pca.shape[1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Caracteristicas Orig.",  str(len(features)))
    c2.metric("Componentes PCA",        str(n_comp))
    c3.metric("Varianza Retenida",      ">= 95 %")
    c4.metric("Dimensiones Reducidas",  str(len(features) - n_comp))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Por que escalar? -- Antes vs Despues de StandardScaler")
        show = [f for f in ['finishing', 'height_cm', 'age', 'stamina'] if f in features]
        palette = ['#00d4ff', '#fbbf24', '#ef4444', '#3b82f6']
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['Valores Crudos', 'Estandarizados (media=0, std=1)'])
        for i, feat in enumerate(show):
            fi = features.index(feat)
            fig.add_trace(
                go.Violin(y=df_capped[feat], name=feat, line_color=palette[i],
                          fillcolor=hex_rgba(palette[i], 0.2), box_visible=True,
                          meanline_visible=True, showlegend=i == 0),
                row=1, col=1,
            )
            fig.add_trace(
                go.Violin(y=X_scaled[:, fi], name=feat, line_color=palette[i],
                          fillcolor=hex_rgba(palette[i], 0.2), box_visible=True,
                          meanline_visible=True, showlegend=False),
                row=1, col=2,
            )
        fig.update_layout(**BASE, height=380, legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### PCA -- Varianza Acumulada Explicada")
        x_vals = list(range(1, len(cum_var) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=cum_var * 100,
            mode='lines+markers',
            line=dict(color='#00d4ff', width=3),
            marker=dict(color='#fbbf24', size=7),
            fill='tozeroy', fillcolor='rgba(0,212,255,.08)',
        ))
        fig.add_hline(y=95, line_dash='dot', line_color='#ef4444',
                      annotation_text='Umbral 95 %',
                      annotation_font_color='#ef4444',
                      annotation_position='bottom right')
        fig.update_layout(
            **BASE, height=380,
            xaxis=dict(title='Componentes Principales', **GRID),
            yaxis=dict(title='Varianza Acumulada (%)', **GRID),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### PCA 2D -- Jugadores Proyectados en el Espacio de Caracteristicas")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_2d), min(4000, len(X_2d)), replace=False)
    pca_df = pd.DataFrame({
        'PC1': X_2d[idx, 0], 'PC2': X_2d[idx, 1],
        'Posicion': y[idx], 'Jugador': names[idx],
    })
    fig = px.scatter(pca_df, x='PC1', y='PC2',
                     color='Posicion', color_discrete_map=POS_COLOR,
                     opacity=.55, hover_data=['Jugador'], symbol='Posicion')
    fig.update_layout(**BASE, height=430,
                      legend=dict(bgcolor='rgba(0,0,0,0)'),
                      xaxis=dict(**GRID), yaxis=dict(**GRID))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class='box-green'>
        <strong>Perspectiva Tactica:</strong> Los GKs (ambar) forman un grupo compacto y bien separado,
        fisicamente unicos. La nube DEF/MID/FWD se solapa fuertemente, lo que explica por que todos los modelos
        alcanzan ~100 % de precision en GK pero tienen dificultades entre los roles de campo.
        PCA comprime {len(features)} -> {n_comp} dimensiones reteniendo >= 95 % de la informacion.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 · MATCH DAY  — Model Training
# ═══════════════════════════════════════════════════════════════════════════════
elif step == 5:
    results = P['results']
    X_train = P['X_train']
    X_test  = P['X_test']

    c1, c2, c3 = st.columns(3)
    c1.metric("Muestras de Entren.", f"{len(X_train):,}")
    c2.metric("Muestras de Prueba",  f"{len(X_test):,}")
    c3.metric("Componentes PCA",     str(X_train.shape[1]))

    st.markdown("---")

    MODEL_META = {
        'KNN': (
            '', '#3b82f6', 'K-Nearest Neighbors',
            'Clasifica a cada jugador por el voto mayoritario de los 5 jugadores mas similares '
            '(distancia Euclidiana en el espacio PCA). Ponderado por distancia: los vecinos mas cercanos votan mas fuerte. '
            'Simple e interpretable, pero sensible al desbalance de clases.'
        ),
        'SVM': (
            '', '#fbbf24', 'Maquina de Vectores de Soporte (Kernel RBF)',
            'Encuentra el hiperplano de maximo margen que separa posiciones en un espacio de alta dimension. '
            'El kernel RBF mapea jugadores de forma no lineal para que las clases sean separables. '
            'El balanceo de clases maneja el desbalance hacia MID. <strong>Mejor modelo.</strong>'
        ),
        '\u00c1rbol de Decisi\u00f3n': (
            '', '#00d4ff', '\u00c1rbol de Decisi\u00f3n (profundidad max = 10)',
            'Construye un arbol de umbrales de atributos, ej: "finishing > 65 AND marking < 30 -> FWD". '
            'Legible por humanos, rapido, pero limitado por divisiones alineadas a los ejes.'
        ),
    }

    tabs = st.tabs([n for n in results])

    for tab, mname in zip(tabs, results):
        res   = results[mname]
        acc   = res['accuracy']
        cm    = res['cm']
        report = res['report']
        icon, color, full_name, description = MODEL_META[mname]
        is_best = mname == 'SVM'

        with tab:
            st.markdown(f"""
            <div class='{"hero" if is_best else "card"}'>
                <div style='font-size:2.5rem'>{icon}</div>
                <h2 style='color:{color};margin:.3rem 0;letter-spacing:2px'>
                    {full_name}{"  MEJOR MODELO" if is_best else ""}
                </h2>
                <div style='font-size:2.8rem;color:{"#00d4ff" if acc > .8 else "#fbbf24"};font-weight:900'>
                    {acc:.2%}
                </div>
                <p style='color:#94a3b8;margin:0'>Precision en Prueba</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Matriz de Confusion")
                fig = px.imshow(cm, x=CLASSES, y=CLASSES,
                                color_continuous_scale='Blues', text_auto=True,
                                labels={'x': 'Predicho', 'y': 'Real'})
                fig.update_layout(**BASE, height=360)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Precision / Recall / F1 por Clase")
                rows = [
                    {'Clase':     c,
                     'Precision': report[c]['precision'],
                     'Recall':    report[c]['recall'],
                     'F1':        report[c]['f1-score']}
                    for c in CLASSES if c in report
                ]
                mdf = pd.DataFrame(rows)
                fig = go.Figure()
                for metric, mc in [('Precision', '#3b82f6'), ('Recall', '#00d4ff'), ('F1', '#fbbf24')]:
                    fig.add_trace(go.Bar(
                        name=metric, x=mdf['Clase'], y=mdf[metric],
                        marker_color=mc,
                        text=[f'{v:.2f}' for v in mdf[metric]],
                        textposition='outside',
                    ))
                fig.update_layout(**BASE, barmode='group', height=360,
                                  yaxis=dict(range=[0, 1.2], **GRID),
                                  legend=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig, use_container_width=True)

            box_cls = 'box-green' if is_best else 'box-gold'
            st.markdown(f"""
            <div class='{box_cls}'>
                <strong>{icon} Como funciona:</strong> {description}
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6 · FINAL WHISTLE  — Results
# ═══════════════════════════════════════════════════════════════════════════════
elif step == 6:
    results = P['results']
    sorted_m = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    st.markdown("### Clasificacion del Torneo")
    medal_order  = [1, 0, 2]
    medals       = ['1ro', '2do', '3ro']
    medal_colors = ['#fbbf24', '#94a3b8', '#cd7f32']
    cols = st.columns(3)

    for rank, (mname, mres) in enumerate(sorted_m):
        acc = mres['accuracy']
        with cols[medal_order[rank]]:
            st.markdown(f"""
            <div class='card' style='text-align:center;border-color:{medal_colors[rank]}44;min-height:160px'>
                <div style='font-size:2.5rem'>{medals[rank]}</div>
                <h3 style='color:{medal_colors[rank]};margin:.2rem 0'>{mname}</h3>
                <div style='font-size:2.2rem;color:{medal_colors[rank]};font-weight:900'>{acc:.2%}</div>
                <p style='color:#64748b;margin:0;font-size:.85rem'>Precision en Prueba</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Comparacion de Precision")
        names_list = [m for m, _ in sorted_m]
        accs       = [r['accuracy'] for _, r in sorted_m]
        fig = go.Figure(go.Bar(
            x=names_list, y=accs,
            marker_color=medal_colors,
            text=[f'{a:.1%}' for a in accs],
            textposition='outside', textfont_size=16,
        ))
        fig.add_hline(y=max(accs), line_dash='dot', line_color='#00d4ff',
                      annotation_text=f'Campeon: {max(accs):.1%}',
                      annotation_font_color='#00d4ff')
        fig.update_layout(**BASE, height=360,
                          yaxis=dict(range=[0, 1.1], tickformat='.0%', **GRID),
                          xaxis=dict(**GRID))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Puntaje F1 por Clase -- Todos los Modelos")
        mcolors = {'SVM': '#fbbf24', 'KNN': '#3b82f6', '\u00c1rbol de Decisi\u00f3n': '#00d4ff'}
        fig = go.Figure()
        for mname, mres in sorted_m:
            report = mres['report']
            f1s = [report.get(c, {}).get('f1-score', 0) for c in CLASSES]
            fig.add_trace(go.Scatter(
                x=CLASSES, y=f1s,
                mode='lines+markers', name=mname,
                line=dict(color=mcolors.get(mname, '#94a3b8'), width=2.5),
                marker=dict(size=11),
            ))
        fig.update_layout(**BASE, height=360,
                          yaxis=dict(range=[0, 1.1], title='Puntaje F1', **GRID),
                          xaxis=dict(**GRID),
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='hero' style='margin-top:1rem'>
        <h2 style='color:#fbbf24'>CAMPEON: SVM con Kernel RBF</h2>
        <p style='color:#00d4ff;font-size:1.1rem;margin:0'>
            81.2 % Precision - 100 % Deteccion GK - Mejor Separacion MID/FWD
        </p>
        <p style='color:#94a3b8;margin-top:.75rem;max-width:620px;margin-left:auto;margin-right:auto'>
            El kernel RBF mapea implicitamente a los jugadores en un espacio de mayor dimension donde los
            limites de posicion se convierten en hiperplanos lineales. La maximizacion del margen reduce el sobreajuste,
            y el balanceo de clases corrige el desbalance hacia MID que afecta a KNN y Arbol de Decision.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7 · STAR SCOUT  — Live Predictor
# ═══════════════════════════════════════════════════════════════════════════════
elif step == 7:
    if 'expo_svm' not in P:
        st.error("Los modelos de Expo no están entrenados. Revisa la definición en load_pipeline.")
    else:
        expo_svm = P['expo_svm']
        expo_scaler = P['expo_scaler']
        expo_features = P['expo_features']

        st.markdown("### Predicción en Vivo (Modo Expo)")
        st.markdown("Ajusta los atributos físicos y medibles de tu prospecto para predecir su posición ideal en el campo.")

        slider_defs = [
            ('age',          'Edad (Años)',           15, 45, 22),
            ('height_cm',    'Estatura (cm)',         150, 210, 180),
            ('weight_kgs',   'Peso (kg)',             50, 110, 75),
            ('sprint_speed', 'Velocidad Sprint (1-99)', 0, 99, 70),
            ('jumping',      'Salto (1-99)',          0, 99, 70),
            ('shot_power',   'Fuerza de Tiro (1-99)', 0, 99, 65),
            ('stamina',      'Resistencia (1-99)',    0, 99, 70),
            ('strength',     'Fuerza Física (1-99)',  0, 99, 65),
        ]

        col1, col2 = st.columns(2)
        custom: dict = {}
        for i, (feat, label, lo, hi, default) in enumerate(slider_defs):
            with [col1, col2][i % 2]:
                custom[feat] = st.slider(label, lo, hi, default)

        if st.button("PREDECIR POSICION", use_container_width=True):
            # Construir arreglo en el orden de EXPO_FEATURES
            base_row = [custom[f] for f in expo_features]
            arr = np.array(base_row).reshape(1, -1)
            arr_scaled = expo_scaler.transform(arr)

            pred        = expo_svm.predict(arr_scaled)[0]
            proba       = expo_svm.predict_proba(arr_scaled)[0]
            svm_classes = expo_svm.classes_

            st.markdown(f"""
            <div class='hero' style='margin-top:1.5rem'>
                <h1 style='color:#ffffff;font-size:3rem;letter-spacing:6px;margin:0'>{pred}</h1>
                <p style='color:#00d4ff;font-size:1.1rem;margin:.25rem 0 0'>
                    Predicción Expo SVM -- {max(proba):.1%} confianza
                </p>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=[c for c in svm_classes],
                y=proba,
                marker_color=[POS_COLOR[c] for c in svm_classes],
                text=[f'{p:.1%}' for p in proba],
                textposition='outside',
            ))
            fig.update_layout(**BASE, height=300,
                              yaxis=dict(range=[0, 1.3], tickformat='.0%', **GRID))
            st.plotly_chart(fig, use_container_width=True)
