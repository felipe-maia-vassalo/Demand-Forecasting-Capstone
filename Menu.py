from __future__ import annotations

import streamlit as st

from core.i18n import tr
from core.sample import load_sample

st.set_page_config(page_title="Demand Forecasting · Previsão & PCP", page_icon="🧭", layout="wide")
languages = ["English", "Português"]
st.session_state.setdefault("language_picker", st.session_state.get("language", "English"))
st.session_state["language"] = st.sidebar.selectbox(
    "Language / Idioma", languages,
    key="language_picker",
)


def home_page():
    st.title(tr("Previsão de demanda e planejamento da produção"))
    st.write(tr("Compare modelos de previsão e transforme a demanda prevista em um plano mestre de produção mensal."))
    st.markdown(tr("Explore tendência e sazonalidade, compare modelos em um horizonte de 6, 8 ou 12 meses e avalie como a previsão afeta estoques, lotes e custos."))
    st.subheader(tr("Experimente com dados fictícios"))
    st.write(tr("Explore seis anos de demanda mensal de cadeiras escolares, de janeiro de 2020 a dezembro de 2025. Os 72 valores são sintéticos."))
    if st.button(tr("Carregar exemplo e explorar"), type="primary"):
        load_sample(st.session_state)
        st.switch_page("pages/02_Serie_Temporal.py")
    st.page_link("pages/01_Upload.py", label=tr("Enviar minha própria série"), icon="📤")


workflow = [
    ("pages/01_Upload.py", "Dados", "📤"),
    ("pages/02_Serie_Temporal.py", "Série temporal", "📈"),
    ("pages/03_Analise_Detalhada.py", "Diagnóstico detalhado", "🔍"),
    ("pages/04_Previsao.py", "Previsão", "🔮"),
    ("pages/05_Inputs_MPS.py", "Parâmetros de produção", "⚙️"),
    ("pages/06_MPS.py", "Plano de produção", "🗓️"),
    ("pages/07_Dashboard_Conclusao.py", "Painel de decisão", "📊"),
]
pages = [st.Page(home_page, title=tr("Início"), icon="🧭", default=True)]
pages.extend(st.Page(path, title=tr(label), icon=icon) for path, label, icon in workflow)
page = st.navigation(pages)
if st.session_state.get("series_source") == "sample":
    st.sidebar.caption(tr("Exemplo sintético · 72 meses"))
page.run()
