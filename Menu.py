# Menu.py
from __future__ import annotations
import streamlit as st

from core.i18n import tr
from core.sample import load_sample, sample_workbook

initial_language = "English" if st.query_params.get("lang") == "en" else "Português"
st.session_state.setdefault("language", initial_language)
st.set_page_config(page_title=tr("Previsão & PCP"), page_icon="🧭", layout="wide")
st.session_state.setdefault("language_picker", st.session_state["language"])


def change_language():
    st.session_state["language"] = st.session_state["language_picker"]


st.sidebar.selectbox("Idioma / Language", ["Português", "English"],
                     key="language_picker", on_change=change_language)

# --------- Home como FUNÇÃO (não arquivo) ---------
def home_page():
    st.title(tr("🧭 Previsão & PCP"))
    st.subheader(tr("Integração entre Modelos de Previsão e PCP"))

    st.markdown(tr("""
        ### O que é  
        Artefato desenvolvido para **gerar previsões de demanda** a partir de modelos **clássicos e de *Machine Learning***, integrando os resultados às ferramentas tradicionais de **PCP**. O sistema também possibilita a criação de **dashboards executivos** que auxiliam a **análise de resultados** e a **tomada de decisão** de forma simples e visual.

        ---

        ### Benefícios  
        - **Apoio tecnológico** para geração de previsões consistentes e embasadas em métodos validados na literatura.  
        - **Integração automática** entre previsão, planejamento (MPS) e indicadores.  
        - **Fluxo contínuo e intuitivo:**  🧾 **DADOS → 🤖 PREVER → 🏭 PLANEJAR**.  

        ---

        ### Principais Outputs  
        - 📈 **Previsão de demanda** para os próximos **6, 8 ou 12 meses**.  
        - 🗓️ **MPS** (Master Production Schedule).  
        - 📊 **Outputs executivos** para visualização consolidada e apoio à decisão.  
        """))

    st.divider()
    st.markdown(tr("### Comece agora"))
    if st.button(tr("Iniciar — Upload da Série"), type="primary"):
        try:
            st.switch_page("pages/01_Upload.py")
        except Exception:
            st.info(tr("Use o menu lateral e clique em **Upload**."))

# --------- Navegação com títulos/ícones custom ---------
nav = st.navigation([
    st.Page(home_page,                          title=tr("Menu"), icon="🧭"),  # <- função
    st.Page("pages/01_Upload.py",               title=tr("Upload"),                   icon="📤"),
    st.Page("pages/02_Serie_Temporal.py",       title=tr("Série Temporal"),          icon="📈"),
    st.Page("pages/03_Analise_Detalhada.py",    title=tr("Análise Detalhada"),       icon="🔍"),
    st.Page("pages/04_Previsao.py",             title=tr("Previsão"),                 icon="🔮"),
    st.Page("pages/05_Inputs_MPS.py",           title=tr("Inputs do MPS"),           icon="⚙️"),
    st.Page("pages/06_MPS.py",                  title=tr("MPS"),      icon="🗓️"),
    st.Page("pages/07_Dashboard_Conclusao.py",  title=tr("Conclusão"),   icon="📊"),
])

with st.sidebar.expander(tr("Série de exemplo")):
    st.caption(tr("Explore seis anos de demanda mensal de cadeiras escolares, de janeiro de 2020 a dezembro de 2025. Os 72 valores são sintéticos."))
    if st.button(tr("Carregar série de exemplo")):
        load_sample(st.session_state)
        st.switch_page("pages/02_Serie_Temporal.py")
    st.download_button(tr("Baixar exemplo em Excel"), sample_workbook(),
                       file_name="synthetic_classroom_chair_demand.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

nav.run()
