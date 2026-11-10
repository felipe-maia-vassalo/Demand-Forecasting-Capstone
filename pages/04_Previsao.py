# -*- coding: utf-8 -*-
from __future__ import annotations
from core.i18n import tr

import time

import altair as alt
import pandas as pd
import streamlit as st
from core import pipeline as pipe

st.set_page_config(page_title=tr('Previsão'), page_icon="🔮", layout="wide")
st.title(tr('🔮 Previsão'))

ss = st.session_state
ss.setdefault("last_result", None)

# ===== série do upload
if not ss.get("upload_ok"):
    st.warning(tr('Nenhuma série encontrada. Volte ao Passo 1 (Upload).'))
    st.page_link("pages/01_Upload.py", label=tr('Ir para o Passo 1 — Upload'))
    st.stop()

_ts = ss.get("ts_df_norm")
if not isinstance(_ts, pd.DataFrame) or not {"ds", "y"}.issubset(_ts.columns):
    st.error(tr("Formato inesperado: esperado DataFrame com colunas ['ds','y']."))
    st.stop()

# ===== helpers de datas
_PT_MON2NUM = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
def _label_to_month_start(v) -> pd.Timestamp:
    if isinstance(v, pd.Timestamp): return v
    s = str(v)
    try:
        if "/" in s:
            mon, yy = s.split("/")
            return pd.Timestamp(year=2000 + int(yy), month=_PT_MON2NUM.get(mon, 1), day=1)
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

# série mensal contínua (MS)
_idx = _ts["ds"].map(_label_to_month_start)
s_monthly = (
    pd.Series(_ts.loc[_idx.notna(), "y"].astype(float).to_numpy(), index=_idx[_idx.notna()])
      .sort_index().asfreq("MS").interpolate("linear").bfill().ffill()
)

# A small search is useful for an interactive demo. Each run owns its settings.
sample_loaded = ss.get("series_source") == "sample"
st.sidebar.header(tr("⚙️ Configurações"))
HORIZON = st.sidebar.selectbox(tr("Horizonte (meses)"), [6, 8, 12], key="forecast_horizon_widget")
quick_search = st.sidebar.checkbox(tr("Busca rápida"), value=sample_loaded,
                                   help=tr("Compara uma grade menor de modelos usando o mesmo pipeline. Desative para executar a busca completa."))
do_log = st.sidebar.checkbox(tr("Aplicar log"), value=not sample_loaded)
do_bootstrap = st.sidebar.checkbox(tr("Ativar bootstrap"), value=not sample_loaded)
replicates = st.sidebar.slider(tr("Réplicas do bootstrap"), 1, 50, 20) if do_bootstrap else 0

if st.button(tr("▶️ Rodar previsão"), type="primary"):
    try:
        start = time.perf_counter()
        rounds = 1 + int(do_log) + replicates
        total_stages = (5 + int(pipe.KERAS_AVAILABLE)) * rounds
        stages = []
        progress = st.progress(0)
        console = st.expander(tr("📜 Console (passos essenciais)"))

        def report_model(preprocessing, model):
            stages.append(model)
            message = tr("Comparando {0}", model) + f" ({preprocessing})"
            progress.progress(min((len(stages) - 1) / total_stages, 1.0), text=message)
            console.write(message)

        with st.spinner(tr("Executando pipeline…")):
            result = pipe.run_full_pipeline(
                s_monthly, horizon=HORIZON, seasonal_period=12,
                do_log=do_log, do_bootstrap=do_bootstrap,
                n_bootstrap=replicates, quick_search=quick_search, on_model=report_model,
            )
        progress.progress(1.0, text=tr("100% — concluído"))
        ss.last_result = result
        ss["experiments_df"] = result
        ss["forecast_committed"] = False
        for key in ("forecast_df", "forecast_h", "mps_table", "mps_detail",
                    "mps_tbl_display", "mps_costs_baseline"):
            ss.pop(key, None)
        st.caption(tr("⏱️ Tempo de execução: {0}s", f"{time.perf_counter() - start:.1f}"))
    except Exception as error:
        st.error(tr("Falha ao executar a previsão. Traceback abaixo:"))
        st.exception(error)

# ===== render
res = ss.get("last_result")
if res is not None:
    # ----- Campeão
    champ = {}
    if hasattr(res, "attrs"):
        champ = res.attrs.get("champion", {}) or {}
    elif isinstance(res, dict):
        champ = res.get("champion", {}) or {}

    modelo_nome = champ.get("model", tr('Desconhecido'))
    if modelo_nome.upper() == "SARIMAX":
        modelo_nome = "SARIMA"

    st.subheader(tr('🏆 Modelo Campeão: {0}', modelo_nome))

    def _fmt(x):
        try: return f"{float(x):.4g}"
        except Exception: return str(x)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("MAE", _fmt(champ.get("MAE")))
    c2.metric("sMAPE (%)", _fmt(champ.get("sMAPE")))
    c3.metric("RMSE", _fmt(champ.get("RMSE")))
    c4.metric("MAPE (%)", _fmt(champ.get("MAPE")))

    st.caption(tr('Parâmetros do modelo campeão:'))
    st.json({
        "preprocess": champ.get("preprocess"),
        "preprocess_params": champ.get("preprocess_params"),
        "model_params": champ.get("model_params"),
    })

    # ===== PREVISÃO do pipeline (sem fallback)
    forecast = None
    if hasattr(res, "attrs"):
        forecast = res.attrs.get("forecast_df", None)
        if forecast is None:
            # se o pipeline reportou erro, exibir claramente
            f_err = res.attrs.get("forecast_error")
            if f_err:
                st.error(tr('Pipeline não conseguiu gerar previsão do campeão: {0}', f_err))
            else:
                st.error(tr("Pipeline não retornou 'forecast_df'."))
    elif isinstance(res, dict):
        forecast = res.get("forecast_df", None)

    if forecast is None:
        st.stop()

    # Normaliza para ['ds','y']
    if isinstance(forecast, pd.DataFrame):
        if {"ds","yhat"}.issubset(forecast.columns):
            forecast_df_std = forecast.rename(columns={"yhat":"y"})[["ds","y"]].copy()
        elif {"ds","y"}.issubset(forecast.columns):
            forecast_df_std = forecast[["ds","y"]].copy()
        else:
            st.error(tr("Formato inesperado de 'forecast_df'. Esperado colunas ['ds','y'] ou ['ds','yhat']."))
            st.stop()
    elif isinstance(forecast, pd.Series):
        forecast_df_std = pd.DataFrame({"ds": forecast.index, "y": forecast.values})
    else:
        st.error(tr("Tipo inesperado de 'forecast_df' retornado pelo pipeline."))
        st.stop()

    # ===== Gráfico Real + Previsão (Altair)
    st.subheader(tr('📈 Histórico + Previsão'))
    hist_df = pd.DataFrame({"ds": s_monthly.index, "valor": s_monthly.values, "série": tr('Real')})
    prev_df = pd.DataFrame({"ds": pd.to_datetime(forecast_df_std["ds"]), "valor": forecast_df_std["y"].values, "série": tr('Previsão')})
    plot_long = pd.concat([hist_df, prev_df], ignore_index=True)

    chart = (
        alt.Chart(plot_long.reset_index(drop=True))
        .mark_line()
        .encode(
            x=alt.X("ds:T", title=tr('Mês')),
            y=alt.Y("valor:Q", title=tr('Quantidade')),
            color=alt.Color(
                "série:N",
                scale=alt.Scale(domain=[tr('Real'),tr('Previsão')], range=["#1e3a8a", "#60a5fa"]),
                legend=alt.Legend(title=None, orient="top")
            ),
            tooltip=[alt.Tooltip("ds:T", title=tr('Período')),
                     alt.Tooltip("série:N", title=tr('Série')),
                     alt.Tooltip("valor:Q", title=tr('Valor'), format=",.0f")]
        )
        .properties(height=280, width="container")
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    # ===== Experimentos — botão de download (sem mostrar tabela)
    st.subheader(tr('📦 Experimentos'))
    try:
        exp_df = res.reset_index(drop=True)
        st.session_state["experiments_df"] = exp_df.copy()
        csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            tr('⬇️ Baixar todos os experimentos (CSV)'),
            data=csv_bytes,
            file_name="experimentos_previsao.csv",
            mime="text/csv",
            help=tr('Contém todas as combinações testadas com métricas e parâmetros.')
        )
    except Exception:
        st.info(tr('Resultados dos experimentos indisponíveis para exportação.'))

    # =============================
    # 🔗 Próximos passos
    # =============================
    st.divider()
    st.subheader(tr('➡️ Próximos passos'))

    # CSS: afinar botão primário
    st.markdown("""
    <style>
    div.stButton > button[kind="primary"]{
        padding: 0.45rem 0.9rem !important;
        font-size: 0.95rem !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Botão Salvar para MPS (usa exatamente o horizonte selecionado)
    can_save = forecast_df_std is not None and len(forecast_df_std) > 0
    if st.button(tr('Salvar previsão para o MPS'), type="primary", disabled=not can_save):
        st.session_state["forecast_df"] = forecast_df_std.copy()
        st.session_state["forecast_h"] = len(forecast_df_std)
        st.session_state["forecast_committed"] = True
        st.success(tr('Previsão salva para o MPS. Agora você pode ir aos Inputs do MPS.'))

    st.markdown("---")
    st.page_link("pages/05_Inputs_MPS.py", label=tr('➡️ Ir para Inputs do MPS'))

    if not st.session_state.get("forecast_committed", False):
        st.info(tr('Clique em **Salvar previsão para o MPS** antes de avançar aos Inputs.'), icon="ℹ️")
