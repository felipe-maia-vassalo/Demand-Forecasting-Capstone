# -*- coding: utf-8 -*-
from __future__ import annotations
from core.i18n import tr, english

import math
import re
import time
import traceback

import altair as alt
import pandas as pd
import streamlit as st

from core import pipeline as pipe

st.set_page_config(page_title=tr('Previsão'), page_icon="🔮", layout="wide")
st.title(tr('🔮 Previsão'))

ss = st.session_state
ss.setdefault("is_running", False)
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

sample_mode = ss.get("series_source") == "sample"
grid = pipe.search_grid(sample_mode)
sizes = {
    "Croston": len(grid["croston_alphas"]),
    "SBA": len(grid["sba_alphas"]),
    "TSB": len(grid["tsb_alpha_grid"]) * len(grid["tsb_beta_grid"]),
    "RF": (len(grid["rf_lags_grid"]) * len(grid["rf_n_estimators_grid"])
           * len(grid["rf_max_depth_grid"])),
    "SARIMAX": math.prod(len(options) for options in grid["sarima_grid"].values()),
}
# ===== form (config)
with st.form(key="previsao_form"):
    st.sidebar.header(tr('⚙️ Configurações'))

    # Horizonte da previsão
    HORIZON = st.sidebar.selectbox(tr('Horizonte (meses)'), [6, 8, 12], index=0)

    # Controles de LOG/BOOTSTRAP
    DO_LOG = st.sidebar.checkbox(tr('Aplicar log'), value=not sample_mode)
    DO_BOOTSTRAP = st.sidebar.checkbox(tr('Ativar bootstrap'), value=not sample_mode)

    # Salva só o horizonte (sem FAST_MODE)
    ss["HORIZON"] = int(HORIZON)

    # Limites das réplicas do bootstrap (sempre os “completos”)
    max_boot, default_boot = 50, 20

    if DO_BOOTSTRAP:
        N_BOOTSTRAP = st.sidebar.slider(
            tr('Réplicas do bootstrap'),
            min_value=1,
            max_value=max_boot,
            value=default_boot,
            step=1,
            help=tr('Reduzir as réplicas acelera a execução, mas pode não encontrar o melhor modelo ideal.')
        )
        st.sidebar.caption(
            tr('⚠️ **Menos réplicas** → roda mais rápido, mas **pode não encontrar** o melhor modelo.')
        )
    else:
        N_BOOTSTRAP = 0

    # Exibe o tamanho da grade usada nesta execução.
    base = sum(sizes.values())

    rounds_desc = ["original"]
    if DO_LOG:
        rounds_desc.append("log")
    if DO_BOOTSTRAP:
        rounds_desc.append(f"bootstrap×{N_BOOTSTRAP}")

    st.caption(
        tr('Configuração: combinações≈{0} | rodadas: {1}', max(1, base), ', '.join(rounds_desc))
    )

    submitted = st.form_submit_button(
        tr('▶️ Rodar previsão'),
        type="primary",
        disabled=ss.is_running,
    )

# ===== console filtrado + progresso acumulativo (rodadas/famílias)
prog = st.progress(0)
prog_text = st.empty()
log_box = st.expander(tr('📜 Console (passos essenciais)'), expanded=False)
log_area = log_box.empty()
progress_lines = []
round_count = 1 + int(DO_LOG) + N_BOOTSTRAP
total_weight = max(1, base * round_count)
completed_weight = 0


# Filter the original pipeline console before translating its display text.
console_filter = re.compile(
    r"PIPELINE INICIADO|^Params:|Realizando testes da série|^→ |bootstrap|"
    r"===== CAMPEÃO|PIPELINE FINALIZADO|ERROR|EXCEPTION|Traceback", re.I
)
log_phrases = {
    "PIPELINE INICIADO": "PIPELINE STARTED", "PIPELINE FINALIZADO": "PIPELINE FINISHED",
    "Linhas totais de experimentos": "Total experiment rows",
    "Realizando testes da série ORIGINAL": "Testing the original series",
    "Realizando testes da série LOG-transformada": "Testing the log-transformed series",
    "Realizando testes da SÉRIE SINTÉTICA": "Testing synthetic series",
    "CAMPEÃO (critério: menor MAE; desempates por RMSE/soma/simplicidade)":
        "BEST MODEL (lowest MAE; ties resolved by RMSE/sum/simplicity)",
    "combinações": "combinations", "Testes": "Tests", "início": "start",
    "fim em": "finished in", "Gerando": "Generating", "réplicas bootstrap": "bootstrap replicates",
    "Concluídos testes": "Completed tests", "linhas": "rows", "progresso": "progress",
}


def show_message(message):
    if not console_filter.search(message):
        return
    if english():
        for pt, en in log_phrases.items():
            message = message.replace(pt, en)
    progress_lines.append(message)
    log_area.text("\n".join(progress_lines[-250:]))


def report_model(preprocessing, model):
    global completed_weight
    family = "RF" if model == "Random Forest" else model
    message = tr("Comparando {0}", model) + f" ({preprocessing})"
    progress = min(completed_weight / total_weight, 1.0)
    prog.progress(progress)
    prog_text.write(f"{progress:.0%} — {message}")
    completed_weight += sizes.get(family, 0)


# ===== execução (sempre reroda ao clicar)
if submitted and not ss.is_running:
    ss.is_running = True
    ss.last_result = None
    try:
        debug_info = {
            "horizon": HORIZON, "seasonal_period": 12, "do_original": True,
            "do_log": DO_LOG, "do_bootstrap": DO_BOOTSTRAP,
            "n_bootstrap": N_BOOTSTRAP, "FAST_MODE": sample_mode,
            "grid_sizes": sizes,
        }
        with st.expander(tr('🔎 Depuração — parâmetros efetivos'), expanded=False):
            st.json(debug_info)

        t0 = time.perf_counter()
        with st.spinner(tr('Executando pipeline…')), pipe.capture_run_log(show_message):
            resultados = pipe.run_full_pipeline(
                data_input=s_monthly, horizon=int(HORIZON), seasonal_period=12,
                do_original=True, do_log=DO_LOG, do_bootstrap=DO_BOOTSTRAP,
                n_bootstrap=N_BOOTSTRAP, bootstrap_block=24, save_dir=None,
                quick_search=sample_mode, on_model=report_model,
            )
        elapsed = time.perf_counter() - t0
        prog.progress(100); prog_text.write(tr('100% — concluído'))
        st.caption(tr('⏱️ Tempo de execução: {0}s', f'{elapsed:.1f}'))

        # =========================================================
        # Salva resultado “bruto” (experimentos) e artefatos úteis
        # =========================================================
        for key in ("forecast_df", "forecast_h", "mps_table", "mps_detail",
                    "mps_tbl_display", "mps_costs_baseline"):
            ss.pop(key, None)
        ss["forecast_committed"] = False
        ss.last_result = resultados

        # Experimentos (opcional para a página 07 e download)
        exp_df = None
        if isinstance(resultados, pd.DataFrame) and not resultados.empty:
            exp_df = resultados.copy()
        elif hasattr(resultados, "attrs"):
            for key in ["experiments", "experiments_df", "resultados", "table", "exp"]:
                val = resultados.attrs.get(key)
                if isinstance(val, pd.DataFrame) and not val.empty:
                    exp_df = val.copy(); break
        if exp_df is None and isinstance(resultados, dict):
            for key in ["experiments", "experiments_df", "resultados", "table", "exp"]:
                val = resultados.get(key)
                if isinstance(val, pd.DataFrame) and not val.empty:
                    exp_df = val.copy(); break

        if isinstance(exp_df, pd.DataFrame) and not exp_df.empty:
            ss["exp_table"] = exp_df.copy()
            ss["experiments_df"] = exp_df.copy()
            ss["experiments_csv"] = exp_df.to_csv(index=False).encode("utf-8")
            ss["experiments_ready"] = True
        else:
            ss["experiments_ready"] = False

    except Exception:
        st.error(tr('Falha ao executar a previsão. Traceback abaixo:'))
        st.code("\n".join(traceback.format_exc().splitlines()), language="text")
    finally:
        ss.is_running = False

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
