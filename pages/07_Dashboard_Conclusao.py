# pages/07_Dashboard_Conclusao.py
from __future__ import annotations
from core.i18n import tr, english, month_label
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title=tr('Conclusão'), page_icon="✅", layout="wide")
st.title(tr('✅ Conclusão (Painel de Decisão)'))

# -----------------------------
# Helpers
# -----------------------------
def _safe_num(x, nd=2):
    try:
        v = float(x)
        if np.isnan(v): return "—"
        if abs(v) >= 1000 and english(): return f"{v:,.0f}"
        if abs(v) >= 1000:
            return f"{v:,.0f}".replace(",", ".")
        return f"{v:.{nd}f}"
    except Exception:
        return "—"

def _to_ts(x):
    # aceita 'Set/25', Timestamp ou string ISO
    if isinstance(x, pd.Timestamp):
        return x
    s = str(x)
    # label tipo "Set/25"
    _PT = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
    if "/" in s and len(s) in (6,7):
        mon, yy = s.split("/")
        try:
            return pd.Timestamp(year=2000+int(yy), month=_PT.get(mon.title(), 1), day=1)
        except Exception:
            pass
    # ISO fallback
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

def _kpi(label, value, help_text=None, key=None):
    c = st.container()
    with c:
        st.metric(label, value)
        if help_text:
            st.caption(help_text)
    return c

# -----------------------------
# Recuperos de memória
# -----------------------------
ss = st.session_state
res = ss.get("last_result")                   # resultado da previsão (dict-like com attrs)
fcst_df = ss.get("forecast_df")              # df salvo na pág. 04 (ds,y)
hist_df_norm = ss.get("ts_df_norm")          # upload normalizado (ds,y)
mps_tbl_display = ss.get("mps_table")        # tabela do MPS (a de exibição)
mps_detail = ss.get("mps_detail")            # detalhe do core (se você decidir guardar)
# Recupera tabela de experimentos (forma robusta)
exp_df = None
for key in ["experiments_df", "experiments_table", "pipeline_experiments"]:
    val = ss.get(key)
    if isinstance(val, pd.DataFrame) and not val.empty:
        exp_df = val
        break


tabs = st.tabs([tr('📊 Acurácia'), tr('🧭 Viéses'), "🏭 MPS & KPIs", tr('💡 Recomendações')])

# ======================================================
# TAB 1 — ACURÁCIA (cards enxutos + gráfico limpo)
# ======================================================
with tabs[0]:
    st.subheader(tr('Desempenho dos modelos de previsão'))

    # ----------------- Pega modelo campeão -----------------
    champion = {}
    if res is not None and hasattr(res, "attrs"):
        champion = res.attrs.get("champion", {}) or {}

    modelo_nome = champion.get("model", tr('Desconhecido'))
    preprocess = champion.get("preprocess", "—")
    model_params = champion.get("model_params", "—")

    st.markdown(tr('**🏆 Modelo Campeão:** {0}', modelo_nome))
    st.caption(tr('Pré-processamento: `{0}` — Parâmetros: `{1}`', preprocess, model_params))

    st.markdown("---")

    # ----------------- Gráfico Real x Previsão -----------------
    hist = None
    if isinstance(hist_df_norm, pd.DataFrame) and {"ds","y"}.issubset(hist_df_norm.columns):
        hist = hist_df_norm.copy()
        hist["ds"] = hist["ds"].apply(_to_ts)
        hist = hist.dropna(subset=["ds"]).rename(columns={"y":tr('Real')})

    prev = None
    if isinstance(fcst_df, pd.DataFrame) and {"ds","y"}.issubset(fcst_df.columns):
        prev = fcst_df.copy()
        prev["ds"] = prev["ds"].apply(_to_ts)
        prev = prev.dropna(subset=["ds"]).rename(columns={"y":tr('Previsão')})

    if hist is None:
        st.info(tr('Sem histórico em memória. Gere o upload na página **01_Upload**.'))
    else:
        plot_df = pd.DataFrame({"ds": hist["ds"], "série": tr('Real'), "valor": hist[tr('Real')]})
        if prev is not None and len(prev) > 0:
            plot_df = pd.concat([
                plot_df,
                pd.DataFrame({"ds": prev["ds"], "série": tr('Previsão'), "valor": prev[tr('Previsão')]})
            ], ignore_index=True)

        import altair as alt
        chart = (
            alt.Chart(plot_df)
            .mark_line()
            .encode(
                x=alt.X("ds:T", title=tr('Mês')),
                y=alt.Y("valor:Q", title=tr('Quantidade')),
                color=alt.Color(
                    "série:N",
                    scale=alt.Scale(domain=[tr('Real'),tr('Previsão')], range=["#1e3a8a", "#60a5fa"]),
                    legend=alt.Legend(title=None, orient="top")
                ),
                tooltip=[
                    alt.Tooltip("ds:T", title=tr('Período')),
                    alt.Tooltip("série:N", title=tr('Série')),
                    alt.Tooltip("valor:Q", title=tr('Valor'), format=",.0f"),
                ]
            )
            .properties(height=360, width="container")
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # ----------------- Métricas e Avaliação -----------------
    mae = champion.get("MAE")
    smape = champion.get("sMAPE")
    rmse = champion.get("RMSE")
    mape = champion.get("MAPE")

    st.markdown(tr('### 📊 Métricas de desempenho'))

    def _avaliar_mae(v):
        if v is None or np.isnan(v): return "—"
        if v < 10: return tr('Excelente precisão (erro médio muito baixo).')
        elif v < 30: return tr('Boa precisão — previsão próxima da realidade.')
        elif v < 60: return tr('Precisão moderada — há flutuações relevantes.')
        else: return tr('Erro alto — revisar modelo e possíveis outliers.')

    def _avaliar_smape(v):
        if v is None or np.isnan(v): return "—"
        if v < 10: return tr('Muito bom (erro percentual simétrico muito baixo).')
        elif v < 20: return tr('Bom desempenho geral.')
        elif v < 40: return tr('Erro moderado — previsão aceitável, mas pode melhorar.')
        else: return tr('Erro alto — previsão instável ou sazonalidade não capturada.')

    def _avaliar_rmse(v):
        if v is None or np.isnan(v): return "—"
        return tr('RMSE mede a **dispersão dos erros** — quanto menor, mais consistente a previsão.')

    def _avaliar_mape(v):
        if v is None or np.isnan(v): return "—"
        if v < 10: return tr('Excelente (erro médio abaixo de 10%).')
        elif v < 20: return tr('Bom (erro entre 10–20%).')
        elif v < 30: return tr('Atenção — erro considerável, revisar tendência.')
        else: return tr('Ruim — erro alto, revisar modelo e dados.')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", _safe_num(mae))
    c2.metric("sMAPE (%)", _safe_num(smape))
    c3.metric("RMSE", _safe_num(rmse))
    c4.metric("MAPE (%)", _safe_num(mape))

    st.markdown(tr('#### 🧠 Interpretação das métricas'))
    st.caption(f"**MAE:** {_avaliar_mae(mae)}")
    st.caption(f"**sMAPE:** {_avaliar_smape(smape)}")
    st.caption(f"**RMSE:** {_avaliar_rmse(rmse)}")
    st.caption(f"**MAPE:** {_avaliar_mape(mape)}")

    st.info(
    tr('Erros mais altos **não significam necessariamente** que a previsão esteja ruim. Eles podem indicar que a série temporal é **complexa e/ou intermitente** (com muitos zeros, picos e lacunas). Nesses casos, vale complementar com **pré-processamento** e técnicas específicas'))

    st.divider()

# ======================================================
# TAB 2 — VIESES (com conclusão automática)
# ======================================================
with tabs[1]:
    st.subheader(tr('Diagnóstico de vieses da previsão'))

    st.markdown(tr('\n    > O **viés** de previsão mede se o modelo tende a **superestimar** ou **subestimar** os valores reais ao longo do tempo.  \n    > Quando o viés é **positivo**, o modelo prevê valores maiores do que o realizado; quando é **negativo**, prevê valores menores.  \n    > Um modelo sem viés apresenta erros que oscilam em torno de zero, indicando previsões equilibradas sem tendência sistemática.\n    '))


    # 1) Tenta recuperar um backtest com ds, y_true, y_pred
    bt = None
    if res is not None and hasattr(res, "attrs"):
        for k in ["backtest", "oos_eval", "cv_last", "val_df", "fitted_df"]:
            obj = res.attrs.get(k)
            if isinstance(obj, pd.DataFrame) and {"ds", "y_true", "y_pred"}.issubset(obj.columns):
                bt = obj[["ds", "y_true", "y_pred"]].copy()
                bt["ds"] = pd.to_datetime(bt["ds"], errors="coerce")
                bt = bt.dropna(subset=["ds"]).sort_values("ds")
                break

    # 2) Sem backtest, não há diagnóstico honesto de viés
    if bt is None or bt.empty:
        st.info(
            tr('Não encontrei um **backtest** com `y_true` e `y_pred` no resultado da previsão. Sem esses dados não é possível calcular vieses históricos. Se quiser, podemos adicionar cross-validation ao pipeline para habilitar essa aba.')
        )
    else:
        # Métricas de viés
        bt["erro"] = bt["y_pred"] - bt["y_true"]
        bias_abs = float(bt["erro"].mean()) if bt["erro"].notna().any() else np.nan

        # viés relativo (%): média de (erro/real) ignorando reais = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_vec = np.where(bt["y_true"] != 0, bt["erro"] / bt["y_true"], np.nan)
        bias_pct = float(np.nanmean(pct_vec) * 100.0)

        # MAE do backtest para escalar a conclusão
        mae_bt = float(np.nanmean(np.abs(bt["erro"]))) if bt["erro"].notna().any() else np.nan

        # KPIs lado a lado
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            _kpi(tr('Viés (nível)'), _safe_num(bias_abs), tr('Média de (previsto − real)'))
        with c2:
            _kpi(tr('Viés (%)'), _safe_num(bias_pct), tr('Média de (previsto − real)/real × 100'))
        with c3:
            _kpi("MAE (backtest)", _safe_num(mae_bt), tr('Média do |erro| no período de teste'))


        st.caption(
            tr('Interpretação: valores **positivos** indicam **superestimação**; negativos, **subestimação**. Quanto mais próximo de 0, menor o viés.')
        )

        # 3) Gráfico do erro (azul escuro) + linha de referência zero
        import altair as alt
        linha_zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9ca3af").encode(y="y:Q")
        ch = (
            alt.Chart(bt[["ds", "erro"]])
            .mark_line(color="#1e3a8a")
            .encode(
                x=alt.X("ds:T", title=tr('Período')),
                y=alt.Y("erro:Q", title=tr('Erro (previsto − real)')),
                tooltip=[
                    alt.Tooltip("ds:T", title=tr('Período')),
                    alt.Tooltip("erro:Q", title=tr('Erro'), format=",.2f"),
                ],
            )
            .properties(height=300, width="container")
            .interactive()
        )
        st.altair_chart(linha_zero + ch, use_container_width=True)

        # 4) Conclusão automática (baseada no viés vs. MAE)
        conclusao = ""
        if np.isfinite(bias_abs) and np.isfinite(mae_bt) and mae_bt > 0:
            # limiar: 5% do MAE → sem viés material; senão aponta direção
            if abs(bias_abs) < 0.05 * mae_bt:
                conclusao = tr('✅ **Sem viés sistemático relevante.** Os erros oscilam ao redor de zero.')
            elif bias_abs > 0:
                conclusao = tr('⚠️ **Viés positivo (superestimação).** Em média o modelo prevê acima do realizado.')
            else:
                conclusao = tr('⚠️ **Viés negativo (subestimação).** Em média o modelo prevê abaixo do realizado.')
        else:
            conclusao = tr('ℹ️ **Não foi possível calcular uma conclusão automática** (dados insuficientes).')

        st.markdown(tr('**Conclusão automática:** {0}', conclusao))

        # 5) Explicação fixa e objetiva
        st.markdown(
            tr('\n**Como ler o gráfico acima:**  \n- A linha mostra o **erro** em cada período (previsto − real).  \n- **Acima de 0** → o modelo **superestimou**; **abaixo de 0** → **subestimou**.  \n- Quando os pontos ficam próximos de 0 e alternam entre positivo/negativo, **não há viés sistemático**.  \n- Deslocamentos persistentes para cima/baixo sugerem **viés** e pedem recalibração do modelo (parâmetros, sazonalidade ou tendência).\n            ')
        )


# ======================================================
# TAB 3 — MPS & Custos (EPQ com Q do usuário; L4L por mês; sem custo de produção)
# ======================================================
with tabs[2]:
    st.subheader(tr('MPS — Custos e Resumo (somente leitura dos inputs)'))

    # ---------- Guardas ----------
    mps_inputs = st.session_state.get("mps_inputs", {})
    if not isinstance(mps_inputs, dict) or not mps_inputs:
        st.info(tr('Não encontrei os **inputs do MPS**. Preencha e salve na página **05_Inputs_MPS**.'))
        st.page_link("pages/05_Inputs_MPS.py", label=tr('⚙️ Ir para 05_Inputs_MPS'))
        st.stop()

    def _get_df_from_state(keys):
        for k in keys:
            obj = st.session_state.get(k, None)
            if isinstance(obj, pd.DataFrame) and not obj.empty:
                return obj
        return None

    # tabela do MPS (precisamos dela para L4L e para rupturas)
    mps_tbl_display = _get_df_from_state(["mps_tbl_display", "mps_table"])
    if mps_tbl_display is None:
        st.info(tr('Não há tabela do MPS na memória. Gere o MPS na página **06_MPS** e volte.'))
        st.page_link("pages/06_MPS.py", label=tr('📅 Ir para 06_MPS'))
        st.stop()

    # ---------- Helpers ----------
    def _safe(v, nd=0):
        try:
            if np.isnan(v): return "—"
        except Exception:
            pass
        try:
            if english(): return f"{float(v):,.{nd}f}"
            return f"{float(v):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(v)

    def _as_float(x, default=0.0):
        try:
            if x is None: return float(default)
            if isinstance(x, str) and x.strip() == "": return float(default)
            return float(x)
        except Exception:
            return float(default)

    def _find_row(df: pd.DataFrame, candidates: list[str]):
        idx_norm = df.index.astype(str).str.strip().str.lower()
        for cand in candidates:
            m = (idx_norm == cand.strip().lower())
            if m.any():
                pos = np.where(m)[0][0]
                label = df.index[pos]
                return df.loc[label]
        return None

    # ---------- Leitura segura (05_Inputs_MPS) ----------
    time_base   = mps_inputs.get("time_base", "por mês")
    A           = _as_float(mps_inputs.get("A", mps_inputs.get("order_cost", 0.0)), 0.0)
    v           = _as_float(mps_inputs.get("v", mps_inputs.get("unit_cost", 0.0)), 0.0)  # não entra no custo, só exibição
    lot_policy  = mps_inputs.get("lot_policy_default", "FX")  # "FX" ou "L4L"
    Q_user      = int(_as_float(mps_inputs.get("lot_size_default", 1), 1))  # tamanho do lote quando FX

    # H (mensal)
    h_mode = mps_inputs.get("h_mode", "Informar H diretamente")
    if h_mode == "Informar H diretamente":
        H_in  = _as_float(mps_inputs.get("H", 0.0), 0.0)
        H_m   = H_in if time_base == "por mês" else H_in / 12.0
        r_show = _as_float(mps_inputs.get("r", None), 0.0)
    else:
        r_val  = _as_float(mps_inputs.get("r", 0.0), 0.0)
        H_calc = r_val * v
        H_m    = H_calc if time_base == "por mês" else H_calc / 12.0
        r_show = r_val

    # D e p sempre mensais (se vieram anuais, converte)
    D_m = _as_float(mps_inputs.get("D_month", mps_inputs.get("D", 0.0)), 0.0)
    p_m = _as_float(mps_inputs.get("p_month", mps_inputs.get("p", 0.0)), 0.0)
    if time_base == "por ano":
        D_m /= 12.0
        p_m /= 12.0

    # fallback: média da previsão salva
    if D_m <= 0 and "forecast_df" in st.session_state:
        _fc = st.session_state["forecast_df"][["ds", "y"]].copy()
        D_m = float(np.nanmean(_fc["y"].values)) if len(_fc) else 0.0

    HORIZ_MESES = max(1, len(mps_tbl_display.columns))

    # Ruptura (se existir)
    row_ruptura = _find_row(mps_tbl_display, ["Ruptura", "falta", "backlog", "não atendido"])
    total_ruptura = float(np.nansum(np.clip(row_ruptura.values.astype(float), 0, None))) if row_ruptura is not None else 0.0
    pi_shortage = _as_float(mps_inputs.get("pi_shortage", 0.0), 0.0)

    # Linha de recebimentos (para L4L e contagem de setups)
    row_qtd_mps = _find_row(mps_tbl_display, ["qtde. mps", "qtde mps", "quantidade mps", "mps qty"])
    qtd_mps_vals = row_qtd_mps.values.astype(float) if row_qtd_mps is not None else np.zeros(HORIZ_MESES)

    # ---------- Custos (Q do usuário) ----------
    # Condição estrutural do EPQ
    epq_estrutura_ok = (p_m > D_m) and (H_m > 0) and (A >= 0)

    if not epq_estrutura_ok:
        C_setup_total = 0.0
        C_hold_total  = 0.0
    else:
        if lot_policy == "FX":
            # EPQ com Q = lote informado
            Q = max(1, int(Q_user))
            C_setup_mes = (A * D_m / Q)
            C_hold_mes  = H_m * (Q / 2.0) * (1.0 - (D_m / p_m))
            C_setup_total = C_setup_mes * HORIZ_MESES
            C_hold_total  = C_hold_mes  * HORIZ_MESES
        else:
            # L4L: por mês com qi do MPS
            setups = int(np.nansum((qtd_mps_vals > 0).astype(int)))
            C_setup_total = A * setups
            fator = (1.0 - (D_m / p_m))
            qi_term = np.clip(qtd_mps_vals, 0, None) / 2.0
            C_hold_total = float(np.nansum(H_m * qi_term * fator))

    cost_ruptura = total_ruptura * pi_shortage
    cost_total   = C_setup_total + C_hold_total + cost_ruptura

        # ------ Guarda baseline de custos para Tab Recomendações ------
    st.session_state["mps_costs_baseline"] = {
        "C_setup_total": float(C_setup_total),
        "C_hold_total": float(C_hold_total),
        "cost_ruptura": float(cost_ruptura),
        "cost_total": float(cost_total),
        "lot_policy": lot_policy,
        "Q_user": int(Q_user) if "Q_user" in locals() else None,
        "H_m": float(H_m),
        "D_m": float(D_m),
        "p_m": float(p_m),
        "time_base": time_base,
    }


    # ---------- Expander com variáveis ----------
    with st.expander(tr('ℹ️ Variáveis (05_Inputs_MPS) e parâmetros usados')):
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("A (setup)", _safe(A, 2))
        a2.metric(tr('v (valor unit.)'), _safe(v, 2))
        a3.metric(tr('H mensal (R$/un·mês)'), _safe(H_m, 2))
        a4.metric(tr('π (ruptura)'), _safe(pi_shortage, 2))
        b1, b2, b3, b4 = st.columns(4)
        b1.metric(tr('D (unid/mês)'), _safe(D_m, 2))
        b2.metric(tr('p (unid/mês)'), _safe(p_m, 2))
        b3.metric(tr('r (taxa man.)'), _safe(r_show, 4))
        b4.metric(tr('Política'), tr('Lote Fixo') if lot_policy == "FX" else tr('Lote-a-Lote'))
        if lot_policy == "FX":
            st.caption(tr('Lote informado (Q): **{0}**', _safe(Q_user, 0)))
        else:
            st.caption(tr('Setups no horizonte (meses com recebimento): **{0}**', int(np.nansum((qtd_mps_vals > 0).astype(int)))))

        # Fórmulas renderizadas
        st.latex(tr('C_{\\text{setup,mês}}=\\frac{A\\,D_m}{Q}\\quad\\text{(FX)}'))
        st.latex(tr('C_{\\text{hold,mês}}=H_m\\cdot\\frac{Q}{2}\\left(1-\\frac{D_m}{p_m}\\right)\\quad\\text{(FX)}'))
        st.latex(r"C_{\text{setup,total}}=\#\text{setups}\cdot A\quad\text{(L4L)}")
        st.latex(r"C_{\text{hold,total}}=\sum_i H_m\cdot\frac{q_i}{2}\left(1-\frac{D_m}{p_m}\right)\quad\text{(L4L)}")
        st.caption(tr('Multiplicamos pelos **meses do horizonte** quando FX. Em L4L, somamos mês a mês.'))

    # ---------- Layout visual (cards + tooltips) ----------
    st.markdown("""
    <style>
    .kpi-card {background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin-bottom:12px}
    .kpi-top {display:flex;align-items:center;gap:6px;color:#374151;font-size:0.95rem}
    .kpi-help {color:#6b7280;cursor:help;font-size:0.95rem}
    .kpi-value {font-size:1.85rem;font-weight:700;line-height:1.1;margin-top:6px}
    .kpi-sub {color:#6b7280;font-size:0.85rem;margin-top:4px}
    </style>
    """, unsafe_allow_html=True)

    def _fmt_money(x, nd=2):
        try:
            if english(): return f"{float(x):,.{nd}f}"
            return f"{float(x):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(x)

    def _help_span(txt: str) -> str:
        safe = (txt or "").replace('"', "&quot;")
        return f'<span class="kpi-help" title="{safe}">ⓘ</span>'

    months_lbl = f"{HORIZ_MESES:d}"
    tip_setup = (
        tr('FX: C_setup,mês = A·D_m/Q; total = C_setup,mês × meses. L4L: C_setup,total = nº de setups × A.')
    )
    tip_hold = (
        tr('FX: C_hold,mês = H_m·(Q/2)·(1 − D_m/p_m); total = C_hold,mês × meses. L4L: C_hold,total = Σ H_m·(q_i/2)·(1 − D_m/p_m).')
    )
    tip_rupt = tr('C_rupt = Σ(Ruptura) × π (se sua tabela tiver a linha de Ruptura).')
    tip_total = tr('Custo total relevante = Setup + Manter + Ruptura.')

    L, R = st.columns(2)

    with L:
        st.markdown(
            tr('<div class="kpi-card">\n                <div class="kpi-top">Custo de setup (R$){0}</div>\n                <div class="kpi-value">{1}</div>\n                <div class="kpi-sub">{2}</div>\n            </div>', _help_span(tip_setup), _fmt_money(C_setup_total, 2), tr('FX (mensal × ') + months_lbl + ')' if lot_policy == 'FX' else 'L4L (setups × A)'),
            unsafe_allow_html=True,
        )
        st.markdown(
            tr('<div class="kpi-card">\n                <div class="kpi-top">Custo de manter (R$){0}</div>\n                <div class="kpi-value">{1}</div>\n                <div class="kpi-sub">{2}</div>\n            </div>', _help_span(tip_hold), _fmt_money(C_hold_total, 2), tr('FX (mensal × ') + months_lbl + ')' if lot_policy == 'FX' else tr('L4L (soma mensal)')),
            unsafe_allow_html=True,
        )

    with R:
        st.markdown(
            tr('<div class="kpi-card">\n                <div class="kpi-top">Custo de ruptura (R$){0}</div>\n                <div class="kpi-value">{1}</div>\n                <div class="kpi-sub">Σ(Ruptura) × π</div>\n            </div>', _help_span(tip_rupt), _fmt_money(cost_ruptura, 2)),
            unsafe_allow_html=True,
        )
        st.markdown(
            tr('<div class="kpi-card">\n                <div class="kpi-top">Custo total relevante{0}</div>\n                <div class="kpi-value">{1}</div>\n                <div class="kpi-sub">Setup + Manter + Ruptura</div>\n            </div>', _help_span(tip_total), _fmt_money(cost_total, 2)),
            unsafe_allow_html=True,
        )

    if not epq_estrutura_ok:
        st.warning(tr('Parâmetros inválidos para EPQ: é preciso **p > D** e **H > 0**. Ajuste no **05_Inputs_MPS**.'))

    # =============================
    # ATP ACUMULADO — capacidade de atender demanda extra FIXA por mês
    # =============================
    st.divider()
    st.markdown(tr('### 🧮 ATP acumulado (até cada mês)'))

    # 1) Pegar a linha ATP(cum) da tabela do MPS
    row_atp_cum = _find_row(
        mps_tbl_display,
        ["ATP(cum)", "atp(cum)", "ATP acumulado", "atp acumulado"]
    )

    if row_atp_cum is None:
        st.info(
            tr('Não encontrei a linha **ATP(cum)** na tabela do MPS. Confira a planilha gerada na página **06_MPS**.')
        )

    else:
        # ORDEM dos meses conforme MPS
        col_labels = list(mps_tbl_display.columns)
        s_atp = pd.Series(row_atp_cum, index=col_labels)

        # Série numérica limpa
        atp_vals = (
            pd.to_numeric(s_atp, errors="coerce")
            .fillna(0)
            .astype(float)
            .to_numpy()
        )
        atp_cum_series = pd.Series(atp_vals, index=col_labels)

        # Rótulos no formato Mes/Ano
        try:
            idx_dt = pd.to_datetime(atp_cum_series.index)
            PT_MON = ["Jan","Fev","Mar","Abr","Mai","Jun",
                    "Jul","Ago","Set","Out","Nov","Dez"]
            pretty_idx = [f"{tr(PT_MON[d.month-1])}/{d.year%100:02d}" for d in idx_dt]
        except Exception:
            pretty_idx = [month_label(x) for x in atp_cum_series.index]

        # DataFrame principal
        atp_df = pd.DataFrame({
            tr('Mês'): pretty_idx,
            tr('ATP acumulado (unid)'): atp_cum_series.values.astype(float),
        })

        # Entrada do usuário
        extra = st.number_input(
            tr('Demanda extra hipotética (un/mês)'),
            min_value=0,
            step=1,
            value=0,
            help=tr('Quantidade adicional fixa de demanda em cada mês.')
        )

        # Lógica correta: compara ATP cumulativo com a demanda fixa
        atp_df[tr('Demanda extra (fixa)')] = extra
        atp_df[tr('Atende?')] = atp_df[tr('ATP acumulado (unid)')] >= extra

        # ============================
        # 🔥 LAYOUT LADO A LADO🔥
        # ============================
        left, right = st.columns([2, 1])

        # ----- GRÁFICO -----
        with left:
            import altair as alt
            chart_data = atp_df.copy()

            color_scale = alt.Scale(
                domain=[True, False],
                range=["#16a34a", "#dc2626"]  # verde / vermelho
            )

            bars = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X(tr('Mês:N'), title=tr('Mês'), sort=None),
                y=alt.Y(tr('ATP acumulado (unid):Q'), title=tr('ATP acumulado (unid)')),
                color=alt.Color(tr('Atende?:N'), title=tr('Atende a extra?'), scale=color_scale),
                tooltip=[
                    tr('Mês'),
                    alt.Tooltip(tr('ATP acumulado (unid):Q'), format=",.0f"),
                    alt.Tooltip(tr('Demanda extra (fixa):Q'), format=",.0f"),
                    tr('Atende?')
                ],
            )

            labels = alt.Chart(chart_data).mark_text(
                dy=-6, fontSize=11
            ).encode(
                x=tr('Mês:N'),
                y=tr('ATP acumulado (unid):Q'),
                text=alt.Text(tr('ATP acumulado (unid):Q'), format=",.0f"),
                color=alt.value("#111827"),
            )

            line = alt.Chart(chart_data).mark_rule(
                color="#0f172a", strokeDash=[6, 4]
            ).encode(
                y=tr('Demanda extra (fixa):Q')
            )

            chart = (
                bars + labels + line
            ).properties(
                height=360,
                title=(
                    tr('ATP acumulado vs. demanda extra (fixa) (extra = {0} un/mês)', extra)
                ),
            ).configure_axis(
                labelFontSize=12,
                titleFontSize=12
            ).configure_legend(
                labelFontSize=12,
                titleFontSize=12
            )

            st.altair_chart(chart, use_container_width=True)


        # ----- TABELA -----
        with right:
            atp_df[tr('✔ Atende?')] = atp_df[tr('Atende?')].map({True: "✅", False: "❌"})
            display_df = atp_df[
                [tr('Mês'), tr('ATP acumulado (unid)'), tr('Demanda extra (fixa)'), tr('✔ Atende?')]
            ]

            st.dataframe(display_df, use_container_width=True, height=360)

        # ----- RESUMO -----
        if extra > 0:
            meses_ok = display_df.loc[atp_df[tr('Atende?')], tr('Mês')].tolist()
            if meses_ok:
                st.success(
                    tr('Com **{0} un/mês** de demanda extra, os meses atendidos são: {1}.', extra, ', '.join(meses_ok))
                )
            else:
                st.warning(
                    tr('Com **{0} un/mês** de demanda extra, nenhum mês teria ATP suficiente.', extra)
                )
        else:
            st.caption(tr('Ajuste a demanda extra acima para testar cenários.'))



# ======================================================
# TAB 4 — Recomendações + “What-if” de Q (FX)
# ======================================================
with tabs[3]:
    import numpy as np
    import pandas as pd
    import altair as alt
    import streamlit as st

    st.subheader(tr('Recomendações'))

    # ----------------- Helpers -----------------
    def _safe(v, nd=0):
        try:
            if np.isnan(v): return "—"
        except Exception:
            pass
        try:
            if english(): return f"{float(v):,.{nd}f}"
            return f"{float(v):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(v)

    def _get_df_from_state(keys: list[str]) -> pd.DataFrame | None:
        for k in keys:
            obj = st.session_state.get(k, None)
            if isinstance(obj, pd.DataFrame) and not obj.empty:
                return obj
        return None

    def _find_row(df: pd.DataFrame, candidates: list[str]):
        if df is None: 
            return None
        idx_norm = df.index.astype(str).str.strip().str.lower()
        for cand in candidates:
            m = (idx_norm == cand.strip().lower())
            if m.any():
                pos = np.where(m)[0][0]
                label = df.index[pos]
                return df.loc[label]
        return None

    # ----------------- Coletas seguras -----------------
    recs: list[str] = []

    # Campeão de previsão (sMAPE), se existir
    _champ = locals().get("champion") or st.session_state.get("forecast_champion") or st.session_state.get("champion")
    smape = None
    if isinstance(_champ, dict):
        smape = _champ.get("sMAPE") or _champ.get("smape") or _champ.get("SMAPE")

    # MPS exibido (para comparação e métricas)
    mps_table = _get_df_from_state(["mps_tbl_display", "mps_table"])

    # Inputs do MPS
    mps_inputs = st.session_state.get("mps_inputs", {}) if isinstance(st.session_state.get("mps_inputs", {}), dict) else {}
    lot_policy = (mps_inputs.get("lot_policy_default") or "").upper()  # "FX" ou "L4L"
    Q_fx       = int(mps_inputs.get("lot_size_default", 0) or 0)
    initial_inv = int(mps_inputs.get("initial_inventory_default", 0) or 0)
    lead_time   = int(mps_inputs.get("lead_time_default", 0) or 0)

    # Parâmetros de custos (normalizados para MÊS)
    time_base  = mps_inputs.get("time_base", "por mês")
    A          = float(mps_inputs.get("A", mps_inputs.get("order_cost", 0.0)) or 0.0)
    D_in       = float(mps_inputs.get("D", 0.0) or 0.0)
    p_in       = float(mps_inputs.get("p", 0.0) or 0.0)
    if time_base == "por ano":
        D_m = D_in / 12.0
        p_m = p_in / 12.0
    else:
        D_m = D_in
        p_m = p_in

    v = float(mps_inputs.get("v", mps_inputs.get("unit_cost", 0.0)) or 0.0)
    h_mode = mps_inputs.get("h_mode", "Informar H diretamente")
    if h_mode == "Informar H diretamente":
        H_m = float(mps_inputs.get("H", 0.0) or 0.0)
    else:
        r = float(mps_inputs.get("r", 0.0) or 0.0)
        H_m = r * v
        if time_base == "por ano":
            H_m = H_m / 12.0

    pi_shortage = float(mps_inputs.get("pi_shortage", mps_inputs.get("shortage_cost", 0.0)) or 0.0)

    # Horizonte (nº de meses) — tenta MPS; se não houver, usa previsão
    if isinstance(mps_table, pd.DataFrame):
        HORIZ_MESES = max(1, len(mps_table.columns))
    elif "forecast_df" in st.session_state:
        HORIZ_MESES = max(1, len(st.session_state["forecast_df"]))
    else:
        HORIZ_MESES = 6  # fallback visual

    # ----------------- Recomendações automáticas -----------------
    if isinstance(smape, (int, float)):
        if smape > 30:
            recs.append(tr('sMAPE **alto** → considere **aumentar histórico**, tratar **outliers** e testar **modelos alternativos**.'))
        elif smape > 15:
            recs.append(tr('sMAPE **moderado** → ajuste **hiperparâmetros** e revise **sazonalidades/regressoras**.'))
        else:
            recs.append(tr('sMAPE **baixo** → mantenha a configuração e **monitore** periodicamente.'))

    if isinstance(mps_table, pd.DataFrame):
        row_est = _find_row(mps_table, ["Estoque Proj.", "Estoque Projetado", "estoque"])
        row_rupt = _find_row(mps_table, ["Ruptura", "falta", "backlog", "não atendido"])
        row_qtd = _find_row(mps_table, ["Qtde. MPS", "Qtde MPS", "quantidade mps", "mps qty"])

        if row_est is not None and len(row_est.values):
            estoque_final = float(row_est.values.astype(float)[-1])
            if estoque_final < 0:
                recs.append(tr('**Estoque final negativo** → **antecipar** produção/compras ou **elevar SS**.'))
            elif estoque_final == 0:
                recs.append(tr('**Estoque final zerado** → atenção a **rupturas** com qualquer desvio.'))
            else:
                recs.append(tr('**Estoque final**: {0} un. — confira se há **excesso** vs. meta de giro.', _safe(estoque_final, 0)))

        if row_rupt is not None:
            rupt = np.clip(row_rupt.values.astype(float), 0, None)
            if np.nansum(rupt) > 0:
                meses_rupt = int(np.nansum(rupt > 0))
                recs.append(tr('**Rupturas** em {0} mês(es) → elevar **SS**, **antecipar MPS** ou **reduzir lead time**.', meses_rupt))
            else:
                recs.append(tr('Sem **rupturas** projetadas no horizonte.'))

        if row_qtd is not None:
            qtd_mes = row_qtd.values.astype(float)
            n_meses_com_pedido = int(np.nansum(qtd_mes > 0))
            if lot_policy == "FX" and len(qtd_mes) > 0:
                if n_meses_com_pedido > 0.8 * len(qtd_mes):
                    recs.append(tr('Com **FX**, há pedidos na maioria dos meses → teste **Q** maior para reduzir setups.'))
            if lot_policy == "L4L" and len(qtd_mes) > 0:
                if n_meses_com_pedido < 0.5 * len(qtd_mes):
                    recs.append(tr('Com **L4L**, poucos meses com pedido → verifique **congelamento**/capacidade (pode haver folga).'))
    else:
        recs.append(tr('Gere o **MPS (página 06)** para habilitar recomendações operacionais.'))

    # Condições básicas do EPQ (para usar fórmulas de manter)
    if p_m <= D_m:
        recs.append(tr('Para EPQ é necessário **p > D** (capacidade maior que a demanda).'))
    if H_m <= 0:
        recs.append(tr('**H (custo de manter)** deve ser positivo. Revise **H** (ou **r·v**) nos inputs.'))

    if recs:
        st.markdown("\n".join(f"- {r}" for r in recs))
    else:
        st.markdown(tr('- Sem recomendações automáticas no momento.'))

    st.divider()

    # ======================================================
    # BLOCO: Simulador interativo de Q (política FX) — HORIZONTE TRAVADO
    # ======================================================
    st.subheader(tr('Sensibilidade — Impacto da Mudança de Parâmetros no Custo Relevante Total'))

    # Horizonte travado no mesmo da previsão (página 04)
    HORIZ_BASE = int(st.session_state.get("forecast_h", len(st.session_state.get("forecast_df", [])) or 1))
    st.caption(tr('Horizonte de comparação travado em **{0} meses** (igual ao da previsão).', HORIZ_BASE))

    # Caixa de parâmetros do simulador
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            D_lbl = st.number_input(
                tr('Demanda média (unid/mês)'),
                value=float(D_m), step=1.0, min_value=0.0, format="%.2f",
                help=tr('Por padrão usa a média da previsão atual (base mensal).')
            )
        with c2:
            p_lbl = st.number_input(
                tr('Capacidade p (unid/mês)'),
                value=float(p_m), step=1.0, min_value=0.0, format="%.2f",
                help=tr('Necessário **p > D** para EPQ/estoque durante produção.')
            )
        with c3:
            A_lbl = st.number_input(
                tr('Custo de setup A (R$)'),
                value=float(A), step=10.0, min_value=0.0, format="%.2f",
                help=tr('Custo fixo por preparação do lote (setup).')
            )
        with c4:
            H_lbl = st.number_input(
                tr('Custo de manter H (R$/un·mês)'),
                value=float(H_m), step=1.0, min_value=0.0, format="%.2f",
                help=tr('Custo mensal por unidade mantida em estoque.')
            )

        # Linha 2: Q e SS extra (sem horizonte — está travado)
        c5, c6 = st.columns([1,1])
        with c5:
            # valor inicial de Q vindo da política atual, se existir
            q_init = Q_fx if 'Q_fx' in locals() and Q_fx > 0 else max(1, int(D_lbl) if D_lbl > 0 else 1)
            Q_user = st.number_input(
                tr('Q — Tamanho do lote (unid)'),
                value=float(q_init), min_value=1.0, step=1.0, format="%.0f",
                help=tr('Valor do lote para a política **FX** no cenário simulado.')
            )
        with c6:
            ss_extra = st.number_input(
                tr('SS adicional (unid/mês) (opcional)'),
                value=0.0, min_value=0.0, step=1.0, format="%.0f",
                help=tr('Só para ver sensibilidade: adiciona estoque médio constante para custo de manter.')
            )

    # Funções de custo (mensais)
    def cost_setup_month(A, D, Q):
        return A * (D / max(Q, 1e-9))

    def cost_holding_month(H, Q, D, p):
        # Fórmula EPQ: I_médio = (Q/2) * (1 - D/p). Se p<=D, não há estoque durante produção.
        if p <= D:
            return 0.0
        fator = (1.0 - D / p)
        I_med = 0.5 * Q * fator
        return H * I_med

    # Calcula custos do cenário Q_user
    setup_m = cost_setup_month(A_lbl, D_lbl, Q_user)
    hold_m  = cost_holding_month(H_lbl, Q_user, D_lbl, p_lbl) + H_lbl * ss_extra
    total_m = setup_m + hold_m

    # Custos no horizonte TRAVADO
    meses_sim = HORIZ_BASE  # <- travado
    cost_setup_total = setup_m * meses_sim
    cost_hold_total  = hold_m  * meses_sim
    cost_total_sim   = cost_setup_total + cost_hold_total

    # Espaço para respirar da UI
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Cards de resultado do cenário
    L, R = st.columns(2)
    with L:
        st.markdown(tr('#### Custos no horizonte (cenário FX com Q informado)'))
        st.metric(tr('Custo de setup (R$)'), _safe(cost_setup_total, 2), help=tr('A × (D/Q) × meses'))
        st.metric(tr('Custo de manter (R$)'), _safe(cost_hold_total, 2), help=tr('H × (Q/2) × (1 − D/p) × meses  +  H × SS_extra × meses'))
        st.metric("Total (R$)", _safe(cost_total_sim, 2))
    with R:
        st.markdown(tr('**Parâmetros aplicados:**'))
        st.write(tr('- Q = {0} un', _safe(Q_user, 0)))
        st.write(tr('- A = R\\$ {0}; H = R\\$ {1} /un·mês', _safe(A_lbl, 2), _safe(H_lbl, 4)))
        st.write(tr('- D = {0} un/mês; p = {1} un/mês', _safe(D_lbl, 2), _safe(p_lbl, 2)))
        st.write(tr('- Horizonte = {0} meses (travado)', meses_sim))

        if p_lbl <= D_lbl:
            st.warning(tr('Com **p ≤ D**, a parcela de **manter** pelo EPQ fica 0 (sem estoque em produção).'))
        if ss_extra > 0:
            st.info(tr('Considerando **SS_extra = {0}** un/mês no custo de manter.', _safe(ss_extra, 0)))

    # ------------------------------------------------------
    # Sensibilidade: C(Q) em uma faixa (apenas quando p > D)
    # ------------------------------------------------------
    st.markdown(tr('##### Curva de sensibilidade C(Q) (FX)'))

    q_min = max(1, int(0.25 * (D_lbl if D_lbl > 0 else 1)))
    q_max = max(q_min + 1, int(4 * (D_lbl if D_lbl > 0 else 10)))
    step  = max(1, (q_max - q_min) // 25)  # grade razoável
    q_grid = np.arange(q_min, q_max + 1, step)

    data = []
    for q in q_grid:
        c_s = float(cost_setup_month(A_lbl, D_lbl, q))
        c_h = float(cost_holding_month(H_lbl, q, D_lbl, p_lbl) + H_lbl * ss_extra)
        data.append({"Q": int(q), tr('Setup (mês)'): c_s, tr('Manter (mês)'): c_h, tr('Total (mês)'): c_s + c_h})

    dfC = pd.DataFrame(data)

    # Converte para formato longo e força tipos numéricos
    if not dfC.empty:
        df_long = (
            dfC.melt(id_vars="Q", var_name=tr('Componente'), value_name=tr('Custo'))
            .dropna(subset=["Q", tr('Custo')])
            .assign(Q=lambda d: pd.to_numeric(d["Q"], errors="coerce").astype("int64"),
                    Custo=lambda d: pd.to_numeric(d[tr('Custo')], errors="coerce").astype(float))
        )
    else:
        df_long = pd.DataFrame(columns=["Q", tr('Componente'), tr('Custo')])

    if df_long.empty or not np.isfinite(df_long[tr('Custo')]).any():
        st.info(tr('Não há dados suficientes para desenhar a curva C(Q). Ajuste D, p, A, H ou Q.'))
    else:
        import altair as alt
        # (opcional) remover limite de linhas do Altair
        alt.data_transformers.disable_max_rows()

        chart = (
            alt.Chart(df_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("Q:Q", title=tr('Tamanho do lote Q (unidades)')),
                y=alt.Y(tr('Custo:Q'), title=tr('Custo mensal (R$)')),
                color=alt.Color(tr('Componente:N')),
                tooltip=[alt.Tooltip("Q:Q"),
                            alt.Tooltip(tr('Componente:N')),
                            alt.Tooltip(tr('Custo:Q'), format=".2f")]
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)
    # ------------------------------------------------------
    # Comparação com MPS atual (baseline vindo da Tab 3)
    # ------------------------------------------------------
    st.markdown(tr('##### Comparação rápida com o MPS atual (Gerado na página 06)'))

    custos_base = st.session_state.get("mps_costs_baseline", None)

    def _fmt_money(x, nd=2):
        try:
            if english(): return f"{float(x):,.{nd}f}"
            return f"{float(x):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(x)

    if not custos_base:
        st.info(
            tr('Não encontrei os **custos do MPS atual**. Vá na aba **MPS & Custos** e salve os dados (Tab 3) para habilitar esta comparação.')
        )
    else:
        base_setup_total = float(custos_base.get("C_setup_total", 0.0))
        base_hold_total  = float(custos_base.get("C_hold_total", 0.0))
        base_rupt_total  = float(custos_base.get("cost_ruptura", 0.0))
        base_total       = float(custos_base.get("cost_total", 0.0))

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric(tr('MPS Simulado — Total (R$)'), _safe(cost_total_sim, 2))
        with colB:
            st.metric(tr('MPS atual — Total (R$)'), _fmt_money(base_total, 2))
        with colC:
            delta = cost_total_sim - base_total
            st.metric(
                tr('Diferença (sim − atual)'),
                _fmt_money(delta, 2),
                help=tr('Positivo = cenário simulado mais caro; negativo = mais barato que o MPS atual.')
            )


    # Download dos resultados do grid
    st.download_button(
        tr('⬇️ Baixar tabela C(Q) (CSV)'),
        data=dfC.to_csv(index=False).encode("utf-8"),
        file_name="sensibilidade_CQ.csv",
        mime="text/csv",
    )

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.page_link("pages/04_Previsao.py", label=tr('🔁 Ajustar Previsão'))
    with c2:
        st.page_link("pages/05_Inputs_MPS.py", label=tr('⚙️ Inputs do MPS'))
    with c3:
        st.page_link("pages/06_MPS.py", label=tr('🗓️ Abrir MPS'))
