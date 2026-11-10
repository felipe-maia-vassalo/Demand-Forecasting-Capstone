# pages/05_Inputs_MPS.py
from __future__ import annotations
from core.i18n import ENGLISH, english, tr, month_label
import numpy as np
import pandas as pd
import streamlit as st

option_label = (lambda value: ENGLISH.get(value, value)) if english() else str

st.title(tr('⚙️ Inputs do MPS'))

# =========================
# GUARDAS DE ETAPA
# =========================
if "ts_df_norm" not in st.session_state:
    st.warning(tr('Preciso da série do Passo 1 (Upload) antes de continuar.'))
    st.page_link("pages/01_Upload.py", label=tr('Ir para o Passo 1 — Upload'))
    st.stop()

if not st.session_state.get("forecast_committed", False):
    st.warning(tr('Preciso que você **salve a previsão** no Passo 2 (Previsão) antes de configurar os inputs.'))
    st.page_link("pages/04_Previsao.py", label=tr('Ir para o Passo 2 — Previsão'))
    st.stop()

if "forecast_df" not in st.session_state or "forecast_h" not in st.session_state:
    st.error(tr('Previsão não encontrada no estado. Volte ao Passo 2, salve a previsão e retorne.'))
    st.page_link("pages/04_Previsao.py", label=tr('Ir para o Passo 2 — Previsão'))
    st.stop()

# =========================
# DADOS BASE
# =========================
fcst = st.session_state["forecast_df"][["ds", "y"]].copy()
horizon = int(st.session_state["forecast_h"])

# Garantimos que 'ds' está em Timestamp mensal (MS)
ds_ts = pd.to_datetime(fcst["ds"]).dt.to_period("M").dt.to_timestamp()

# rótulo bonito MÊS/ANO em PT-BR (ex.: Set/25)
PT_MON = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
def fmt_mmyy(ts: pd.Timestamp) -> str:
    m = ts.month
    yy = ts.year % 100
    return f"{PT_MON[m-1]}/{yy:02d}"

labels_raw: list[pd.Timestamp] = ds_ts.tolist()          # valores "reais"
labels_str: list[str] = [fmt_mmyy(ts) for ts in ds_ts]    # rótulos para UI
idx_by_label_str = {s: i for i, s in enumerate(labels_str)}

st.caption(tr('🔗 Horizonte atual da Previsão: **{0} meses**.', horizon))

# defaults
defaults = st.session_state.get("mps_inputs", {})
def get(k, dv):
    value = defaults.get(k)
    return dv if value is None else value

# =========================
# 1) ITEM E POLÍTICAS PADRÃO
# =========================
st.subheader(tr('1) Item e políticas padrão'))
c1, c2, c3, c4 = st.columns(4)
with c1:
    product_name = st.session_state.get("product_name", None)
    default_item = get("item_name", product_name if product_name else tr('Item (sem nome)'))
    item_name = st.text_input(tr('Item (nome)'), value=default_item)
with c2:
    lot_policy_default = st.selectbox(
        tr('Política padrão'), [tr('Lote Fixo (FX)'), tr('Lote-a-Lote (L4L)')],
        index=0 if get("lot_policy_default", "FX") == "FX" else 1
    )
    lot_policy_default = "FX" if lot_policy_default.startswith(tr('Lote Fixo')) else "L4L"
with c3:
    lot_size_default = st.number_input(
        tr('Tamanho do lote padrão (se FX)'), min_value=1, step=10, value=int(get("lot_size_default", 150))
    )
with c4:
    initial_inventory_default = st.number_input(
        tr('Estoque em mão inicial (padrão)'), min_value=0, step=5, value=int(get("initial_inventory_default", 55))
    )

# Lead time (meses)
lt1, lt2, lt3, lt4 = st.columns(4, gap="small")
with lt1:
    lead_time_default = st.number_input(
        tr('Lead time (meses)'), min_value=0, value=int(get("lead_time_default", 1)), step=1
    )
with lt2: st.write("")
with lt3: st.write("")
with lt4: st.write("")

# =========================
# 2) ESTOQUE DE SEGURANÇA — PARÂMETROS
# =========================
st.subheader(tr('2) Estoque de segurança — parâmetros'))
row = st.columns([1.2, 1, 1])
with row[0]:
    auto_ss = st.checkbox(tr('Ativar SS automático (variável por mês)'), value=bool(get("auto_ss", True)))
    ss_method = st.radio(
        tr('Método'), ["CV (%)", "σ absoluto"], format_func=option_label,
        index=0 if get("ss_method", "CV (%)") == "CV (%)" else 1,
        horizontal=True
    )
with row[1]:
    z_choice = st.selectbox(
        tr('Nível de serviço (z)'), ["90%", "95%", "97.5%", "99%"],
        index={"90%":0, "95%":1, "97.5%":2, "99%":3}[get("z_choice", "95%")]
    )
with row[2]:
    if ss_method == "CV (%)":
        cv_pct = st.number_input(tr('CV da demanda/erro (%)'), min_value=0.0, value=float(get("cv_pct", 15.0) or 0.0), step=1.0)
        sigma_abs = None
    else:
        sigma_abs = st.number_input(tr('σ absoluto (unid/mês)'), min_value=0.0, value=float(get("sigma_abs", 20.0)), step=1.0)
        cv_pct = None

# =========================
# 3) CONGELAMENTO DE HORIZONTE (intervalo)
# =========================
st.subheader(tr('3) Congelamento de horizonte (intervalo)'))
start_label_str = labels_str[0]

freeze_on_default = bool(defaults.get("freeze_on", False))
freeze_on = st.checkbox(tr('Ativar congelamento'), value=freeze_on_default)

_saved = defaults.get("frozen_range", (start_label_str, start_label_str))
saved_end_str = _saved[1] if isinstance(_saved, (list, tuple)) and len(_saved) == 2 else start_label_str
if saved_end_str not in labels_str:
    try:
        ts_try = pd.to_datetime(saved_end_str, errors="coerce")
        if pd.notna(ts_try):
            saved_end_str = fmt_mmyy(ts_try.to_period("M").to_timestamp())
    except Exception:
        saved_end_str = start_label_str
if saved_end_str not in labels_str:
    saved_end_str = start_label_str

if freeze_on:
    end_label_str = st.select_slider(
        tr('Selecione o **fim** do período a congelar (início fixo em {0})', month_label(start_label_str)),
        options=labels_str, format_func={v: month_label(v) for v in labels_str}.__getitem__,
        value=saved_end_str,
    )
    frozen_range = (start_label_str, end_label_str)
    st.caption(tr('Período congelado: **{0} → {1}**', month_label(frozen_range[0]), month_label(frozen_range[1])))
else:
    frozen_range = None
    st.caption(tr('Período congelado: **sem congelamento**'))

# =========================
# 4) PEDIDOS EM CARTEIRA
# =========================
st.subheader(tr('4) Pedidos firmes — Em carteira'))

if "mps_firm_orders" in st.session_state:
    current_orders = st.session_state["mps_firm_orders"].copy()
    cur_labels_str = [fmt_mmyy(pd.to_datetime(x).to_period("M").to_timestamp())
                      for x in current_orders["ds"].tolist()]
    if cur_labels_str != labels_str:
        current_orders = pd.DataFrame({"ds": labels_raw, "y": [0]*len(labels_raw)})
else:
    current_orders = pd.DataFrame({"ds": labels_raw, "y": [0]*len(labels_raw)})

orders_row_df = pd.DataFrame([current_orders["y"].tolist()], index=[tr('Em carteira')], columns=labels_str)
orders_row_df = st.data_editor(
    orders_row_df,
    use_container_width=True,
    num_rows="fixed",
    column_config={lab: st.column_config.NumberColumn(month_label(lab), min_value=0, step=1) for lab in labels_str},
    key="orders_row_inputs_mps",
)
orders_row_df = orders_row_df.fillna(0)
values = orders_row_df.loc[tr('Em carteira')].astype(int).reindex(labels_str).tolist()
orders_df = pd.DataFrame({"ds": labels_raw, "y": values})

# =========================
# 5)  Parâmetros de custos (nomenclatura da aula)
# =========================
st.subheader(tr('5) Parâmetros de custos'))

def _as_float(x, default=0.0) -> float:
    try:
        if x is None: return float(default)
        if isinstance(x, str) and x.strip() == "": return float(default)
        return float(x)
    except Exception:
        return float(default)

# Base de tempo
time_base = st.radio(
    tr('Base de tempo dos parâmetros (para D, H e r):'),
    options=["por mês", "por ano"], format_func=option_label,
    index=0 if get("time_base", "por mês") == "por mês" else 1,
    horizontal=True,
    help=tr('A conversão, se necessária, será feita na página de conclusão.')
)

# A (setup), D (taxa de demanda), p (taxa de produção)
cA, cD, cP = st.columns(3)
with cA:
    A = st.number_input(
        tr('A — Custo fixo por setup (R$)'),
        min_value=0.0, step=10.0, value=_as_float(get("A", 0.0), 0.0),
        help=tr('Custo fixo cada vez que prepara a produção (setup).')
    )
with cD:
    # D vem da média da previsão (já calculada na página de previsão); exibimos somente-leitura
    fcst_df = st.session_state["forecast_df"][["ds", "y"]].copy()
    D_media_mes = float(fcst_df["y"].mean()) if len(fcst_df) else 0.0
    D = D_media_mes if time_base == "por mês" else D_media_mes * 12.0
    st.number_input(
        tr('D — Taxa de demanda (unid/{0})', tr("mês") if time_base == "por mês" else tr("ano")),
        value=float(D), step=1.0, min_value=0.0, disabled=True,
        help=tr('Média da demanda prevista no horizonte atual.')
    )
with cP:
    p = st.number_input(
        tr('p — Taxa de produção (unid/{0})', tr("mês") if time_base == "por mês" else tr("ano")),
        min_value=0.0, step=1.0, value=_as_float(get("p", 0.0), 0.0),
        help=tr('Capacidade de produção (EPQ exige p > D).')
    )

# H — custo de manter (direto) ou via r·v
h_mode = st.radio(
    tr('Como deseja informar H?'),
    options=["Informar H diretamente", "Calcular H a partir de r e v"], format_func=option_label,
    index=0 if get("h_mode", "Informar H diretamente") == "Informar H diretamente" else 1,
    horizontal=True
)

# v é obrigatório SEMPRE
v_default = _as_float(get("v", None), _as_float(get("unit_cost", 0.0), 0.0))
colV, colH1, colH2 = st.columns(3)
with colV:
    v = st.number_input(
        tr('v — Valor unitário do item (R$/unid) **(obrigatório)**'),
        min_value=0.0, step=1.0, value=float(v_default),
        help=tr('Usado em C_prod = v·D e, se escolher, em H = r·v.')
    )

if h_mode == "Informar H diretamente":
    with colH1:
        H = st.number_input(
            tr('H — Custo de manter (R$ por unid/{0})', tr("mês") if time_base == "por mês" else tr("ano")),
            min_value=0.0, step=1.0, value=_as_float(get("H", 0.0), 0.0),
            help=tr('Se preferir informar H direto, ainda assim v é obrigatório para os demais cálculos.')
        )
    r = get("r", None)  # opcional (não usado)
else:
    with colH1:
        r = st.number_input(
            tr('r — Taxa de manutenção (R$/$ por {0})', tr("mês") if time_base == "por mês" else tr("ano")),
            min_value=0.0, step=0.01, value=_as_float(get("r", 0.20 if time_base=='por ano' else 0.02), 0.0),
            help=tr('Taxa multiplicada por v para obter H = r·v.')
        )
    H = None  # será calculado na conclusão

# (opcional) Custo de falta / ruptura por unidade
pi_shortage = st.number_input(
    tr('π — Custo de falta/ruptura (R$ por unidade não atendida)'),
    min_value=0.0, step=1.0, value=_as_float(get("pi_shortage", get("shortage_cost", 0.0)), 0.0),
    help=tr('Opcional. Se informado, será usado para penalizar rupturas no horizonte.')
)

# =========================
# SALVAR
# =========================
if st.button(tr('Salvar inputs do MPS'), type="primary"):
    # ponto de partida: preserva o que já havia
    new_cfg = dict(st.session_state.get("mps_inputs", {}))

    # Seção 1 — políticas e parâmetros do item (inclui Q = lot_size_default)
    sec1 = {
        "item_name": item_name,
        "lot_policy_default": "FX" if lot_policy_default == "FX" else "L4L",
        "lot_size_default": int(lot_size_default),    # <-- Q (lote do usuário)
        "Q_user": int(lot_size_default),              # <-- alias para a aba de custos
        "initial_inventory_default": int(initial_inventory_default),
        "lead_time_default": int(lead_time_default),
    }

    # Seção 2 — estoque de segurança
    sec2 = {
        "auto_ss": bool(auto_ss),
        "ss_method": ss_method,
        "z_choice": z_choice,
        "cv_pct": float(cv_pct) if 'cv_pct' in locals() and cv_pct is not None else None,
        "sigma_abs": float(sigma_abs) if 'sigma_abs' in locals() and sigma_abs is not None else None,
    }

    # Seção 3 — congelamento
    sec3 = {
        "freeze_on": bool(freeze_on),
        "frozen_range": tuple(frozen_range) if freeze_on else None,
    }

    # Seção 4 — pedidos firmes (linha “Em carteira”)
    sec4 = {
        "firm_orders": orders_df.copy(),
    }

    # Seção 5 — EPQ / custos
    # guarda nas duas bases (informada e mensal); a mensal é a mais usada nos cálculos
    D_month = float(D) if time_base == "por mês" else float(D) / 12.0
    p_month = float(p) if time_base == "por mês" else float(p) / 12.0
    sec5 = {
        "time_base": time_base,   # "por mês" | "por ano"
        "A": float(A),
        "D": float(D),            # na base escolhida
        "p": float(p),            # na base escolhida
        "D_month": float(D_month),
        "p_month": float(p_month),
        "h_mode": h_mode,
        "H": float(H) if 'H' in locals() and H is not None else None,
        "r": float(r) if 'r' in locals() and r is not None else None,
        "v": float(v),            # v é obrigatório e sempre salvo
        "pi_shortage": float(pi_shortage),
    }

    # junta tudo e salva
    new_cfg.update(sec1)
    new_cfg.update(sec2)
    new_cfg.update(sec3)
    new_cfg.update(sec4)
    new_cfg.update(sec5)

    st.session_state["mps_inputs"] = new_cfg
    st.session_state["mps_firm_orders"] = orders_df.copy()

    st.success(tr('Inputs do MPS salvos com sucesso! ✅'))

# -------- Navegação --------
st.divider()
c_back, c_next = st.columns([1, 1], gap="large")
with c_back:
    st.page_link("pages/04_Previsao.py", label=tr('⬅️ Retornar para Previsão'))
with c_next:
    st.page_link("pages/06_MPS.py", label=tr('➡️ Ir para MPS (Plano Mestre de Produção)'))
