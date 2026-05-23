# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# =========================================================
# CONFIGURAÇÃO
# =========================================================

st.set_page_config(
    page_title="Scanner Fibonacci Tendencial",
    layout="wide"
)

# =========================================================
# LISTA DE ATIVOS
# =========================================================

ATIVOS = [

    # =====================================================
    # AÇÕES
    # =====================================================

    "PETR4.SA",
    "VALE3.SA",
    "BBAS3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "WEGE3.SA",
    "PRIO3.SA",
    "RENT3.SA",

    "ELET3.SA",
    "ELET6.SA",
    "CPLE6.SA",
    "CMIG4.SA",
    "TAEE11.SA",
    "EGIE3.SA",
    "VIVT3.SA",
    "TIMS3.SA",

    "ABEV3.SA",
    "RADL3.SA",
    "SUZB3.SA",
    "GGBR4.SA",
    "GOAU4.SA",
    "USIM5.SA",
    "CSNA3.SA",
    "RAIL3.SA",

    "SBSP3.SA",
    "EQTL3.SA",
    "HYPE3.SA",
    "MULT3.SA",
    "LREN3.SA",
    "ARZZ3.SA",
    "TOTS3.SA",
    "EMBR3.SA",

    "JBSS3.SA",
    "BEEF3.SA",
    "MRFG3.SA",
    "BRFS3.SA",
    "SLCE3.SA",
    "SMTO3.SA",
    "B3SA3.SA",
    "BBSE3.SA",

    "BPAC11.SA",
    "SANB11.SA",
    "ITSA4.SA",
    "BRSR6.SA",
    "CXSE3.SA",
    "POMO4.SA",
    "STBP3.SA",
    "TUPY3.SA",

    "DIRR3.SA",
    "CYRE3.SA",
    "EZTC3.SA",
    "JHSF3.SA",
    "KEPL3.SA",
    "POSI3.SA",
    "MOVI3.SA",
    "PETZ3.SA",

    "COGN3.SA",
    "YDUQ3.SA",
    "MGLU3.SA",
    "NTCO3.SA",
    "AZUL4.SA",
    "GOLL4.SA",
    "CVCB3.SA",
    "RRRP3.SA",

    "RECV3.SA",
    "ENAT3.SA",
    "ORVR3.SA",
    "AURE3.SA",
    "ENEV3.SA",
    "UGPA3.SA",

    # =====================================================
    # ETFs
    # =====================================================

    "BOVA11.SA",
    "IVVB11.SA",
    "SMAL11.SA",
    "HASH11.SA",
    "GOLD11.SA",
    "DIVO11.SA",
    "NDIV11.SA",

    # =====================================================
    # FIIs
    # =====================================================

    "HGLG11.SA",
    "XPLG11.SA",
    "VISC11.SA",
    "MXRF11.SA",
    "KNRI11.SA",
    "KNCR11.SA",
    "KNIP11.SA",

    "CPTS11.SA",
    "IRDM11.SA",
    "TRXF11.SA",
    "TGAR11.SA",
    "HGRU11.SA",
    "ALZR11.SA",
    "AUVP11.SA",

    "GARE11.SA",
    "IEEX11.SA",
    "UTLL11.SA",

    # =====================================================
    # BDRs
    # =====================================================

    "AAPL34.SA",
    "AMZO34.SA",
    "GOGL34.SA",
    "MSFT34.SA",
    "TSLA34.SA",
    "META34.SA",
    "NFLX34.SA",

    "NVDC34.SA",
    "MELI34.SA",
    "BABA34.SA",
    "DISB34.SA",
    "PYPL34.SA",
    "JNJB34.SA",
    "VISA34.SA",

    "WMTB34.SA",
    "NIKE34.SA",
    "ADBE34.SA",
    "CSCO34.SA",
    "INTC34.SA",
    "JPMC34.SA",
    "ORCL34.SA"

]

# =========================================================
# FUNÇÕES
# =========================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def stochastic(df, period=14, smooth_k=3, smooth_d=3):

    low_min = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()

    k = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    k = k.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()

    return k, d


def dmi(df, period=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = (
        100 * plus_dm.rolling(period).mean() / atr
    )

    minus_di = (
        100 * minus_dm.rolling(period).mean() / atr
    )

    dx = (
        ((plus_di - minus_di).abs())
        /
        ((plus_di + minus_di).abs())
    ) * 100

    adx = dx.rolling(period).mean()

    return plus_di, minus_di, adx


def calcular_fibonacci(df):

    lookback = 60

    topo = df["High"].tail(lookback).max()
    fundo = df["Low"].tail(lookback).min()

    diff = topo - fundo

    fib_382 = topo - (diff * 0.382)
    fib_50 = topo - (diff * 0.5)
    fib_618 = topo - (diff * 0.618)

    return topo, fundo, fib_382, fib_50, fib_618


# =========================================================
# ANÁLISE
# =========================================================

def analisar_ativo(ticker):

    try:

        # =====================================================
        # DOWNLOAD
        # =====================================================

        df = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df.empty or len(df) < 250:
            return None

        # =====================================================
        # INDICADORES
        # =====================================================

        df["EMA69"] = ema(df["Close"], 69)
        df["EMA21"] = ema(df["Close"], 21)

        df["K"], df["D"] = stochastic(df)

        df["DI+"], df["DI-"], df["ADX"] = dmi(df)

        # =====================================================
        # FIBONACCI
        # =====================================================

        topo, fundo, fib_382, fib_50, fib_618 = calcular_fibonacci(df)

        # =====================================================
        # CANDLES
        # =====================================================

        hoje = df.iloc[-1]
        ontem = df.iloc[-2]

        close = float(hoje["Close"])

        # =====================================================
        # TENDÊNCIA
        # =====================================================

        tendencia = (
            close > hoje["EMA69"]
        )

        # =====================================================
        # IMPULSO
        # =====================================================

        impulso = (
            ((topo - fundo) / fundo) >= 0.08
        )

        # =====================================================
        # FIBONACCI
        # =====================================================

        dentro_fib = (
            close <= fib_382
            and
            close >= fib_618
        )

        # =====================================================
        # DMI
        # =====================================================

        dmi_ok = (
            hoje["DI+"] > hoje["DI-"]
        )

        # =====================================================
        # ADX
        # =====================================================

        adx_ok = (
            hoje["ADX"] > 16
        )

        # =====================================================
        # ESTOCÁSTICO
        # =====================================================

        estocastico_ok = (
            hoje["K"] > hoje["D"]
        )

        # =====================================================
        # CANDLE DE REJEIÇÃO
        # =====================================================

        amplitude = hoje["High"] - hoje["Low"]

        if amplitude == 0:
            amplitude = 0.0001

        candle_rejeicao = (

            hoje["Low"] < ontem["Low"]

            and

            hoje["Close"] >
            (
                hoje["Low"] + (amplitude * 0.6)
            )

        )

        # =====================================================
        # DISTÂNCIA EMA69
        # =====================================================

        dist_ema69 = (
            ((close / hoje["EMA69"]) - 1) * 100
        )

        dist_ema_ok = (
            dist_ema69 <= 15
        )

        # =====================================================
        # SCORE
        # =====================================================

        score = 0

        filtros = {

            "Tendência": tendencia,
            "Impulso": impulso,
            "Fib": dentro_fib,
            "DMI": dmi_ok,
            "ADX": adx_ok,
            "Estocástico": estocastico_ok,
            "Rejeição": candle_rejeicao

        }

        for valor in filtros.values():
            if valor:
                score += 1

        # =====================================================
        # STATUS
        # =====================================================

        if score >= 7:

            status = "🔥 ENTRADA IDEAL"

        elif score >= 5:

            status = "🟢 ENTRADA ANTECIPADA"

        elif score >= 3:

            status = "🟡 OBSERVAÇÃO"

        else:

            status = "❌ DESCARTAR"

        # =====================================================
        # PROXIMIDADE DOS FIBOS
        # =====================================================

        dist_382 = (
            ((close / fib_382) - 1) * 100
        )

        dist_50 = (
            ((close / fib_50) - 1) * 100
        )

        dist_618 = (
            ((close / fib_618) - 1) * 100
        )

        # =====================================================
        # RESULTADO
        # =====================================================

        return {

            "Ativo": ticker.replace(".SA", ""),

            "Status": status,

            "Score": score,

            "Preço": round(close, 2),

            "ADX": round(float(hoje["ADX"]), 1),

            "DI+": round(float(hoje["DI+"]), 1),

            "DI-": round(float(hoje["DI-"]), 1),

            "K": round(float(hoje["K"]), 1),

            "D": round(float(hoje["D"]), 1),

            "Fib 38.2": round(float(fib_382), 2),

            "Fib 50": round(float(fib_50), 2),

            "Fib 61.8": round(float(fib_618), 2),

            "Dist EMA69 %": round(float(dist_ema69), 2),

            "Dist Fib38 %": round(float(dist_382), 2),

            "Dist Fib50 %": round(float(dist_50), 2),

            "Dist Fib61 %": round(float(dist_618), 2),

            "Tendência": "SIM" if tendencia else "NÃO",

            "Fib": "SIM" if dentro_fib else "NÃO",

            "DMI": "SIM" if dmi_ok else "NÃO",

            "ADX Forte": "SIM" if adx_ok else "NÃO",

            "Estocástico": "SIM" if estocastico_ok else "NÃO",

            "Rejeição": "SIM" if candle_rejeicao else "NÃO"

        }

    except:
        return None


# =========================================================
# TÍTULO
# =========================================================

st.title("📈 Scanner Fibonacci Tendencial PRO")

st.markdown("""

### O scanner procura:

- tendência acima da EMA69;
- pullback em Fibonacci;
- DMI comprador;
- ADX forte;
- estocástico comprador;
- candle de rejeição institucional.

### Classificação:

- 🔥 ENTRADA IDEAL → setup extremamente alinhado
- 🟢 ENTRADA ANTECIPADA → excelente para acompanhar
- 🟡 OBSERVAÇÃO → ativo promissor
- ❌ DESCARTAR → fora do padrão

""")

# =========================================================
# BOTÃO
# =========================================================

if st.button("ESCANEAR MERCADO"):

    resultados = []

    progresso = st.progress(0)

    total = len(ATIVOS)

    for i, ativo in enumerate(ATIVOS):

        resultado = analisar_ativo(ativo)

        if resultado:
            resultados.append(resultado)

        progresso.progress((i + 1) / total)

    # =====================================================
    # RESULTADOS
    # =====================================================

    if resultados:

        df_resultados = pd.DataFrame(resultados)

        ordem_status = {

            "🔥 ENTRADA IDEAL": 0,
            "🟢 ENTRADA ANTECIPADA": 1,
            "🟡 OBSERVAÇÃO": 2,
            "❌ DESCARTAR": 3

        }

        df_resultados["Ordem"] = (
            df_resultados["Status"].map(ordem_status)
        )

        df_resultados = df_resultados.sort_values(
            by=["Ordem", "Score"],
            ascending=[True, False]
        )

        df_resultados = df_resultados.drop(
            columns=["Ordem"]
        )

        # =================================================
        # FILTRO
        # =================================================

        mostrar_descartados = st.checkbox(
            "Mostrar ativos descartados",
            value=False
        )

        if not mostrar_descartados:

            df_resultados = df_resultados[
                df_resultados["Status"] != "❌ DESCARTAR"
            ]

        # =================================================
        # RESULTADOS
        # =================================================

        st.subheader("📊 Resultado do Scanner")

        st.dataframe(
            df_resultados,
            use_container_width=True,
            height=700
        )

        # =================================================
        # ENTRADAS IDEAIS
        # =================================================

        entradas_ideais = df_resultados[
            df_resultados["Status"] == "🔥 ENTRADA IDEAL"
        ]

        st.subheader("🔥 Entradas Ideais")

        if not entradas_ideais.empty:

            st.success(
                f"{len(entradas_ideais)} ativo(s) encontrado(s)."
            )

            st.dataframe(
                entradas_ideais,
                use_container_width=True
            )

        else:

            st.warning(
                "Nenhuma entrada ideal encontrada hoje."
            )

        # =================================================
        # ENTRADAS ANTECIPADAS
        # =================================================

        entradas_antecipadas = df_resultados[
            df_resultados["Status"] == "🟢 ENTRADA ANTECIPADA"
        ]

        st.subheader("🟢 Entradas Antecipadas")

        if not entradas_antecipadas.empty:

            st.success(
                f"{len(entradas_antecipadas)} ativo(s) encontrados."
            )

            st.dataframe(
                entradas_antecipadas,
                use_container_width=True
            )

        else:

            st.info(
                "Nenhuma entrada antecipada encontrada."
            )

    else:

        st.error(
            "Nenhum ativo retornou dados."
        )

# =========================================================
# RODAPÉ
# =========================================================

st.markdown("---")

st.caption(
    "Scanner Fibonacci Tendencial PRO | EMA69 + Fibonacci + DMI + ADX + Estocástico"
)
