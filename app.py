# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Scanner Fibonacci Inteligente PRO",
    layout="wide"
)

# =========================================================
# ATIVOS
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
# EMA
# =========================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()

# =========================================================
# ESTOCÁSTICO
# =========================================================

def stochastic(df, period=14):

    low_min = df["Low"].rolling(period).min()

    high_max = df["High"].rolling(period).max()

    k = 100 * (
        (df["Close"] - low_min)
        /
        (high_max - low_min)
    )

    k = k.rolling(3).mean()

    d = k.rolling(3).mean()

    return k, d

# =========================================================
# DMI / ADX
# =========================================================

def dmi(df, period=14):

    high = df["High"]

    low = df["Low"]

    close = df["Close"]

    plus_dm = high.diff()

    minus_dm = low.diff() * -1

    plus_dm[plus_dm < 0] = 0

    minus_dm[minus_dm < 0] = 0

    tr1 = high - low

    tr2 = abs(high - close.shift())

    tr3 = abs(low - close.shift())

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = (
        100
        * plus_dm.rolling(period).mean()
        / atr
    )

    minus_di = (
        100
        * minus_dm.rolling(period).mean()
        / atr
    )

    dx = (
        abs(plus_di - minus_di)
        /
        (plus_di + minus_di)
    ) * 100

    adx = dx.rolling(period).mean()

    return plus_di, minus_di, adx

# =========================================================
# FIBONACCI
# =========================================================

def calcular_fibs(df):

    lookback = 60

    topo = df["High"].tail(lookback).max()

    fundo = df["Low"].tail(lookback).min()

    diff = topo - fundo

    fib_382 = topo - (diff * 0.382)

    fib_50 = topo - (diff * 0.5)

    fib_618 = topo - (diff * 0.618)

    return topo, fundo, fib_382, fib_50, fib_618

# =========================================================
# MELHOR FIB HISTÓRICA
# =========================================================

def melhor_fib_historica(df, ticker):

    resultados = {
        "38.2": [],
        "50": [],
        "61.8": []
    }

    # =====================================================
    # IMPULSO MÍNIMO ADAPTATIVO
    # =====================================================

    if "11.SA" in ticker:

        impulso_min = 0.04

    else:

        impulso_min = 0.08

    for i in range(120, len(df) - 20):

        trecho = df.iloc[i-60:i]

        topo = trecho["High"].max()

        fundo = trecho["Low"].min()

        if fundo <= 0:
            continue

        impulso = (topo - fundo) / fundo

        # =================================================
        # FILTRO DE IMPULSO
        # =================================================

        if impulso < impulso_min:
            continue

        diff = topo - fundo

        fib_382 = topo - (diff * 0.382)

        fib_50 = topo - (diff * 0.5)

        fib_618 = topo - (diff * 0.618)

        preco = df.iloc[i]["Close"]

        futuro = df.iloc[i:i+15]

        ganho_max = (
            futuro["High"].max() - preco
        ) / preco

        if abs(preco - fib_382) / fib_382 <= 0.01:

            resultados["38.2"].append(
                ganho_max >= 0.05
            )

        if abs(preco - fib_50) / fib_50 <= 0.01:

            resultados["50"].append(
                ganho_max >= 0.05
            )

        if abs(preco - fib_618) / fib_618 <= 0.01:

            resultados["61.8"].append(
                ganho_max >= 0.05
            )

    probabilidades = {}

    for fib in resultados:

        dados = resultados[fib]

        if len(dados) > 0:

            probabilidades[fib] = round(
                (
                    sum(dados)
                    /
                    len(dados)
                ) * 100,
                1
            )

        else:

            probabilidades[fib] = 0

    melhor_fib = max(
        probabilidades,
        key=probabilidades.get
    )

    return melhor_fib, probabilidades

# =========================================================
# IDENTIFICAR FIB ATUAL
# =========================================================

def identificar_fib_atual(
    close,
    fib_382,
    fib_50,
    fib_618
):

    if close >= fib_382:

        return "38.2"

    elif close >= fib_50:

        return "50"

    elif close >= fib_618:

        return "61.8"

    else:

        return "ABAIXO 61.8"

# =========================================================
# CHANCE REVERSÃO
# =========================================================

def chance_reversao(
    close,
    fib_atual,
    melhor_fib,
    hoje
):

    score = 0

    # =====================================================
    # FIBONACCI
    # =====================================================

    if fib_atual == melhor_fib:

        score += 35

    elif (
        fib_atual == "50"
        and
        melhor_fib == "61.8"
    ):

        score += 15

    elif (
        fib_atual == "38.2"
        and
        melhor_fib == "50"
    ):

        score += 15

    # =====================================================
    # ESTOCÁSTICO
    # =====================================================

    if hoje["K"] > hoje["D"]:

        score += 20

    # =====================================================
    # DMI
    # =====================================================

    if hoje["DI+"] > hoje["DI-"]:

        score += 20

    # =====================================================
    # ADX
    # =====================================================

    if hoje["ADX"] > 18:

        score += 15

    # =====================================================
    # EMA69
    # =====================================================

    if close > hoje["EMA69"]:

        score += 10

    return min(score, 95)

# =========================================================
# CONTINUAÇÃO CORREÇÃO
# =========================================================

def chance_continuacao(
    fib_atual,
    melhor_fib,
    hoje
):

    score = 0

    # =====================================================
    # FIBONACCI
    # =====================================================

    if (
        fib_atual == "38.2"
        and
        melhor_fib == "61.8"
    ):

        score += 40

    elif (
        fib_atual == "50"
        and
        melhor_fib == "61.8"
    ):

        score += 30

    elif (
        fib_atual == "38.2"
        and
        melhor_fib == "50"
    ):

        score += 25

    # =====================================================
    # ESTOCÁSTICO
    # =====================================================

    if hoje["K"] < hoje["D"]:

        score += 20

    # =====================================================
    # DMI
    # =====================================================

    if hoje["DI-"] > hoje["DI+"]:

        score += 20

    # =====================================================
    # ADX
    # =====================================================

    if hoje["ADX"] > 20:

        score += 10

    return min(score, 95)

# =========================================================
# ALVOS FIBONACCI
# =========================================================

def calcular_alvos(
    topo,
    fundo,
    close
):

    impulso = topo - fundo

    alvo_1 = topo

    alvo_1272 = topo + (impulso * 0.272)

    alvo_1618 = topo + (impulso * 0.618)

    gain_1 = (
        (alvo_1 - close)
        / close
    ) * 100

    gain_1272 = (
        (alvo_1272 - close)
        / close
    ) * 100

    gain_1618 = (
        (alvo_1618 - close)
        / close
    ) * 100

    return (
        alvo_1,
        alvo_1272,
        alvo_1618,
        gain_1,
        gain_1272,
        gain_1618
    )

# =========================================================
# CLASSIFICAÇÃO
# =========================================================

def classificar(
    reversao,
    continuacao,
    fib_atual
):

    if (
        reversao >= 70
        and
        fib_atual in ["50", "61.8"]
    ):

        return "🔥 ENTRADA IDEAL"

    elif (
        reversao >= 55
        and
        fib_atual == "38.2"
    ):

        return "🟢 ENTRADA ANTECIPADA"

    elif continuacao >= 55:

        return "⏳ ESPERAR CORREÇÃO"

    else:

        return "🟡 OBSERVAÇÃO"

# =========================================================
# ANÁLISE
# =========================================================

def analisar_ativo(ticker):

    try:

        df = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = (
                df.columns.get_level_values(0)
            )

        df = df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        ].copy()

        df.dropna(inplace=True)

        if len(df) < 250:

            return None

        # =================================================
        # INDICADORES
        # =================================================

        df["EMA69"] = ema(
            df["Close"],
            69
        )

        df["K"], df["D"] = stochastic(df)

        df["DI+"], df["DI-"], df["ADX"] = dmi(df)

        df.dropna(inplace=True)

        # =================================================
        # FIBS
        # =================================================

        topo, fundo, fib_382, fib_50, fib_618 = (
            calcular_fibs(df)
        )

        # =================================================
        # MELHOR FIB
        # =================================================

        melhor_fib, probs = (
            melhor_fib_historica(
                df,
                ticker
            )
        )

        hoje = df.iloc[-1]

        close = float(hoje["Close"])

        # =================================================
        # FIB ATUAL
        # =================================================

        fib_atual = identificar_fib_atual(
            close,
            fib_382,
            fib_50,
            fib_618
        )

        # =================================================
        # REVERSÃO
        # =================================================

        reversao = chance_reversao(
            close,
            fib_atual,
            melhor_fib,
            hoje
        )

        # =================================================
        # CONTINUAÇÃO
        # =================================================

        continuacao = chance_continuacao(
            fib_atual,
            melhor_fib,
            hoje
        )

        # =================================================
        # SCORE
        # =================================================

        score = 0

        if close > hoje["EMA69"]:
            score += 1

        if hoje["DI+"] > hoje["DI-"]:
            score += 1

        if hoje["ADX"] > 18:
            score += 1

        if hoje["K"] > hoje["D"]:
            score += 1

        if fib_atual in ["50", "61.8"]:
            score += 1

        # =================================================
        # ALVOS
        # =================================================

        (
            alvo_1,
            alvo_1272,
            alvo_1618,
            gain_1,
            gain_1272,
            gain_1618
        ) = calcular_alvos(
            topo,
            fundo,
            close
        )

        # =================================================
        # STATUS
        # =================================================

        status = classificar(
            reversao,
            continuacao,
            fib_atual
        )

        return {

            "Ativo": ticker.replace(".SA", ""),

            "Status": status,

            "Preço": round(close, 2),

            "Fib Atual": fib_atual,

            "Melhor Fib": melhor_fib,

            "Chance Reversão %": reversao,

            "Chance Continuar Correção %": continuacao,

            "Prob 38.2": probs["38.2"],

            "Prob 50": probs["50"],

            "Prob 61.8": probs["61.8"],

            "Score": score,

            "ADX": round(float(hoje["ADX"]), 1),

            "DI+": round(float(hoje["DI+"]), 1),

            "DI-": round(float(hoje["DI-"]), 1),

            "K": round(float(hoje["K"]), 1),

            "D": round(float(hoje["D"]), 1),

            "Fib 38.2": round(float(fib_382), 2),

            "Fib 50": round(float(fib_50), 2),

            "Fib 61.8": round(float(fib_618), 2),

            "Alvo 1.0": round(float(alvo_1), 2),

            "Gain 1.0 %": round(float(gain_1), 1),

            "Alvo 1.272": round(float(alvo_1272), 2),

            "Gain 1.272 %": round(float(gain_1272), 1),

            "Alvo 1.618": round(float(alvo_1618), 2),

            "Gain 1.618 %": round(float(gain_1618), 1)

        }

    except:

        return None

# =========================================================
# APP
# =========================================================

st.title(
    "📈 Scanner Fibonacci Inteligente PRO"
)

st.markdown("""

### O scanner identifica:

- Região atual da correção;
- Melhor Fibonacci histórica;
- Chance de reversão;
- Chance de continuar corrigindo;
- Alvos Fibonacci;
- Gains prováveis.

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

        progresso.progress(
            (i + 1) / total
        )

    # =====================================================
    # RESULTADOS
    # =====================================================

    if resultados:

        df_resultados = pd.DataFrame(resultados)

        ordem = {

            "🔥 ENTRADA IDEAL": 0,
            "🟢 ENTRADA ANTECIPADA": 1,
            "⏳ ESPERAR CORREÇÃO": 2,
            "🟡 OBSERVAÇÃO": 3

        }

        df_resultados["Ordem"] = (
            df_resultados["Status"].map(ordem)
        )

        df_resultados = df_resultados.sort_values(
            by=[
                "Ordem",
                "Chance Reversão %",
                "Gain 1.272 %",
                "Score"
            ],
            ascending=[True, False, False, False]
        )

        df_resultados.drop(
            columns=["Ordem"],
            inplace=True
        )

        mostrar_observacao = st.checkbox(
            "Mostrar observação",
            value=True
        )

        if not mostrar_observacao:

            df_resultados = df_resultados[
                df_resultados["Status"]
                !=
                "🟡 OBSERVAÇÃO"
            ]

        st.dataframe(
            df_resultados,
            use_container_width=True,
            height=750
        )

        # =================================================
        # RESUMO
        # =================================================

        ideal = len(
            df_resultados[
                df_resultados["Status"]
                ==
                "🔥 ENTRADA IDEAL"
            ]
        )

        antecipada = len(
            df_resultados[
                df_resultados["Status"]
                ==
                "🟢 ENTRADA ANTECIPADA"
            ]
        )

        esperar = len(
            df_resultados[
                df_resultados["Status"]
                ==
                "⏳ ESPERAR CORREÇÃO"
            ]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🔥 Entradas Ideais",
            ideal
        )

        col2.metric(
            "🟢 Antecipadas",
            antecipada
        )

        col3.metric(
            "⏳ Esperar Correção",
            esperar
        )

    else:

        st.error(
            "Nenhum ativo encontrado."
        )

# =========================================================
# RODAPÉ
# =========================================================

st.markdown("---")

st.caption(
    "Scanner Fibonacci Inteligente PRO"
)
