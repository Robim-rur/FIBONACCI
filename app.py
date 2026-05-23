# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Scanner Fibonacci Probabilístico PRO",
    layout="wide"
)

# =========================================================
# ATIVOS
# =========================================================

ATIVOS = [

    "PETR4.SA","VALE3.SA","BBAS3.SA","ITUB4.SA","BBDC4.SA",
    "WEGE3.SA","PRIO3.SA","RENT3.SA","ELET3.SA","ELET6.SA",
    "CPLE6.SA","CMIG4.SA","TAEE11.SA","EGIE3.SA","VIVT3.SA",
    "TIMS3.SA","ABEV3.SA","RADL3.SA","SUZB3.SA","GGBR4.SA",
    "GOAU4.SA","USIM5.SA","CSNA3.SA","RAIL3.SA","SBSP3.SA",
    "EQTL3.SA","HYPE3.SA","MULT3.SA","LREN3.SA","ARZZ3.SA",
    "TOTS3.SA","EMBR3.SA","JBSS3.SA","BEEF3.SA","MRFG3.SA",
    "BRFS3.SA","SLCE3.SA","SMTO3.SA","B3SA3.SA","BBSE3.SA",
    "BPAC11.SA","SANB11.SA","ITSA4.SA",

    "BOVA11.SA","IVVB11.SA","SMAL11.SA","GOLD11.SA",

    "HGLG11.SA","XPLG11.SA","MXRF11.SA","KNRI11.SA",
    "AUVP11.SA",

    "AAPL34.SA","AMZO34.SA","GOGL34.SA","MSFT34.SA",
    "TSLA34.SA","META34.SA","NVDC34.SA"

]

# =========================================================
# FUNÇÕES
# =========================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


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
# PROBABILIDADE FIBONACCI
# =========================================================

def calcular_probabilidade_fib(df):

    try:

        resultados = {
            "38.2": 0,
            "50": 0,
            "61.8": 0
        }

        total_382 = 0
        total_50 = 0
        total_618 = 0

        janela = 30

        for i in range(120, len(df) - 20):

            trecho = df.iloc[i-60:i]

            topo = trecho["High"].max()
            fundo = trecho["Low"].min()

            if fundo <= 0:
                continue

            impulso = (topo - fundo) / fundo

            if impulso < 0.08:
                continue

            diff = topo - fundo

            fib_382 = topo - (diff * 0.382)
            fib_50 = topo - (diff * 0.5)
            fib_618 = topo - (diff * 0.618)

            preco = df.iloc[i]["Close"]

            futuro = df.iloc[i:i+15]

            max_futuro = futuro["High"].max()

            ganho = (
                (max_futuro - preco)
                /
                preco
            )

            # =================================================
            # FIB 38.2
            # =================================================

            if abs(preco - fib_382) / fib_382 <= 0.01:

                total_382 += 1

                if ganho >= 0.05:
                    resultados["38.2"] += 1

            # =================================================
            # FIB 50
            # =================================================

            if abs(preco - fib_50) / fib_50 <= 0.01:

                total_50 += 1

                if ganho >= 0.05:
                    resultados["50"] += 1

            # =================================================
            # FIB 61.8
            # =================================================

            if abs(preco - fib_618) / fib_618 <= 0.01:

                total_618 += 1

                if ganho >= 0.05:
                    resultados["61.8"] += 1

        probabilidades = {}

        probabilidades["38.2"] = (
            round(
                (resultados["38.2"] / total_382) * 100,
                1
            )
            if total_382 > 0
            else 0
        )

        probabilidades["50"] = (
            round(
                (resultados["50"] / total_50) * 100,
                1
            )
            if total_50 > 0
            else 0
        )

        probabilidades["61.8"] = (
            round(
                (resultados["61.8"] / total_618) * 100,
                1
            )
            if total_618 > 0
            else 0
        )

        melhor_fib = max(
            probabilidades,
            key=probabilidades.get
        )

        melhor_prob = probabilidades[melhor_fib]

        return melhor_fib, melhor_prob

    except:

        return "-", 0


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
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        # =====================================================
        # MULTIINDEX
        # =====================================================

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)

        # =====================================================
        # LIMPEZA
        # =====================================================

        df = df[
            ["Open", "High", "Low", "Close", "Volume"]
        ].copy()

        df.dropna(inplace=True)

        if len(df) < 250:
            return None

        # =====================================================
        # INDICADORES
        # =====================================================

        df["EMA69"] = ema(df["Close"], 69)

        df["K"], df["D"] = stochastic(df)

        df["DI+"], df["DI-"], df["ADX"] = dmi(df)

        df.dropna(inplace=True)

        if len(df) < 100:
            return None

        # =====================================================
        # FIBONACCI
        # =====================================================

        topo, fundo, fib_382, fib_50, fib_618 = (
            calcular_fibonacci(df)
        )

        # =====================================================
        # PROBABILIDADE
        # =====================================================

        melhor_fib, melhor_prob = (
            calcular_probabilidade_fib(df)
        )

        # =====================================================
        # CANDLE
        # =====================================================

        hoje = df.iloc[-1]

        close = float(hoje["Close"])

        # =====================================================
        # FILTROS
        # =====================================================

        tendencia = (
            close > hoje["EMA69"]
        )

        fib_ok = (
            close <= fib_382
            and
            close >= fib_618
        )

        dmi_ok = (
            hoje["DI+"] > hoje["DI-"]
        )

        adx_ok = (
            hoje["ADX"] > 16
        )

        estocastico_ok = (
            hoje["K"] > hoje["D"]
        )

        # =====================================================
        # SCORE
        # =====================================================

        score = sum([
            tendencia,
            fib_ok,
            dmi_ok,
            adx_ok,
            estocastico_ok
        ])

        # =====================================================
        # STATUS
        # =====================================================

        if score >= 5:

            status = "🔥 ENTRADA IDEAL"

        elif score >= 4:

            status = "🟢 ENTRADA ANTECIPADA"

        elif score >= 3:

            status = "🟡 OBSERVAÇÃO"

        else:

            status = "❌ DESCARTAR"

        # =====================================================
        # RESULTADO
        # =====================================================

        return {

            "Ativo": ticker.replace(".SA", ""),

            "Status": status,

            "Score": score,

            "Preço": round(close, 2),

            "Melhor Fib": melhor_fib,

            "Probabilidade %": melhor_prob,

            "ADX": round(float(hoje["ADX"]), 1),

            "DI+": round(float(hoje["DI+"]), 1),

            "DI-": round(float(hoje["DI-"]), 1),

            "K": round(float(hoje["K"]), 1),

            "D": round(float(hoje["D"]), 1),

            "Fib 38.2": round(float(fib_382), 2),

            "Fib 50": round(float(fib_50), 2),

            "Fib 61.8": round(float(fib_618), 2)

        }

    except:

        return None


# =========================================================
# TÍTULO
# =========================================================

st.title("📈 Scanner Fibonacci Probabilístico PRO")

st.markdown("""

### O scanner utiliza:

- EMA69
- Fibonacci
- DMI
- ADX
- Estocástico
- Estatística histórica

### NOVA FUNÇÃO:

Agora o app calcula:
- qual Fibonacci historicamente funciona melhor;
- probabilidade estatística de reação.

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

        ordem = {

            "🔥 ENTRADA IDEAL": 0,
            "🟢 ENTRADA ANTECIPADA": 1,
            "🟡 OBSERVAÇÃO": 2,
            "❌ DESCARTAR": 3

        }

        df_resultados["Ordem"] = (
            df_resultados["Status"].map(ordem)
        )

        df_resultados = df_resultados.sort_values(
            by=[
                "Ordem",
                "Probabilidade %",
                "Score"
            ],
            ascending=[True, False, False]
        )

        df_resultados.drop(
            columns=["Ordem"],
            inplace=True
        )

        mostrar_descartados = st.checkbox(
            "Mostrar descartados",
            value=False
        )

        if not mostrar_descartados:

            df_resultados = df_resultados[
                df_resultados["Status"]
                !=
                "❌ DESCARTAR"
            ]

        st.dataframe(
            df_resultados,
            use_container_width=True,
            height=700
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

        observacao = len(
            df_resultados[
                df_resultados["Status"]
                ==
                "🟡 OBSERVAÇÃO"
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
            "🟡 Observação",
            observacao
        )

    else:

        st.error(
            "Nenhum resultado encontrado."
        )

# =========================================================
# RODAPÉ
# =========================================================

st.markdown("---")

st.caption(
    "Scanner Fibonacci Probabilístico PRO"
)
