import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(page_title="AktieRadar", page_icon="📈", layout="wide")
st.title("📈 AktieRadar – aktiescanner")
st.caption("Rangerer aktier efter historiske signaler. Det er et analyseværktøj – ikke en sikker kursprognose eller personlig investeringsrådgivning.")

DEFAULT_TICKERS = """NVDA, MSFT, AMZN, GOOGL, META, AVGO, AMD, PLTR, TSLA,
LLY, JPM, V, COST, NFLX, ASML, TSM, SHOP, CRWD, MELI, NU,
NOVO-B.CO, DSV.CO, VWS.CO, MAERSK-B.CO, ORSTED.CO, CARL-B.CO,
GEN.CO, DEMANT.CO, PNDORA.CO, ZEAL.CO"""

with st.sidebar:
    st.header("Indstillinger")
    tickers_text = st.text_area("Aktier (Yahoo Finance-symboler, adskilt af komma)", DEFAULT_TICKERS, height=220)
    period = st.selectbox("Historik", ["2y", "5y"], index=0)
    top_n = st.slider("Vis top", 5, 30, 10)
    capital = st.number_input("Eksempelbeløb (DKK)", min_value=0, value=60000, step=5000)
    holdings = st.slider("Antal aktier i eksempelportefølje", 3, 10, 5)
    st.caption("Danske aktier bruger ofte .CO, fx NOVO-B.CO. Kontrollér altid symbol og børs.")

@st.cache_data(ttl=900, show_spinner=False)
def get_history(symbol, period):
    return yf.Ticker(symbol).history(period=period, auto_adjust=True)

def calc_metrics(symbol, hist):
    if hist is None or hist.empty or len(hist) < 220:
        return None
    close = hist["Close"].dropna()
    volume = hist["Volume"].reindex(close.index).fillna(0)
    if len(close) < 220:
        return None
    last = float(close.iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    ret6 = float(last / close.iloc[-min(127, len(close))] - 1)
    ret12 = float(last / close.iloc[-min(252, len(close))] - 1)
    daily = close.pct_change().dropna()
    vol = float(daily.tail(60).std() * np.sqrt(252))
    peak = close.cummax()
    drawdown = float((last / peak.iloc[-1]) - 1)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])
    avgvol = float(volume.tail(20).mean())
    prevvol = float(volume.tail(60).head(40).mean()) if len(volume) >= 60 else avgvol
    volratio = avgvol / prevvol if prevvol > 0 else 1.0
    return dict(Symbol=symbol, Pris=last, **{"6 mdr.":ret6, "12 mdr.":ret12, "RSI":rsi,
        "Volatilitet":vol, "Fald fra top":drawdown, "Over 50d":last>ma50,
        "Over 200d":last>ma200, "Volumenforhold":volratio})

def score_rows(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Percentilbaserede scores reducerer afhængigheden af aktiernes prisniveau.
    def pct(s, higher=True):
        r = s.rank(pct=True)
        return r if higher else 1-r
    momentum = 0.5*pct(df["6 mdr."]) + 0.5*pct(df["12 mdr."])
    trend = 0.5*df["Over 50d"].astype(float) + 0.5*df["Over 200d"].astype(float)
    rsi_score = 1 - (abs(df["RSI"] - 60) / 60).clip(0, 1)
    risk_score = 0.5*pct(df["Volatilitet"], False) + 0.5*pct(df["Fald fra top"])
    volume_score = pct(df["Volumenforhold"].clip(upper=3))
    df["Score"] = (100*(0.35*momentum + 0.25*trend + 0.15*rsi_score +
                        0.15*risk_score + 0.10*volume_score)).round(1)
    return df.sort_values("Score", ascending=False)

if st.button("🔎 Scan aktier", type="primary", use_container_width=True):
    symbols = list(dict.fromkeys(x.strip().upper() for x in tickers_text.replace("\n", ",").split(",") if x.strip()))
    if not symbols:
        st.error("Indtast mindst ét aktiesymbol.")
        st.stop()
    rows, errors = [], []
    progress = st.progress(0)
    status = st.empty()
    for i, symbol in enumerate(symbols):
        status.write(f"Henter {symbol} …")
        try:
            metrics = calc_metrics(symbol, get_history(symbol, period))
            if metrics:
                rows.append(metrics)
            else:
                errors.append(symbol)
        except Exception:
            errors.append(symbol)
        progress.progress((i+1)/len(symbols))
    status.empty()
    progress.empty()
    ranked = score_rows(rows)
    if ranked.empty:
        st.error("Ingen brugbare kursdata. Kontrollér symbolerne og prøv igen.")
        st.stop()
    st.session_state["ranked"] = ranked
    st.session_state["errors"] = errors

if "ranked" in st.session_state:
    df = st.session_state["ranked"].copy()
    st.subheader("Dagens rangliste")
    st.caption(f"Senest hentede kursdata: {date.today().isoformat()} · Data kan være forsinkede eller mangle.")
    show = df.head(top_n).copy()
    show["6 mdr."] = show["6 mdr."].map(lambda x: f"{x:.1%}")
    show["12 mdr."] = show["12 mdr."].map(lambda x: f"{x:.1%}")
    show["Volatilitet"] = show["Volatilitet"].map(lambda x: f"{x:.1%}")
    show["Fald fra top"] = show["Fald fra top"].map(lambda x: f"{x:.1%}")
    show["RSI"] = show["RSI"].map(lambda x: f"{x:.0f}")
    st.dataframe(show[["Symbol","Score","Pris","6 mdr.","12 mdr.","RSI","Volatilitet","Fald fra top","Over 50d","Over 200d"]],
                 use_container_width=True, hide_index=True)
    st.download_button("Download rangliste som CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       "aktierangliste.csv", "text/csv")
    st.subheader("Eksempel på ligelig fordeling")
    selected = df.head(holdings)
    if len(selected):
        per_stock = capital / len(selected)
        alloc = selected[["Symbol","Score"]].copy()
        alloc["Eksempelbeløb (DKK)"] = round(per_stock, 2)
        st.dataframe(alloc, use_container_width=True, hide_index=True)
        st.info(f"{capital:,.0f} kr. fordelt ligeligt på {len(selected)} aktier = {per_stock:,.0f} kr. pr. aktie. Dette er kun en regneøvelse, ikke en købsanbefaling.")
    if st.session_state.get("errors"):
        st.warning("Ingen eller utilstrækkelige data for: " + ", ".join(st.session_state["errors"]))
    st.subheader("Sådan læses scoren")
    st.write("Scoren er en relativ rangering blandt de indtastede aktier: momentum (35 %), trend (25 %), RSI (15 %), historisk risiko/fald fra top (15 %) og volumen (10 %). Den er ikke en sandsynlighed for afkast.")
    st.caption("Vigtigt: Historiske afkast forudsiger ikke fremtidige afkast. Scanneren vurderer kun den liste, du indtaster – den gennemsøger ikke automatisk alle børsnoterede aktier. Undersøg regnskaber, gæld, værdiansættelse, valuta, skat og handelsomkostninger før investering.")
else:
    st.info("Tryk på **Scan aktier** for at hente kursdata og beregne ranglisten.")
