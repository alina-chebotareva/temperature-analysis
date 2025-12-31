import time
import asyncio
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from utils import (
    load_temperature_csv,
    city_slice,
    add_rolling,
    season_stats_for_city,
    add_season_bounds,
    fetch_current_weather_sync,
    fetch_current_weather_async,  # должна существовать в utils
)

st.set_page_config(page_title="Эксперименты", layout="wide")
st.title("Эксперименты")

# ---------------- Sidebar ----------------
st.sidebar.header("Данные")
uploaded = st.sidebar.file_uploader("temperature_data.csv", type=["csv"])
if uploaded is None:
    st.info("Загрузи файл `temperature_data.csv` в меню слева.")
    st.stop()

df = load_temperature_csv(uploaded)
cities = sorted(df["city"].unique())

st.sidebar.header("Ключ доступа")
api_key = st.sidebar.text_input("API key", type="password")


# ---------------- Helpers ----------------
def _extract_temp_from_owm_response(data: dict) -> float:
    if str(data.get("cod")) != "200":
        raise ValueError("API error")
    return float(data["main"]["temp"])


async def _async_many(cities_list: list[str], key: str) -> pd.DataFrame:
    tasks = [fetch_current_weather_async(city, key) for city in cities_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rows = []
    for c, r in zip(cities_list, results):
        try:
            if isinstance(r, Exception):
                rows.append({"city": c, "temp": None})
            else:
                rows.append({"city": c, "temp": _extract_temp_from_owm_response(r)})
        except Exception:
            rows.append({"city": c, "temp": None})

    return pd.DataFrame(rows)


def analyze_city(city_name: str, df_all: pd.DataFrame, window: int) -> dict:
    cdf = city_slice(df_all, city_name)
    cdf = add_rolling(cdf, window=window)
    season_stats = season_stats_for_city(cdf)
    cdf = add_season_bounds(cdf, season_stats)

    return {
        "city": city_name,
        "rolling_anomalies": int(cdf["roll_anomaly"].sum()),
        "season_anomalies": int(cdf["season_anomaly"].sum()),
    }


# ---------------- Layout ----------------
tab_api, tab_parallel = st.tabs(["Sync vs Async", "Sequential vs Parallel"])

with tab_api:
    st.subheader("Запрос текущей температуры")

    if not api_key:
        st.info("Введи API key в меню слева.")
        st.stop()

    sel = st.multiselect("Города", options=cities, default=cities[:5])

    col_btn, _ = st.columns([1, 6])
    with col_btn:
        run_api = st.button("Запустить", use_container_width=True, key="run_api")

    if run_api:
        if not sel:
            st.warning("Выбери хотя бы один город.")
            st.stop()

        # --- sync benchmark ---
        t0 = time.perf_counter()
        sync_rows = []
        for c in sel:
            try:
                data = fetch_current_weather_sync(c, api_key)
                temp = _extract_temp_from_owm_response(data)
                sync_rows.append({"city": c, "temp": temp})
            except Exception:
                sync_rows.append({"city": c, "temp": None})
        t_sync = time.perf_counter() - t0

        # --- async benchmark ---
        t1 = time.perf_counter()
        async_df = asyncio.run(_async_many(sel, api_key))
        t_async = time.perf_counter() - t1

        sync_df = pd.DataFrame(sync_rows)

        m1, m2, m3 = st.columns(3)
        m1.metric("Sync, сек", f"{t_sync:.3f}")
        m2.metric("Async, сек", f"{t_async:.3f}")
        m3.metric("Ускорение, раз", f"{(t_sync / t_async):.2f}" if t_async > 0 else "—")

        out = async_df.copy()
        out["temp"] = out["temp"].round(2)
        st.dataframe(out[["city", "temp"]], use_container_width=True, hide_index=True)

with tab_parallel:
    st.subheader("Исторический анализ по городам")

    window = st.slider("Окно rolling, дней", 7, 60, 30)
    workers = st.slider("Потоков", 1, 16, 8)

    col_btn, _ = st.columns([1, 6])
    with col_btn:
        run_hist = st.button("Запустить", use_container_width=True, key="run_hist")

    if run_hist:
        # --- sequential ---
        t0 = time.perf_counter()
        seq_rows = [analyze_city(c, df, window) for c in cities]
        t_seq = time.perf_counter() - t0

        # --- parallel ---
        t1 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            par_rows = list(ex.map(lambda c: analyze_city(c, df, window), cities))
        t_par = time.perf_counter() - t1

        m1, m2, m3 = st.columns(3)
        m1.metric("Sequential, сек", f"{t_seq:.3f}")
        m2.metric("Parallel, сек", f"{t_par:.3f}")
        m3.metric("Ускорение, раз", f"{(t_seq / t_par):.2f}" if t_par > 0 else "—")

        res = pd.DataFrame(par_rows).sort_values("city").reset_index(drop=True)
        st.dataframe(
            res[["city", "rolling_anomalies", "season_anomalies"]],
            use_container_width=True,
            hide_index=True,
        )
