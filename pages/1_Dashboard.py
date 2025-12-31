import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime

from utils import (
    load_temperature_csv,
    city_slice,
    add_rolling,
    season_stats_for_city,
    add_season_bounds,
    yearly_profile,
    get_season_by_month,
    fetch_current_weather_sync,
)

st.set_page_config(page_title="Анализ", layout="wide")
st.title("Анализ температур по городу")

# ---------------- Sidebar: inputs ----------------
st.sidebar.header("Данные")
uploaded = st.sidebar.file_uploader("temperature_data.csv", type=["csv"])

if uploaded is None:
    st.info("Загрузи файл `temperature_data.csv` в меню слева.")
    st.stop()

df = load_temperature_csv(uploaded)
if df["timestamp"].isna().any():
    st.error("В файле есть строки с некорректной датой в `timestamp`.")
    st.stop()

cities = sorted(df["city"].unique())

st.sidebar.header("Параметры")
city = st.sidebar.selectbox("Город", cities)
window = st.sidebar.slider("Окно сглаживания, дней", 7, 60, 30)

st.sidebar.header("Ключ доступа")
api_key = st.sidebar.text_input("API key", type="password")

# ---------------- Prepare data ----------------
city_df = city_slice(df, city)
city_df = add_rolling(city_df, window=window)

season_stats = season_stats_for_city(city_df)
city_df = add_season_bounds(city_df, season_stats)

# ---------------- Tabs (ничего "закреплённого" до вкладок) ----------------
tab1, tab2, tab3 = st.tabs(["Временной ряд", "Сезоны", "Текущая температура"])

with tab1:
    # метрики + описательная статистика — внутри вкладки "Временной ряд"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Дней", f"{len(city_df):,}".replace(",", " "))
    m2.metric("Период", f"{city_df['timestamp'].min().date()} — {city_df['timestamp'].max().date()}")
    m3.metric("Rolling аномалий", f"{int(city_df['roll_anomaly'].sum())}")
    m4.metric("Сезонных аномалий", f"{int(city_df['season_anomaly'].sum())}")

    st.subheader("Описательная статистика")

    desc = city_df["temperature"].describe()

    desc_row = pd.DataFrame([{
        "city": city,
        "count": int(desc["count"]),
        "mean": round(float(desc["mean"]), 4),
        "std": round(float(desc["std"]), 4),
        "min": round(float(desc["min"]), 4),
        "25%": round(float(desc["25%"]), 4),
        "50%": round(float(desc["50%"]), 4),
        "75%": round(float(desc["75%"]), 4),
        "max": round(float(desc["max"]), 4),
    }])

    st.dataframe(
        desc_row,
        hide_index=True,
        use_container_width=False,
        width=900,
    )

    st.subheader("Температура ")
    fig = px.scatter(
        city_df,
        x="timestamp",
        y="temperature",
        color="roll_anomaly",
        labels={"timestamp": "Дата", "temperature": "Температура, °C", "roll_anomaly": "Аномалия"},
    )
    fig.add_scatter(
        x=city_df["timestamp"],
        y=city_df["roll_mean"],
        mode="lines",
        name=f"Rolling mean, {window} дней",
    )
    fig.add_scatter(x=city_df["timestamp"], y=city_df["roll_upper"], mode="lines", name="+2σ")
    fig.add_scatter(x=city_df["timestamp"], y=city_df["roll_lower"], mode="lines", name="-2σ")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Долгосрочный тренд: средняя температура по годам")
    yearly = yearly_profile(city_df)
    fig_trend = px.line(
        yearly,
        x="year",
        y="temp_mean",
        markers=True,
        labels={"year": "Год", "temp_mean": "Средняя температура, °C"},
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.subheader("Сезонная статистика")
    view_stats = season_stats.copy()
    view_stats.columns = ["season", "mean", "std"]
    st.dataframe(view_stats, use_container_width=True, hide_index=True)

    st.subheader("Сезонные аномалии")
    fig2 = px.scatter(
        city_df,
        x="timestamp",
        y="temperature",
        color="season_anomaly",
        labels={"timestamp": "Дата", "temperature": "Температура, °C", "season_anomaly": "Аномалия"},
    )
    fig2.add_scatter(x=city_df["timestamp"], y=city_df["season_upper"], mode="lines", name="+2σ")
    fig2.add_scatter(x=city_df["timestamp"], y=city_df["season_lower"], mode="lines", name="-2σ")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Сезонный профиль")
    fig_box = px.box(
        city_df,
        x="season",
        y="temperature",
        labels={"season": "Сезон", "temperature": "Температура, °C"},
    )
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.subheader(f"{city} сейчас")

    if not api_key:
        st.info("Введи API key в меню слева — тогда появится текущая температура.")
        st.stop()

    if st.button("Получить текущую температуру"):
        try:
            data = fetch_current_weather_sync(city, api_key)
        except Exception as e:
            st.error(f"Ошибка сети/запроса: {e}")
            st.stop()

        if str(data.get("cod")) != "200":
            st.error(f"Ошибка API: {data}")
            st.stop()

        current_temp = float(data["main"]["temp"])
        current_season = get_season_by_month(datetime.now().month)

        row = season_stats[season_stats["season"] == current_season]
        if row.empty:
            st.error(f"Нет статистики для сезона {current_season}")
            st.stop()

        season_mean = float(row["season_mean"].iloc[0])
        season_std = float(row["season_std"].iloc[0])
        lower = season_mean - 2 * season_std
        upper = season_mean + 2 * season_std
        is_normal = lower <= current_temp <= upper

        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("Текущая температура", f"{current_temp:.2f} °C")
        c2.metric("Сезон", current_season)
        c3.write(f"Норма для сезона: **{lower:.2f} … {upper:.2f} °C**")

        if is_normal:
            st.success("Температура в пределах нормы для сезона.")
        else:
            st.warning("Температура аномальна для сезона.")
