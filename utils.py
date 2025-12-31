from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import aiohttp
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_temperature_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def get_season_by_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def city_slice(df: pd.DataFrame, city: str) -> pd.DataFrame:
    return df[df["city"] == city].sort_values("timestamp").reset_index(drop=True)


def add_rolling(df_city: pd.DataFrame, window: int) -> pd.DataFrame:
    d = df_city.copy()
    d["roll_mean"] = d["temperature"].rolling(window=window, min_periods=window).mean()
    d["roll_std"] = d["temperature"].rolling(window=window, min_periods=window).std()
    d["roll_upper"] = d["roll_mean"] + 2 * d["roll_std"]
    d["roll_lower"] = d["roll_mean"] - 2 * d["roll_std"]
    d["roll_anomaly"] = (
        d["roll_mean"].notna()
        & ((d["temperature"] > d["roll_upper"]) | (d["temperature"] < d["roll_lower"]))
    )
    return d


def season_stats_for_city(df_city: pd.DataFrame) -> pd.DataFrame:
    return (
        df_city.groupby("season")["temperature"]
        .agg(season_mean="mean", season_std="std")
        .reset_index()
    )


def add_season_bounds(df_city: pd.DataFrame, season_stats: pd.DataFrame) -> pd.DataFrame:
    d = df_city.merge(season_stats, on="season", how="left")
    d["season_upper"] = d["season_mean"] + 2 * d["season_std"]
    d["season_lower"] = d["season_mean"] - 2 * d["season_std"]
    d["season_anomaly"] = (
        (d["temperature"] > d["season_upper"]) | (d["temperature"] < d["season_lower"])
    )
    return d


def yearly_profile(df_city: pd.DataFrame) -> pd.DataFrame:
    d = df_city.copy()
    d["year"] = d["timestamp"].dt.year
    return d.groupby("year")["temperature"].mean().reset_index(name="temp_mean")

def fetch_current_weather_sync(city: str, api_key: str, timeout: int = 15) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    r = requests.get(url, params=params, timeout=timeout)
    return r.json()


async def fetch_current_weather_async(city: str, api_key: str, timeout: int = 15) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=timeout) as resp:
            return await resp.json()


async def fetch_many_cities_async(cities: List[str], api_key: str, timeout: int = 15) -> List[dict]:
    url = "https://api.openweathermap.org/data/2.5/weather"
    async with aiohttp.ClientSession() as session:
        tasks = []
        for c in cities:
            params = {"q": c, "appid": api_key, "units": "metric"}
            tasks.append(session.get(url, params=params, timeout=timeout))
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for resp in responses:
            if isinstance(resp, Exception):
                results.append({"cod": "error", "message": str(resp)})
                continue
            async with resp:
                results.append(await resp.json())
        return results


def analyze_city_block(city: str, df_city: pd.DataFrame, window: int = 30) -> dict:
    d = df_city.sort_values("timestamp").copy()

    d["roll_mean"] = d["temperature"].rolling(window=window, min_periods=window).mean()
    d["roll_std"] = d["temperature"].rolling(window=window, min_periods=window).std()
    upper = d["roll_mean"] + 2 * d["roll_std"]
    lower = d["roll_mean"] - 2 * d["roll_std"]
    roll_anoms = (d["roll_mean"].notna() & ((d["temperature"] > upper) | (d["temperature"] < lower))).sum()

    stats = d.groupby("season")["temperature"].agg(["mean", "std"])
    d = d.join(stats, on="season", rsuffix="_season")
    season_upper = d["mean"] + 2 * d["std"]
    season_lower = d["mean"] - 2 * d["std"]
    season_anoms = ((d["temperature"] > season_upper) | (d["temperature"] < season_lower)).sum()

    return {
        "city": city,
        "n_days": int(len(d)),
        "period_start": d["timestamp"].min().date(),
        "period_end": d["timestamp"].max().date(),
        "rolling_anomalies": int(roll_anoms),
        "season_anomalies": int(season_anoms),
    }


def benchmark_historical(df: pd.DataFrame, cities: List[str], window: int, workers: int) -> dict:
    import time

    city_groups = {
        c: df[df["city"] == c][["city", "timestamp", "temperature", "season"]].copy()
        for c in cities
    }

    t0 = time.perf_counter()
    seq_results = []
    for c, dcity in city_groups.items():
        seq_results.append(analyze_city_block(c, dcity, window=window))
    t_seq = time.perf_counter() - t0

    t1 = time.perf_counter()
    par_results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(analyze_city_block, c, dcity, window) for c, dcity in city_groups.items()]
        for f in as_completed(futures):
            par_results.append(f.result())
    t_par = time.perf_counter() - t1

    return {
        "t_seq": t_seq,
        "t_par": t_par,
        "speedup": (t_seq / t_par) if t_par > 0 else None,
        "results_df": pd.DataFrame(par_results).sort_values("city").reset_index(drop=True),
    }
