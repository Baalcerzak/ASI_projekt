import json
import os
from typing import Dict, Any

import requests
import streamlit as st

st.set_page_config(page_title="Car Price Prediction", layout="centered")


def get_api_base() -> str:
    return os.getenv("STREAMLIT_API_URL", os.getenv("API_URL", "http://localhost:8000")).rstrip("/")


def api_get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{get_api_base()}{path}"
    r = requests.get(url, params=params or {}, timeout=15)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{get_api_base()}{path}"
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def car_form(prefix: str = "") -> Dict[str, Any]:
    st.subheader("Parametry auta")
    col1, col2 = st.columns(2)

    with col1:
        manufacturer = st.text_input("Manufacturer", value="TOYOTA", key=f"{prefix}manufacturer")
        model = st.text_input("Model", value="Camry", key=f"{prefix}model")
        prod_year = st.number_input("Prod. year", min_value=1990, max_value=2026, value=2018, step=1, key=f"{prefix}prod_year")
        category = st.text_input("Category", value="Sedan", key=f"{prefix}category")
        leather_interior = st.selectbox("Leather interior", ["Yes", "No"], index=0, key=f"{prefix}leather")
        fuel_type = st.text_input("Fuel type", value="Petrol", key=f"{prefix}fuel")
        engine_volume = st.number_input("Engine volume (L)", min_value=0.1, max_value=10.0, value=2.5, step=0.1, key=f"{prefix}engine")
        mileage = st.number_input("Mileage (km)", min_value=0.0, value=50000.0, step=1000.0, key=f"{prefix}mileage")

    with col2:
        cylinders = st.number_input("Cylinders", min_value=1.0, max_value=16.0, value=4.0, step=1.0, key=f"{prefix}cyl")
        gear_box_type = st.text_input("Gear box type", value="Automatic", key=f"{prefix}gear")
        drive_wheels = st.selectbox("Drive wheels", ["Front", "Rear", "4x4"], index=0, key=f"{prefix}drive")
        doors = st.number_input("Doors", min_value=2, max_value=6, value=4, step=1, key=f"{prefix}doors")
        wheel = st.selectbox("Wheel", ["Left wheel", "Right wheel"], index=0, key=f"{prefix}wheel")
        color = st.text_input("Color", value="Black", key=f"{prefix}color")
        airbags = st.number_input("Airbags", min_value=0, max_value=20, value=8, step=1, key=f"{prefix}airbags")
        levy = st.number_input("Levy (USD)", min_value=0.0, value=0.0, step=50.0, key=f"{prefix}levy")
        turbo = st.selectbox("Turbo", [0, 1], index=0, key=f"{prefix}turbo")

    return {
        "manufacturer": manufacturer,
        "model": model,
        "prod_year": int(prod_year),
        "category": category,
        "leather_interior": leather_interior,
        "fuel_type": fuel_type,
        "engine_volume": float(engine_volume),
        "mileage": float(mileage),
        "cylinders": float(cylinders),
        "gear_box_type": gear_box_type,
        "drive_wheels": drive_wheels,
        "doors": int(doors),
        "wheel": wheel,
        "color": color,
        "airbags": int(airbags),
        "levy": float(levy),
        "turbo": int(turbo),
    }


def page_predict():
    st.header("Predykcja ceny samochodu")

    with st.expander("Status backendu"):
        try:
            h = api_get("/health")
            st.success(f"API OK • Model loaded: {h.get('model_loaded')} • {h.get('timestamp')}")
        except Exception as e:
            st.error(f"Brak połączenia z API ({get_api_base()}) – {e}")

    car = car_form()

    if st.button("Wyceń samochód", type="primary"):
        try:
            payload = {"cars": [car]}
            res = api_post("/predict", payload)

            st.subheader("Wynik")
            price = res.get("prices_usd", [None])[0]
            latency = res.get("latency_ms")
            model_used = res.get("model_used")
            drift = res.get("drift_warnings", [])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Cena [USD]", f"{price:,.0f}" if price is not None else "–")
            with c2:
                st.metric("Latency", f"{latency} ms" if latency is not None else "–")
            with c3:
                st.metric("Model", model_used or "–")

            if drift:
                st.warning("Wykryto potencjalny drift danych:")
                for w in drift:
                    st.write("• ", w)
            else:
                st.info("Brak ostrzeżeń o drifcie dla tego wejścia.")

            with st.expander("Surowa odpowiedź API"):
                st.code(json.dumps(res, indent=2), language="json")
        except requests.HTTPError as e:
            st.error(f"Błąd API: {e.response.text}")
        except Exception as e:
            st.error(f"Nieoczekiwany błąd: {e}")


def page_history():
    st.header("Historia predykcji")
    limit = st.slider("Ile ostatnich wpisów pokazać?", min_value=5, max_value=100, value=20, step=5)

    try:
        data = api_get("/predictions/history", params={"limit": limit})
        preds = data.get("predictions", [])
        st.write(f"Łącznie wpisów w logu: {data.get('total', 0)}")
        if not preds:
            st.info("Brak danych w logu.")
            return

        for entry in preds:
            with st.expander(f"{entry.get('timestamp', '')}"):
                st.code(json.dumps(entry, indent=2), language="json")
    except Exception as e:
        st.error(f"Nie udało się pobrać historii: {e}")


def page_model():
    st.header("Informacje o modelu i drift")
    cols = st.columns(2)
    try:
        with cols[0]:
            info = api_get("/model/info")
            st.subheader("Model info")
            st.json(info)
        with cols[1]:
            drift = api_get("/drift/report")
            st.subheader("Drift report")
            st.json(drift)
    except Exception as e:
        st.error(f"Błąd pobierania informacji: {e}")


def main():
    with st.sidebar:
        st.markdown("### Ustawienia")
        st.text_input("API URL", value=get_api_base(), key="api_url_box", help="Adres serwisu FastAPI")
        st.caption("Aby trwale zmienić, ustaw zmienną środowiskową STREAMLIT_API_URL.")
        st.divider()
        st.markdown("Made with Streamlit")

    tabs = st.tabs(["Predykcja", "Historia", "Model & Drift"])
    with tabs[0]:
        page_predict()
    with tabs[1]:
        page_history()
    with tabs[2]:
        page_model()


if __name__ == "__main__":
    main()
