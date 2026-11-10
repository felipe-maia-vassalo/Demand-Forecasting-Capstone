"""Translate presentation text. Stored inputs and model data use stable keys."""
import json
from pathlib import Path

import streamlit as st

with Path(__file__).with_name("en.json").open(encoding="utf-8") as catalog:
    ENGLISH = json.load(catalog)


def english():
    return st.session_state.get("language", "English") == "English"


def tr(portuguese, *values):
    text = ENGLISH.get(portuguese, portuguese) if english() else portuguese
    return text.format(*values) if values else text


def month_label(label):
    month, separator, year = str(label).partition("/")
    return tr(month) + separator + year
