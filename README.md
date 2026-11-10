# Demand Forecasting and Production Planning

How many chairs should a factory make each month when school orders fluctuate? This bilingual Streamlit app, developed for a Production Engineering capstone at PUC-Rio, compares demand forecasts and turns the selected forecast into a monthly **master production schedule (MPS)**.

**[Open the app in English](https://felipe-demand-forecasting.streamlit.app/?lang=en)** · **[Abrir em português](https://felipe-demand-forecasting.streamlit.app/)**

## Try the classroom-chair example

1. Open **Sample time series** in the sidebar and select **Load sample series**. The sample contains 72 months of synthetic classroom-chair demand, from January 2020 through December 2025.
2. On **Time Series**, explore the monthly demand chart and seasonal pattern. Then open **Forecast**, keep the six-month horizon, and select **Run forecast**.
3. Review the selected model and its error metrics. Select **Save forecast for production planning**, open **MPS Inputs**, and save the planning assumptions. Open **MPS** to see when production should start and how much inventory is expected to remain each month.
4. To explore a planning decision, change the lot size under **MPS Inputs**, save it, and revisit **MPS** and **Conclusion**. Compare how the change affects production timing, inventory, and cost.

You can also test the Excel upload using the same data. [Download the sample workbook](examples/synthetic_classroom_chair_demand.xlsx?raw=1)

## How the forecast feeds the plan

| Stage | What the app does |
|---|---|
| Explore demand | Plots trend and seasonality, flags missing months and outliers, and provides diagnostics for intermittency, stationarity, and autocorrelation. |
| Compare forecasts | Tests Croston, SBA, and TSB alongside random forest and seasonal ARIMA models. Forecast horizons are 6, 8, or 12 months. |
| Select a model | Ranks models by mean absolute error (MAE), using root mean squared error (RMSE) as the first tiebreaker. MAE is measured in the product’s own units. For example, an MAE of 17 chairs means the forecasts differed from actual demand by 17 chairs on average over the evaluation period. |
| Schedule production | Combines the forecast with firm orders, opening inventory, safety stock, lot size, and lead time to calculate when to start production and when finished units should be available. The dashboard shows forecast bias, availability, and production costs. |

The built-in sample uses a smaller parameter search so reviewers can run it interactively. To try your own data, upload an Excel workbook with one date column and one numeric quantity column named after the product. A history of at least 50 observations is recommended.

## Run locally

From the repository root, with a Python virtual environment active, run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run Menu.py
```

`Menu.py` defines navigation. The `pages/` folder contains the seven workflow pages, `core/pipeline.py` compares forecasting models, and `core/mps.py` calculates the production schedule.

## Project credits

**Felipe Maia Vassalo and Vitor Castro** developed the original capstone at **PUC-Rio** under the supervision of **Rodrigo Caiado**. [View the original capstone repository](https://github.com/Castro0401/Artefato_TCC)
