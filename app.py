from PortBack import run_optimizer, monte_carlo_simulation
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf

#OPTIMIZER IS ONLY GRABBING DATA TO THE EARLIEST DATE POSSIBLE, (i.e CAVA is young so only pulls days since IPO, this is an issue if other stocks have 5 years of data)

st.title("PORTFOLIO TERMINAL")
st.caption("INTSTITUTIONAL FRAMEWORK. RETAIL ACCESS.")

# Step 1: input tickers
initial_investment = st.number_input("INTIAL INVESTMENT AMOUNT ($):", min_value=100, max_value=100000000, value=10000)

tickers_input = st.text_input("ENTER TICKERS (COMMA SEPARATED):", "AAPL, MSFT, GOOGL, SPY, QQQ")
TICKERS = [t.strip() for t in tickers_input.split(",")]
valid_tickers = []
invalid_tickers = []
is_valid_input = True

for ticker in TICKERS:
    try:
        data = yf.Ticker(ticker).history(period="5d")

        if data.empty:
            invalid_tickers.append(ticker)
            is_valid_input = False
        else:
            valid_tickers.append(ticker)

    except:
        invalid_tickers.append(ticker)
        is_valid_input = False

if not is_valid_input:
    st.error(f"INVALID TICKERS: {', '.join(invalid_tickers)}. PLEASE CORRECT BEFORE PROCEEDING.")
    st.stop()

# Step 2: create slider for each ticker
weight_constraints = {}
for t in TICKERS:
    weight_constraints[t] = st.slider(f"WEIGHT CONSTRAINTS FOR {t}",0.0,1.0,(0.0,0.1), key=f"slider_{t}")

total_weight = sum(max_w for min_w, max_w in weight_constraints.values())
if total_weight < 1:
    st.error(f"INCREASE MAX WEIGHTS TO AT LEAST 100%. CURRENTLY: {total_weight*100:.1f}%.")
    st.stop()

min_sum = sum([min_w for (min_w, _) in weight_constraints.values()])

if min_sum > 1:
    st.error(f"DECREASE MIN WEIGHTS TO AT MOST 100%. CURRENTLY: {min_sum*100:.1f}%.")
    st.stop()


#Ensure Alignment
aligned_tickers = sorted(weight_constraints.items())
TICKERS,weight_constraints = zip(*aligned_tickers)
weight_constraints = dict(zip(TICKERS, weight_constraints))
constraints_df = pd.DataFrame.from_dict(weight_constraints, orient='index', columns = ["Min Weight", "Max Weight"])

#Display Constraints
st.dataframe(constraints_df.style.format({"Min Weight": "{:.2%}", "Max Weight": "{:.2%}"}))


st.markdown("PRESS OPTIMIZE TO RUN PORTFOLIO OPTIMIZER WITH THE ABOVE CONSTRAINTS")

if st.button("OPTIMIZE"):
    results = run_optimizer(TICKERS, weight_constraints=weight_constraints)
    col1, col2, col3 = st.columns(3)
    col1.metric("EXPECTED RETURN", f"{results['expected_return']:.2%}")
    col2.metric("VOLATILITY", f"{results['volatility']:.2%}")
    col3.metric("SHARPE RATIO", f"{results['sharpe_ratio']:.2f}")

    weights = pd.Series(results["weights"])
    weights = weights[weights > 0].sort_values(ascending=False)
    dollar_values = initial_investment * weights
    
    fig = px.pie(
    names=weights.index,
    values=weights.values,
    title="Portfolio Weights Distribution",
    hover_data={"Dollar Value ($)": dollar_values.values},
    )
    st.plotly_chart(fig, use_container_width=True)
    #Creating a historical performance chart

    hist_returns = results["prices"].pct_change().dropna()

    portfolio_returns = hist_returns[weights.index].dot(weights)
    spy_returns = hist_returns["SPY"]

    perf_df = pd.DataFrame({
    "PORTFOLIO": portfolio_returns,
    "SPY": spy_returns
    })
    perf_df = (1 + perf_df).cumprod() - 1
    perf_df = perf_df.reset_index().melt(id_vars="Date", var_name="Series", value_name="CUMULATIVE RETURN")
    perf_fig = px.line(perf_df, x="Date", y="CUMULATIVE RETURN", color="Series", title="HISTORICAL PERFORMANCE VS. BENCHMARK")
    st.plotly_chart(perf_fig, use_container_width=True)

    # Correlation matrix using only portfolio assets
    corr_matrix = hist_returns[weights.index].corr()

    corr_fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlation Matrix"
    )

    corr_fig.update_layout(
    template="plotly_dark",
    title_x=0
    )

    st.plotly_chart(corr_fig, use_container_width=True)

    mc_paths = monte_carlo_simulation(results["prices"], results["weights"], initial_investment)

    p5 = np.percentile(mc_paths, 5, axis=1)
    p25 = np.percentile(mc_paths, 25, axis=1)
    p50 = np.percentile(mc_paths, 50, axis=1)
    p75 = np.percentile(mc_paths, 75, axis=1)
    p95 = np.percentile(mc_paths, 95, axis=1)


    percentile_df = pd.DataFrame({ 
        "Day": range(len(p5)),
        "5th Percentile": p5, 
        "25th Percentile": p25, 
        "50th Percentile": p50, 
        "75th Percentile": p75, 
        "95th Percentile": p95
    })
    percentile_df = percentile_df.melt(id_vars="Day", var_name="Percentile", value_name="Portfolio Value")


    mc_df = pd.DataFrame(mc_paths[:,:100])
    mc_df["Day"] = mc_df.index
    mc_df = mc_df.melt(id_vars="Day", var_name="Simulation", value_name="Portfolio Value")

    mc_fig = px.line(mc_df, x="Day", y="Portfolio Value", color="Simulation", title="MONTE CARLO SIMULATION OF PORTFOLIO VALUE - ALL PATHS")

    mc_fig.update_layout(showlegend=False, template="plotly_dark", title_x=0)

    st.plotly_chart(mc_fig, use_container_width=True)
    
    percentile_fig = px.line(percentile_df, x="Day", y="Portfolio Value", color="Percentile", title="MONTE CARLO PERCENTILES")

    pl_p5 =p5[-1] - initial_investment
    pl_p50 =p50[-1] - initial_investment
    pl_p95 =p95[-1] - initial_investment
    st.plotly_chart(percentile_fig, use_container_width=True)
    col1, col2, col3 = st.columns(3)

    col1.metric("5% Worst Case", f"{((p5[-1]-initial_investment)/initial_investment)*100:.2f}%",round(pl_p5, 2))
    col2.metric("Median Outcome", f"{((p50[-1]-initial_investment)/initial_investment)*100:.2f}%",round(pl_p50, 2))
    col3.metric("95% Best Case", f"{((p95[-1]-initial_investment)/initial_investment)*100:.2f}%",round(pl_p95, 2))

