# Stock-Ranking GNN: Research Framework

This project starts after the stock universe has been supplied. It includes an optional downloader that fetches historical prices and volume for those tickers and caches them locally. The goal is to test whether relationships between stocks help rank likely market outperformers, and whether those rankings produce a diversified portfolio after costs.

## 1. Research question and experiment rules

At each monthly rebalance date, rank the available stocks by their predicted probability of outperforming the market over the next month. Compare a graph neural network (GNN) with:

1. SPY and QQQ buy-and-hold total returns over the same test dates.
2. An equal-weight portfolio of the same eligible stocks.
3. A non-graph model with the same node features, training dates, labels, and portfolio rules.

Write these choices down before looking at test results. Keep the final test period untouched until model and portfolio choices are fixed.

## 2. Input data contract (provided by the data team)

The downloader expects a text file with one ticker per line, or a CSV with a `ticker` or `symbol` column. A comma-separated ticker-only list is also accepted. It adds SPY as a benchmark by default. From the repository root, run:

```bash
python -m pip install -r requirements.txt
python src/download_prices.py tickers.txt --start 2018-01-01
```

Each symbol is saved as `data/prices/<TICKER>.csv`, with `date`, `ticker`, `adj_close`, and `volume` columns. The downloader uses `auto_adjust=True`, so yfinance's `Close` is stored as `adj_close`; no second adjustment should be applied. Existing files are reused without an API request. Use `--refresh` to replace cached files, `--end YYYY-MM-DD` for an exclusive end date, or `--cache-dir PATH` to change the cache location. If you change the requested date range, refresh the cache so the files reflect it.

The cache provides market data only. Add point-in-time sector classifications separately, then combine the per-ticker files into the stock and benchmark tables described below. A supplied list of today's tickers does not establish historical index membership; using it for the full history can introduce survivorship bias, so obtain historical eligible-universe membership where possible.

## Running the starter GNN experiment

Install the project dependencies and make sure `data/prices/` contains the stocks in your universe plus SPY:

```bash
python -m pip install -r requirements.txt
python src/download_prices.py tickers.txt --start 2018-01-01
python -m src.gnn_framework --data-dir data/prices --benchmark SPY
```

The GNN command writes model checkpoints, the saved experiment configuration, out-of-sample monthly returns, portfolio summary metrics, and ranking metrics under `reports/gnn/`. It compares a weighted two-layer graph convolution with an MLP that receives the same node features but no graph edges, equal weighting, and cached SPY/QQQ benchmarks. QQQ is included only when a QQQ CSV exists in the price cache. The default experiment uses a 60/20/20 chronological split; label windows crossing split boundaries are purged. Features and correlations use only data through the close before each simulated rebalance close. The test period is held out from training and validation.

The first version uses 21-session return, 63-session annualized volatility, and recent-versus-prior average-volume change. It creates monthly graphs from positive trailing return correlations, keeps the strongest neighbors, and trains on the binary target of beating SPY over the following 21 sessions. The default transaction-cost assumption is 10 basis points per dollar traded, one-way. Change settings with CLI options such as `--top-k`, `--max-position`, `--max-sector`, `--cost-rate`, and `--edges-per-node`; record all selected settings when interpreting a run.

Sector limits can only be enforced if the price CSVs contain a historical `sector` column. Without sector data, the framework warns and cannot apply sector caps. In that case, add point-in-time sector classifications before interpreting the portfolio as sector-diversified. A ticker list containing today's stocks is also not a historical constituent list: survivorship bias remains unless historical universe eligibility is supplied.

The current working cache is only a small smoke-test universe (AAPL, MSFT, NVDA, and SPY), with no QQQ or sector classifications. It is not enough to treat results as a credible investment backtest. The framework requires at least two stocks to form a graph and at least ten labeled monthly snapshots to perform chronological splits; use the intended 50-100 stock universe and inspect the saved test window and metrics before drawing conclusions. This is a research example, not investment advice.

Use two point-in-time tables. One row per stock/date in the first table, and one row per date in the second:

**`prices.csv`**

| Column | Meaning |
|---|---|
| `date` | Trading date |
| `ticker` | Stable security identifier |
| `adj_close` | Split- and distribution-adjusted closing price |
| `volume` | Trading volume |
| `sector` | Sector classification known on that date |

**`benchmarks.csv`**

| Column | Meaning |
|---|---|
| `date` | Trading date |
| `spy_adj_close` | Adjusted SPY close |
| `qqq_adj_close` | Adjusted QQQ close |

The supplied stock panel should include only securities eligible at each date, rather than applying today's constituents to the full history. Preserve historical ticker/security identity and historical sector classifications where possible. Document missing values, corporate-action treatment, and the universe's inclusion/exclusion rules. The framework assumes prices are adjusted; do not adjust them a second time.

## 3. Prepare the point-in-time panel

1. Parse dates, sort by `ticker, date`, and reject duplicate stock/date rows or nonpositive prices.
2. Check missing trading days, stale prices, missing volume, and stock/benchmark date alignment. Do not silently fill missing prices with future information.
3. Compute each stock's daily simple return from adjusted close:

   `r[i,t] = adj_close[i,t] / adj_close[i,t-1] - 1`

4. Compute SPY's daily return using the same convention. Do not use a date in a feature until its closing data would actually have been available for the simulated trade.

## 4. Make monthly graph snapshots and node features

For each monthly rebalance date `t`, use data through the previous completed close (or consistently assume execution at the next session's open). Create one graph containing the eligible stocks on that date.

**Node features** (initial practical set):

- 21-trading-day trailing return.
- 63-trading-day trailing volatility, annualized consistently.
- Change in recent average daily volume versus the prior 60-trading-day average, e.g. `mean(volume[t-20:t]) / mean(volume[t-80:t-20]) - 1`.
- Optional: point-in-time fundamentals, only if their publication/availability dates are known. Lag them to their actual public availability.

Use trailing windows only. Standardize features using statistics fit on the training period; never fit a scaler on validation/test data. Handle missing features using a documented train-fitted imputation rule, and flag or exclude nodes that lack the minimum history.

**Edges / correlation graph**:

1. Calculate pairwise Pearson correlations of trailing daily stock returns, for example over the most recent 60 sessions.
2. Retain the strongest positive relationships per node (e.g. top 5), or apply a predeclared absolute-correlation threshold. Symmetrize the result.
3. Store the correlation as the edge weight. Decide in advance whether negative correlations are excluded or represented as a separate signed relation; do not silently treat negative weights as positive.
4. Add self-loops in the GNN's normalized adjacency matrix.

The graph is rebuilt from the trailing window at every rebalance date. Neither edges nor node features may use the forward label period.

## 5. Define the prediction target

For stock `i` at rebalance date `t`, define its next-month cumulative return over the next 21 trading sessions:

`R[i,t+1:t+21] = product(1 + r[i,d]) - 1`

Define the market return over those same sessions from SPY, then set:

`y[i,t] = 1 if R[i,t+1:t+21] > R[SPY,t+1:t+21], else 0`

This makes the model predict a binary relative-outperformance event. Exclude training examples whose full 21-session outcome is not yet observable at the training cutoff. Consider reporting continuous excess return as a secondary analysis, but keep the primary target fixed for the initial experiment.

## 6. Split dates chronologically

Split by **rebalance date**, not by randomly shuffling stock rows. Example: earliest 60% of dates for training, next 20% for validation, latest 20% for final test. Tune model settings and portfolio rules on validation only.

Because 21-session labels overlap across adjacent monthly dates, purge training examples whose outcome window overlaps the validation/test boundary, and leave at least a 21-session embargo around each boundary. Fit feature scaling and any learned preprocessing on training dates only. For stronger evidence, add expanding-window walk-forward folds and report the untouched final holdout separately.

## 7. Train comparable models

Train a GNN to output one score/probability per stock in each date's graph. A simple starting architecture is a two-layer weighted graph convolution:

`H = ReLU(A_norm X W1)`  
`logits = A_norm H W2`

where `X` contains node features and `A_norm` is the symmetrically normalized, weighted adjacency with self-loops. Train with binary cross-entropy on training-date nodes; report class balance and use training-only class weights if needed. Fix and record random seeds.

Train the non-graph comparison on exactly the same features and examples, but with no graph/message-passing connections (for example, an MLP). This isolates whether graph information adds value rather than whether one model simply has more data.

Choose hyperparameters, early stopping, and the ranking score on validation dates only. Save the model, scaler, feature names, graph rule, configuration, and training cutoff together.

## 8. Turn predictions into a portfolio

At each monthly rebalance:

1. Score eligible stocks and rank from highest to lowest predicted probability.
2. Select a fixed number or fraction of top-ranked stocks (choose on validation; keep fixed for test).
3. Set equal weights among selected stocks as a simple starting policy.
4. Apply predeclared position and sector limits (for example, maximum 5% per name and 25% per sector), then renormalize remaining weights. If limits make the portfolio infeasible, use a documented deterministic rule rather than relaxing limits silently.
5. Hold until the next rebalance. Calculate turnover from old to new weights and charge costs on traded notional.

Model the cost as `cost_rate * sum(abs(new_weight - old_weight))`; state whether the rate is one-way or round-trip and include commissions, spread, and slippage assumptions as appropriate. Apply costs when trades occur, not as a one-time deduction at the end. Use the same execution timing and cost convention for all strategies.

## 9. Backtest and compare

Run all portfolio strategies on the same unseen dates with matching rebalance schedule and return convention. Include:

- GNN portfolio after costs.
- No-graph model portfolio after costs.
- Equal-weight eligible-stock portfolio after costs.
- SPY and QQQ buy-and-hold total-return benchmarks.

Report at minimum: cumulative return, annualized return, annualized volatility, Sharpe ratio (state the risk-free-rate assumption), maximum drawdown, turnover, estimated transaction costs, and number of holdings. For the stock-ranking task, also report out-of-sample precision/recall or AUC and mean realized excess return of selected stocks. Show results by year and across walk-forward folds; do not rely on accuracy alone or claim success from one favorable period.

## 10. Leakage and robustness checklist

- Every feature and graph at date `t` uses only information available by the stated decision time.
- Labels are forward-looking and are never included in the input.
- Scalers, imputers, feature selection, and model tuning use training/validation data only, as appropriate.
- The final test dates do not influence choices.
- Historical universe membership and sector classifications are point-in-time.
- All strategies use identical dates, eligible names, rebalance timing, and cost assumptions.
- Check results under reasonable alternative correlation windows, graph sparsity, transaction costs, and rebalance timing, while treating these as sensitivity analyses rather than selecting whichever test result looks best.

## Suggested implementation layout

```text
data/                 # supplied input files; keep raw files unchanged
src/
  download_prices.py # downloads supplied tickers and caches adjusted prices/volume
  gnn_framework.py  # point-in-time graphs, training, portfolio, and backtest
  data.py             # schema checks, returns, date alignment
  snapshots.py        # point-in-time features, labels, and graph edges
  models.py           # GNN and non-graph model
  train.py            # chronological splits, fitting, validation
  portfolio.py        # ranking, position/sector limits, turnover and costs
  backtest.py         # monthly simulation and benchmark comparisons
configs/
  experiment.yaml     # windows, split dates, model and portfolio choices
tests/
  test_gnn_framework.py
  test_no_lookahead.py
  test_graph.py
  test_portfolio.py
reports/              # metrics and figures from fixed experiment runs
```

## Recommended run order

1. Validate the supplied tables and produce a data-quality report.
2. Generate a few hand-checked monthly snapshots; verify that feature and edge dates precede each label window.
3. Test graph construction, weight/sector constraints, and transaction-cost arithmetic on tiny synthetic examples.
4. Establish equal-weight and no-graph baselines before tuning the GNN.
5. Select settings on chronological validation data, lock the experiment configuration, then run the test once.
6. Save predictions, holdings, daily/monthly returns, costs, metrics, and configuration so every reported result is reproducible.
