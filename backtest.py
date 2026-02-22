def backtest(df):
    df["signal"] = (df["ema20"] > df["ema50"]).astype(int)
    df["ret"] = df["Close"].pct_change()
    df["strategy"] = df["signal"].shift(1) * df["ret"]
    df["equity"] = (1 + df["strategy"]).cumprod()
    return df[["equity"]]

def walk_forward(df, train=400, test=150):
    splits = []
    for i in range(train, len(df) - test, test):
        splits.append((df.iloc[:i], df.iloc[i:i+test]))
    return splits
