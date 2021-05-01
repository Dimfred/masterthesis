import numpy as np
import pandas as pd


def read(path):
    csv = pd.read_csv(path)
    return csv


def to_numpy(df):
    filtered = None
    for col in range(1, 7):
        current_col = df.iloc[:, col]
        if "DIV" in str(current_col[0]):
            continue

        current_col = current_col.to_numpy().reshape(-1, 1)
        if filtered is None:
            filtered = current_col
        else:
            filtered = np.append(filtered, current_col, axis=1)

    return filtered


def calc(df):
    arr = to_numpy(df)
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)

    res = [[name_, mean_, std_] for name_, mean_, std_ in zip(df.iloc[:, 0], mean, std)]
    df = pd.DataFrame(res, columns=["Name", "Mean", "Std"])
    return df


def main():

    store = []
    for lr in (
        "0_01000",
        "0_00500",
        "0_00250",
        "0_00100",
        "0_00050",
        "0_00025",
        "0_00001",
    ):
        df = read(f"lr_{lr}.csv")
        df = calc(df)

        print(f"LR = {lr}")
        print(df)

        m_mean_std = df.iloc[-1, 1:]
        store.append([lr, *m_mean_std])

    summary = pd.DataFrame(store, columns=["LR", "MEAN", "STD"])
    print(summary)


if __name__ == "__main__":
    main()
