import numpy as np
import pandas as pd


def read(path):
    csv = pd.read_csv(path)
    return csv


def main():
    csv = read("lr_0_01000.csv")
    print(csv)


if __name__ == "__main__":
    main()
