import pandas as pd

DATASET_PATH = './Data/dataset.csv'


if __name__ == '__main__':
    df = pd.read_csv(DATASET_PATH)
    df.head()
