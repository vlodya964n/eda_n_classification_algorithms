import pandas as pd

DATA_PATH = 'data/abalone.data'

COLUMNS_NAMES = [
    'Sex', 'Length', 'Diameter',
    'Height', 'Whole_weight', 'Shucked_weight',
    'Viscera_weight', 'Shell_weight', 'Rings'
    ]

def data_analisys():

    df = pd.read_csv(DATA_PATH, names=COLUMNS_NAMES)

    df.drop(df[(df['Height'] > 0.25)].index, inplace=True)
    df.drop(df[(df['Height'] == 0)].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    df.to_csv('data/preprocessed_data.csv',index=False)

if __name__ == '__main__':
    data_analisys()
