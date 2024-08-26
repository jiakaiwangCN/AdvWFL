import pandas as pd


def calAverage(file_name):
    df = pd.read_csv(file_name)
    mean = df.iloc[:, 4].mean()
    return mean


def calMax(file_name):
    df = pd.read_csv(file_name)
    mean = df.iloc[:, 4].max()
    return mean


if __name__ == '__main__':
    df = pd.DataFrame(columns=['value', 'distance', 'layer', 'delta'])
    base = 0
    for distance in range(1, 11):
        for layer in range(0, 10):
            value = 0
            for times in range(1, 4):
                file_name = '../data covered/' + str(times) + '/' + str(layer) + '-' + str(distance) + '.csv'
                value += calAverage(file_name)
            if layer == 0:
                base = value / 3
            df = df.append({'value': value / 3, 'distance': distance, 'layer': layer, 'delta': value / 3 - base},
                           ignore_index=True)
    df.to_csv('merged.csv', index=False)


