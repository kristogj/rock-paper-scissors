import pandas as pd
import seaborn as sns


def generate_lineplot(data, y_title="Loss", data_type='Dataset', type_names=('Training', 'Validation')):
    # preprocess data
    columns = ['Epoch', data_type, y_title]
    df = pd.DataFrame(columns=columns)

    for i in range(len(data)):
        for j in range(len(data[i])):
            df.loc[len(df)] = [j + 1, type_names[i], data[i][j]]

    sns.set(style="darkgrid")
    plt = sns.lineplot(data=df, x='Epoch', y=y_title, style=data_type, hue=data_type)
    plt.figure.savefig("./graphs/loss.png")
    plt.figure.clf()
