import numpy as np
import pandas as pd


class EA():
    def __init__(
        self,
        threshold: int
    ) -> None:
        self.name = []
        self.values = []
        self.threshold = threshold

    def run(
        self,
        data: np.array,
        df: pd.DataFrame
    ) -> None:
        names = []
        values = []

        for i in range(0, data.shape[1]):
            hist, _ = np.histogram(data[:, i], bins='auto', density=True)
            h_x = -hist * np.log(hist)
            h_x = np.where(np.isnan(h_x), 0, h_x)

            names.append(df.columns[i])
            values.append(np.sum(h_x))

        d = {'Feature': names, 'Entropy': values}
        self.df_entropy = pd.DataFrame(data=d)

        self.df_entropy.sort_values(
            by=['Entropy'],
            ascending=False,
            inplace=True
            )

        df_reduced = df.drop(
            self.df_entropy[
                self.df_entropy['Entropy'] < self.threshold
                ]['Feature'].to_list(),
            axis=1
        )

        return df_reduced

    def invoke(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        try:
            df_reduced = df.drop(
                self.df_entropy[
                    self.df_entropy['Entropy'] < self.threshold
                    ]['Feature'].to_list(),
                axis=1
            )
            return df_reduced
        except Exception as e:
            print(e.args)
            return None
