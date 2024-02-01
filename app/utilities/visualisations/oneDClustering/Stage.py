import seaborn as sns


class Stage:
    def getStage(self):
        sns.set(style="whitegrid", font="CMU Serif", font_scale=1.4)
        ax = sns.lineplot(
            # legend=False
        )
        return ax
