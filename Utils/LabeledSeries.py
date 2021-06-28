from torch.utils.data import Dataset


class LabeledSeries(Dataset):
    def __init__(self, series, labels):
        super(LabeledSeries, self).__init__()

        assert len(series) == len(labels)

        self.series = series
        self.labels = labels

    def __len__(self):
        return self.series.shape[0]

    def __getitem__(self, indices):
        return self.series[indices], self.labels[indices]