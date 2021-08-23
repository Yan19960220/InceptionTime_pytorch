from typing import List

import numpy as np
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d


class Data_Process:
    def __init__(self, name: str) -> None:
        self.type = name
        self.fmin = 10
        self.fmax = 2000
        self.fs = 4096
        self.sample_space = 1.0 / self.fs

    def set_type(self, name: str) -> None:
        self.type = name

    def build_filters(self, input: List, labels: List, threshold: float=0.9) -> (List, List):
        if self.type == 'snr':
            self._snr(input)
        elif self.type == 'filter_snr_label':
            return self._filter_snr_with_labels(input, labels, threshold)
        elif self.type == 'whiten':
            return self._whiten_list(input, labels)
        else:
            return self._interp1d(input, 10)

    def _snr(self, input):
        pass

    def _interp1d(self, input, plot=False):
        NFFT = 4 * self.fs
        psd, freqs = mlab.psd(input, Fs=self.fs, NFFT=NFFT)

        # We will use interpolations of the ASDs computed above for whitening:
        psd = interp1d(freqs, psd)
        # if plot:
        #     plt.figure()
        #     plt.loglog(freqs, np.sqrt(psd), 'r', label='data')
        #     plt.axis([self.fmin, self.fmax, 1e-24, 1e-19])
        #     plt.grid('on')
        #     plt.ylabel('ASD (strain/rtHz)')
        #     plt.xlabel('Freq (Hz)')
        #     plt.legend(loc='upper center')
        #     plt.title('Advanced LIGO/VIRGO data')
        #     plt.savefig('ASDs.png')
        return psd

    def whiten(self, input: np.array) -> np.array:
        window_length = len(input)
        freqs = np.fft.rfftfreq(window_length, self.sample_space)

        input_f = np.fft.rfft(input)

        # calculate the psd of the time series
        interp_psd = self._interp1d(input)

        whiten_input_f = input_f / np.sqrt(interp_psd(freqs))

        whiten_input_t = np.fft.irfft(whiten_input_f, n=window_length)

        return whiten_input_t

    def _whiten_list(self, series: List, labels: List):
        for i in range(len(series)):
            for j in range(len(series[i])):
                series[i][j] = self.whiten(np.array(series[i][j])).tolist()
        return series, labels

    def signaltonoise(self, a: np.array, axis: int = 0, ddof: int = 0) -> np.array:
        if isinstance(a, List):
            a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m / sd)

    def _filter_snr(self, input: List, threshold: float) -> (List, List):
        output_list = []
        index_list = []
        input = np.asanyarray(input)
        if input.ndim == 2:
            for index, ele in np.ndenumerate(input):
                out = self.signaltonoise(ele)
                if out > threshold:
                    output_list.append(ele)
                    index_list.append(index)
            return output_list, index_list
        elif input.ndim == 1:
            l_out = []
            l_length = []
            for i in input:
                l_time_series = []
                for j in i:
                    out = self.signaltonoise(j)
                    if out > threshold:
                        l_time_series.append(j)
                l_out.append(l_time_series)
                l_length.append(l_time_series.__len__())
            return l_out, l_length
        else:
            print(f"Number of dimension is wrong".center(90, '-'))

    def _filter_snr_with_labels(self, input: List, labels: List, threshold: float) -> (List, List):
        l_ele, l_index = self._filter_snr(input, threshold)
        l_out_label = []
        if np.array(l_index).ndim == 1:
            for i in range(len(l_index)):
                l_out_label.append(labels[i][0:l_index[i]])
        else:
            for i in l_index:
                l_out_label.append(labels.get(i))
        return l_ele, l_out_label
