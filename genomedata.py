import os
import numpy as np

class GenomeData:
    def __init__(self, data_file):
        self.data_file = data_file
        self.bin_size = np.empty(0)
        self.barcodes = np.empty(0)
        self.data_chr_all = np.empty(0)
        self.data_bin_all = np.empty(0)
        self.data_gc_all = np.empty(0)
        self.data_map_all = np.empty(0)
        self.data_rc_all = np.empty(0)
        self.data_lrc_all = np.empty(0)

    def load_data(self):
        with open(self.data_file) as f:
            line = f.readline().rstrip()
            fields = line.split('=')
            if len(fields) != 2 or fields[0].rstrip() != '#bin size':
                print('Malformed input file ', self.data_file)
                return False
            bin_size = int(fields[1].rstrip())
            line = f.readline().rstrip()
            fields = line.split(',')
            barcodes = fields[3:]
            data_chr_all = []
            data_gc_all = []
            data_map_all = []
            data_rc_all = []
            for i, line in enumerate(f.readlines()):
                line = line.rstrip('\n')
                fields = line.split(',')
                if fields[0].find('chr') != -1:
                    fields[0] = fields[0][3:]
                data_chr_all.append(int(fields[0]))
                data_gc_all.append(float(fields[1]))
                data_map_all.append(float(fields[2]))
                data_rc_all.append([])
                for j in fields[3:]:
                    data_rc_all[i].append(float(j))

        data_bin_all = np.array(range(1, len(data_chr_all) + 1))
        chroms = np.unique(data_chr_all)
        for i in range(len(chroms)):
            tv = data_chr_all == chroms[i]
            data_bin_all[tv] = np.array(range(1, sum(tv) + 1))
        print('Total {} cells and {} bins are '
              'loaded from file {}.'.format(len(barcodes), len(data_chr_all), self.data_file))

        if len(data_chr_all) < 2000:
            print('Warning: the number of bins loaded is too small, check the format of data file and if data are '
                  'completely loaded!')
        self.bin_size = np.array(bin_size)
        self.barcodes = np.array(barcodes)
        self.data_chr_all = np.array(data_chr_all)
        self.data_bin_all = np.array(data_bin_all)
        self.data_gc_all = np.array(data_gc_all)
        self.data_map_all = np.array(data_map_all)
        self.data_rc_all = np.array(data_rc_all)

    def preprocess_data(self):
        # only use autosome
        chroms = np.unique(self.data_chr_all)
        chroms = set(chroms) & set(range(1, 23))
        chroms = list(chroms)
        tv1 = [1 if self.data_chr_all[i] in chroms else 0 for i in range(len(self.data_chr_all))]
        tv2 = np.logical_and(self.data_gc_all > 0.1, self.data_gc_all < 0.9)
        tv = np.logical_and(tv1, tv2)
        self.filter_data(tv)
        print(self.data_rc_all.shape)

        # normalize for library size
        m = np.median(np.mean(self.data_rc_all, axis=0))
        for i in range(self.data_rc_all.shape[1]):
            self.data_rc_all[:, i] = m * self.data_rc_all[:, i] / (np.mean(self.data_rc_all[:, i]) + 1e-10)

        # filter bins by read counts
        tmp = np.mean(self.data_rc_all, axis=1)
        low_p = np.percentile(tmp, 1)
        high_p = np.percentile(tmp, 99)
        tv = np.logical_and(tmp > low_p, tmp < high_p)
        self.filter_data(tv)
        print(self.data_rc_all.shape)

        # filter cells by gini_coef
        num_bin, num_cell = self.data_rc_all.shape
        tmp = np.transpose(self.data_rc_all)
        m = int(500000/self.bin_size)
        if m > 0:
            seq_len = int(num_bin/m)
            tmp = tmp[:, 0:seq_len*m]
            tmp = tmp.reshape(num_cell, seq_len, -1).sum(2)
        x = np.array(range(1, tmp.shape[1] + 1)) / tmp.shape[1]
        gini_coefs = np.zeros(num_cell)
        for i in range(num_cell):
            y = np.cumsum(np.sort(tmp[i, :]))
            y = y / y[-1]
            gini_coefs[i] = np.sum((x[1:] - y[1:] + x[0:-1] - y[0:-1]) * 0.5 / tmp.shape[1]) / 0.5

        tv = gini_coefs < 0.35
        gini_coefs = gini_coefs[tv]
        self.barcodes = self.barcodes[tv]
        self.data_rc_all = self.data_rc_all[:, tv]
        print(self.data_rc_all.shape)

        self.data_rc_all = np.float32(np.transpose(self.data_rc_all))

    def filter_data(self, tv):
        self.data_chr_all = self.data_chr_all[tv]
        self.data_bin_all = self.data_bin_all[tv]
        self.data_gc_all = self.data_gc_all[tv]
        self.data_map_all = self.data_map_all[tv]
        self.data_rc_all = self.data_rc_all[tv,]
        if self.data_lrc_all.shape[0] > 0:
            self.data_lrc_all = self.data_lrc_all[tv,]
