# -*- coding: utf-8 -*-
# @Time: 2024/11/22
# @File: read_file.py
# @Author: fwb
import numpy as np
import os
import struct
import scipy.io as sio


class ReadFile:
    def bin_file_reader(self, filepath):
        """
        :param filepath: NCaltech101/NMNIST dataset file specified by 'filename'
        :return: All x, y, t, p in the file
        """
        f = open(filepath, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()
        raw_data = np.uint32(raw_data)
        all_x = np.squeeze(raw_data[0::5])
        all_y = np.squeeze(raw_data[1::5])
        all_t = np.squeeze(((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5]))
        all_p = np.squeeze((raw_data[2::5] & 128) >> 7)  # bit 7
        # Process time stamp overflow events.
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_t[overflow_index:] = all_t[overflow_index:] + time_increment
        # Everything else is a proper spike.
        td_indices = np.where(all_y != 240)[0]
        x = all_x[td_indices]
        y = all_y[td_indices]
        t = all_t[td_indices]  # timestamp in microseconds
        p = all_p[td_indices]
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]

    def npy_file_reader(self, filepath):
        """
        :param filepath: THU/NCars dataset file specified by 'filename'
        :return: All x, y, t, p in the file
        """
        events = np.load(filepath).astype(np.float32)
        x, y, t, p = events.T
        t_diff = t[1] - t[0]
        if t_diff != 0 and t_diff < 1:
            t = t * 1e6  # timestamp in microseconds
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]

    def npz_file_reader(self, filepath):
        """
        :param filepath: DVSGesture dataset file specified by 'filename'
        :return: All x, y, t, p in the file
        """
        events = np.load(filepath)
        x = events['x']
        y = events['y']
        t = events['t']
        p = events['p']
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]

    def aedat_file_reader(self, filepath):
        """
        :param filepath: PAF dataset file specified by 'filename'
        :return: All x, y, t, p in the file
        """
        startTime = 0
        sizeX = 346
        sizeY = 260
        x0 = 0
        y0 = 0
        x1 = sizeX
        y1 = sizeY

        polmask = int('800', 16)
        xmask = int('003FF000', 16)
        ymask = int('7FC00000', 16)
        typemask = int('80000000', 16)
        typedvs = int('00', 16)
        xshift = 12
        yshift = 22
        polshift = 11
        x = []
        y = []
        ts = []
        pol = []
        numeventsread = 0

        length = 0
        aerdatafh = open(filepath, 'rb')
        k = 0
        p = 0
        statinfo = os.stat(filepath)
        if length == 0:
            length = statinfo.st_size

        lt = aerdatafh.readline()
        while lt and str(lt)[2] == "#":
            p += len(lt)
            k += 1
            lt = aerdatafh.readline()
            continue

        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        while p < length:
            ad, tm = struct.unpack_from('>II', tmp)
            ad = abs(ad)
            if tm >= startTime:
                if (ad & typemask) == typedvs:
                    xo = sizeX - 1 - float((ad & xmask) >> xshift)
                    yo = float((ad & ymask) >> yshift)
                    polo = 1 - float((ad & polmask) >> polshift)
                    if x0 <= xo < x1 and y0 <= yo < y1:
                        x.append(xo)
                        y.append(yo)
                        pol.append(polo)
                        ts.append(tm)
            aerdatafh.seek(p)
            tmp = aerdatafh.read(8)
            p += 8
            numeventsread += 1
        x = np.array(x)
        y = np.array(y)
        t = np.array(ts)
        p = np.array(pol)
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]

    def txt_file_reader(self, filepath):
        """
        :param filepath: NeuroHAR dataset file specified by 'filename'
        :return: All x, y, t, p in the file
        """
        events = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                values = list(map(float, line.strip().split()))
                events.append(values)
        events_arr = np.array(events)
        t, x, y, p = events_arr.T
        t = t * 1e6
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]

    def mat_file_reader(self, filepath):
        """
        :param filepath: UCF101-DVS dataset file specified by 'filename'
        :return: All x, y, t, p in the file
        """
        events = sio.loadmat(filepath)
        x = np.squeeze(events['x'])
        y = np.squeeze(events['y'])
        t = np.squeeze(events['ts'])
        p = np.squeeze(events['pol'])
        p = np.where(p == 0, -1, p.astype(int))
        return [x, y, t, p]
    