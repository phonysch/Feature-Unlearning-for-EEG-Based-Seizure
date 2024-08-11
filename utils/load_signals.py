import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample
from myio.save_load import save_hickle_file, load_hickle_file, save_pickle_file, load_pickle_file
from utils.group_seizure_Kaggle2014Pred import group_seizure



def load_signals_Kaggle2014Pred(data_dir, target, data_type):  # 数据路径， 患者名， 数据类型（发作期或发作间期）
    print('load_signals_Kaggle2014Pred for Patient', target)

    dir1 = os.path.join(data_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        if i < 10:
            nstr = '000%d' % i
        elif i < 100:
            nstr = '00%d' % i
        elif i < 1000:
            nstr = '0%d' % i
        else:
            nstr = '%d' % i

        filename = '%s/%s_%s_segment_%s.mat' % (dir1, target, data_type, nstr)
        if os.path.exists(filename):
            data = scipy.io.loadmat(filename)  # https://blog.csdn.net/weixin_36049506/article/details/91850299
            # discard preictal segments from 66 to 35 min prior to seizure
            if data_type == 'preictal':
                for skey in data.keys():
                    if "_segment_" in skey.lower():
                        mykey = skey
                sequence = data[mykey][0][0][4][0][0]
                if sequence <= 3:
                    print('Skipping %s....' % filename)
                    continue
            yield data
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True

def load_signals_FB(data_dir, target, data_type):
    print('load_signals_FB for Patient', target)

    def strcv(num):
        if num < 10:
            return '000' + str(num)
        elif num < 100:
            return '00' + str(num)
        elif num < 1000:
            return '0' + str(num)
        elif num < 10000:
            return str(num)

    # sch (原本么有)
    strtrg = ''
    #
    if int(target) < 10:
        strtrg = '00' + str(target)
    elif int(target) < 100:
        strtrg = '0' + str(target)

    if data_type == 'ictal':

        sop = 30 * 60 * 256
        target_ = 'pat%sIktal' % strtrg
        dir2 = os.path.join(data_dir, target_)
        df_sz = pd.read_csv(
            os.path.join(data_dir, 'seizure.csv'), index_col=None, header=0)
        df_sz = df_sz[df_sz.patient == int(target)]
        df_sz.reset_index(inplace=True, drop=True)

        print(df_sz)
        print('Patient %s has %d seizures' % (target, df_sz.shape[0]))
        for i in range(df_sz.shape[0]):
            data = []
            filename = df_sz.iloc[i]['filename']
            st = df_sz.iloc[i]['start'] - 5*60*256
            print('Seizure %s starts at %d' % (filename, st))
            for ch in range(1, 7):
                filename2 = '%s/%s_%d.asc' % (dir2, filename, ch)
                if os.path.exists(filename2):
                    tmp = np.loadtxt(filename2)
                    seq = int(filename[-4:])
                    prevfile = '%s/%s%s_%d.asc' % (dir2, filename[:-4], strcv(seq - 1), ch)

                    if st - sop >= 0:
                        tmp = tmp[st - sop:st]
                    else:
                        prevtmp = np.loadtxt(prevfile)
                        if os.path.exists(prevfile):
                            if st > 0:
                                tmp = np.concatenate((prevtmp[st - sop:], tmp[:st]))
                            else:
                                tmp = prevtmp[st - sop:st]
                        else:
                            if st > 0:
                                tmp = tmp[:st]
                            else:
                                raise Exception("file %s does not contain useful info" % filename)

                    tmp = tmp.reshape(1, tmp.shape[0])
                    data.append(tmp)

                else:
                    raise Exception("file %s not found" % filename)
            if len(data) > 0:
                concat = np.concatenate(data)
                print(concat.shape)
                yield concat  # 原来为：yield(concat)

    elif data_type == 'interictal':
        target_ = 'pat%sInteriktal' % strtrg
        dir2 = os.path.join(data_dir, target_)
        text_files = [f for f in os.listdir(dir2) if f.endswith('.asc')]
        prefixes = [text[:8] for text in text_files]
        prefixes = set(prefixes)
        prefixes = sorted(prefixes)

        totalfiles = len(text_files)
        print(prefixes, totalfiles)

        done = False
        count = 0

        for prefix in prefixes:
            i = 0
            while not done:

                i += 1

                stri = strcv(i)
                data = []
                for ch in range(1, 7):
                    filename = '%s/%s_%s_%d.asc' % (dir2, prefix, stri, ch)

                    if os.path.exists(filename):
                        # noinspection PyBroadException
                        # 去掉上面这一行，except会报警(因为 except 需要指定错误类型？)
                        try:                           
                            tmp = np.loadtxt(filename)
                            tmp = tmp.reshape(1, tmp.shape[0])
                            data.append(tmp)
                            count += 1
                        except:
                            print('OOOPS, this file can not be loaded', filename)
                    elif count >= totalfiles:
                        done = True
                    elif count < totalfiles:
                        break
                    else:
                        raise Exception("file %s not found" % filename)

                if i > 99999:
                    break

                if len(data) > 0:
                    yield np.concatenate(data)

def load_signals_CHBMIT(data_dir, target, data_type):
    print('load_signals_CHBMIT for Patient', target)
    from mne.io import RawArray, read_raw_edf
    # from mne.channels import read_montage
    from mne import create_info, concatenate_raws, pick_types
    from mne.filter import notch_filter

    onset = pd.read_csv(os.path.join(data_dir, 'seizure_summary.csv'), header=0)  # https://zhuanlan.zhihu.com/p/340441922 取第0行为表头（感觉记录的应该是开始和结束的秒数）
    #print (onset)
    osfilenames, szstart, szstop = onset['File_name'], onset['Seizure_start'], onset['Seizure_stop']  # 发作数据文件名，发作开始时间，发作结束时间
    osfilenames = list(osfilenames)
    #print ('Seizure files:', osfilenames)

    segment = pd.read_csv(os.path.join(data_dir, 'segmentation.csv'), header=None)  # '0' 表示对应的为发作间期文件名，那其他的呢？
    nsfilenames = list(segment[segment[1] == 0][0])  # segment[1]是第一列？(先列后行， 奇怪)  发作间期
    # chb14---只有     在nsfilenames里
    # chb14_24.edf,0
    # chb14_32.edf,0
    # chb14_37.edf,0
    # chb14_39.edf,0
    # chb14_42.edf,0

    nsdict = {
            '0': []
    }
    targets = [
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
        '22',
        '23'
    ]
    for t in targets:  # 把每一位患者的发作间期数据文件名放到一个键中
        nslist = [elem for elem in nsfilenames if
                  elem.find('chb%s_' % t) != -1 or
                  elem.find('chb0%s_' % t) != -1 or
                  elem.find('chb%sa_' % t) != -1 or
                  elem.find('chb%sb_' % t) != -1 or
                  elem.find('chb%sc_' % t) != -1]
        nsdict[t] = nslist  # 特定患者的发作间期数据文件名
    # nsfilenames = shuffle(nsfilenames, random_state=0)

    special_interictal = pd.read_csv(os.path.join(data_dir, 'special_interictal.csv'), header=None)
    sifilenames, sistart, sistop = special_interictal[0], special_interictal[1], special_interictal[2]  # 特殊发作数据文件名，发作开始时间，发作结束时间
    sifilenames = list(sifilenames)

    def strcv(num):  # 为小于10的患者编号加0
        if num < 10:
            return '0' + str(num)
        elif num < 100:
            return str(num)

    strtrg = 'chb' + strcv(int(target))  # chb14(如果是患者14的脑电数据，此处就是chb14)
    chb_dir = os.path.join(data_dir, strtrg)  # 设置（chb14）患者的癫痫数据的读取路径
    text_files = [f for f in os.listdir(chb_dir) if f.endswith('.edf')]  # （chb14）患者非发作期数据文件名
    #print (target,strtrg)
    print(text_files)

    # sch 原来没有这一句
    filenames = []
    #
    if data_type == 'ictal':
        filenames = [filename for filename in text_files if filename in osfilenames]  # 特定患者，文件名与发作期数据文件名相同则为发作前期数据文件(chb14)
        #print ('ictal files', filenames)
    elif data_type == 'interictal':
        filenames = [filename for filename in text_files if filename in nsdict[target]]  # 特定患者发作间期数据文件名(chb14)
        #print ('interictal files', filenames)

    totalfiles = len(filenames)  # 特定患者发作期或者发作间期的数量
    print('Total %s files %d' % (data_type, totalfiles))
    for filename in filenames:  # 特定患者特定时期
        exclude_chs = []
        if target in ['4', '9']:
            exclude_chs = [u'T8-P8']

        if target in ['13', '16']:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'FZ-CZ', u'CZ-PZ']
        elif target in ['4']:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT10-T8']
        else:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT9-FT10', u'FT10-T8']


        raweeg = read_raw_edf('%s/%s' % (chb_dir, filename),  # read_raw_edf()-----https://mp.weixin.qq.com/s?__biz=Mzg4MzYzNDgwMQ==&mid=2247508131&idx=1&sn=1bc6f4bbe93945b5b41c21b68d259466&source=41#wechat_redirect
                              #exclude=exclude_chs,  #only work in mne 0.16
                              verbose=0, preload=True)  # 预加载（preload=Ture）

        raweeg.pick_channels(chs)  # 按通道读取脑电信号 https://zhuanlan.zhihu.com/p/349988266 pick_channels() 根据通道名称获取指定的通道，也可以去除特定的通道，返回的是通道索引列表
        #print (raweeg.ch_names)
        #raweeg.notch_filter(freqs=np.arange(60,121,60))  # 陷波滤波器
        tmp = raweeg.to_data_frame()  # https://blog.csdn.net/weixin_43240818/article/details/87551326  to_data_frame()将读取的(脑电)数据转换成pandas的DataFrame数据格式
        tmp = tmp.as_matrix()  # tmp = tmp.as_matrix()---高版本的pandas已弃用，可替换为:tmp = tmp.values()  转换成numpy的特有数据格式

        if data_type == 'ictal':  # 发作期及发作前期处理
            sop = 30 * 60 * 256  # 30分钟 * 60分钟每秒 * 256的采样率 （两个发作时间间隔小于30min认为是一个发作）
            # get seizure onset information
            indices = [ind for ind, x in enumerate(osfilenames) if x == filename]  # 特定患者的发作前期数据文件名（chb14）
            if len(indices) > 0:  # 判断有没有发作期数据
                print('%d seizures in the file %s' % (len(indices), filename))
                prev_sp = -1e6  # -1000000 ？（前一个发作结束？）
                for i in range(len(indices)):
                    st = szstart[indices[i]] * 256 - 5 * 60 * 256  # SPH=5min  发作开始前五分钟（szstart和szstop记录的单位是秒数）可能小于零（从开始记录到发作不到五分钟）
                    sp = szstop[indices[i]] * 256  # 发作结束
                    #print ('Seizure %s %d starts at %d stops at %d last sz stop is %d' % (filename, i, (st+5*60*256),sp,prev_sp))

                    # take care of some special filenames
                    if filename[6] == '_':  # chb17a_03.edf
                        seq = int(filename[7:9])  # 取第七第八位（chb17a_03.edf）--- 03
                    else:  # chb14_01.edf
                        seq = int(filename[6:8])  # 取第六第七位（chb14_01.edf）--- 01
                    if filename == 'chb02_16+.edf':
                        prevfile = 'chb02_16.edf'
                    else:
                        if filename[6] == '_':
                            prevfile = '%s_%s.edf' % (filename[:6], strcv(seq-1))  # 减 1
                        else:
                            prevfile = '%s_%s.edf' % (filename[:5], strcv(seq-1))

                    # 取癫痫发作前 35 分钟到前 5 分钟的数据
                    if st - sop > prev_sp:  # 发作前5分钟的采样点 - 30min*60min/s*256采样率的采样点    ? st - sop 最小是 - 35 * 60 * 256 > -1e6
                        prev_sp = sp  # 用于判断一次发作与下一次发作中间时间间隔是否大于等于 35 分钟
                        if st - sop >= 0:  # 癫痫脑电记录开始到癫痫开始发作的时间间隔是否大于等于 35 min
                            data = tmp[st - sop: st]  # 数据：癫痫开始前 35 min 至 癫痫发作前 5 分钟
                        else:  # 用前一个序的脑电数据和这个序的脑电数据拼接成一个 30min 的片段
                            if os.path.exists('%s/%s' % (chb_dir, prevfile)):  # 判断前一个序的脑电数据是否存在（正在处理的是：chb14_03.edf，这句话是在判断 chb14_02.edf 是否存在）
                                raweeg = read_raw_edf('%s/%s' % (chb_dir, prevfile), preload=True, verbose=0)  # 读前一个序的脑电数据（rawEEG中数据已被替换）
                                raweeg.pick_channels(chs)
                                prevtmp = raweeg.to_data_frame()
                                prevtmp = prevtmp.as_matrix()  # prevtmp = prevtmp.as_matrix()
                                if st > 0:  #
                                    data = np.concatenate((prevtmp[st - sop:], tmp[:st]))  # https://www.cnblogs.com/shueixue/p/10953699.html 拼接前一个序和这个序的有效数据
                                else:
                                    data = prevtmp[st - sop:st]

                            else:
                                if st > 0:
                                    data = tmp[:st]  # 实在没办法了，就取 0 到癫痫发作的数据（没有前一个序）
                                else:  # 记录一开始就发作
                                    #raise Exception("file %s does not contain useful info" % filename)
                                    print("WARNING: file %s does not contain useful info" % filename)
                                    continue
                    else:  #
                        prev_sp = sp  # 判断一次发作与下一次发作中间时间间隔是否大于等于 35 分钟（第二次及以后的循环中才是这样）
                        continue

                    print('data shape : ', data.shape)
                    if data.shape[0] == sop:  # shape[0]就是读取矩阵第一维度的长度(列数？)
                        yield data  # 原来为：yield(data) https://blog.csdn.net/mieleizhi0522/article/details/82142856 没大看懂
                    else:
                        continue

        elif data_type == 'interictal':  # 发作间期
            if filename in sifilenames:  # 发作间期是特殊发作间期
                st = sistart[sifilenames.index(filename)]  # 特殊发作间期开始（special_interictal.csv）
                sp = sistop[sifilenames.index(filename)]  # 特殊发作间期结束（special_interictal.csv）
                if sp < 0:  # -1 就是直到最后（有些 sp 是等于 -1 的，在 special_interictal.csv 文件中）
                    data = tmp[st * 256:]
                else:
                    data = tmp[st * 256: sp * 256]
            else:  # 发作间期不是特殊发作间期
                data = tmp  # 整个文件都是发作间期数据
            print('data shape', data.shape)
            yield data  # 原来为：yield(data)

class PrepData:  # class PreData():
    def __init__(self, target, type, settings):
        self.target = target
        self.settings = settings
        self.type = type
        self.global_proj = np.array([0.0]*114)  # 114行的列向量，元素全为 0.0

    def read_raw_signal(self):
        if self.settings['dataset'] == 'CHBMIT':
            self.samp_freq = 256  # 样本采样频率
            self.freq = 256  #
            self.global_proj = np.array([0.0]*114)  # 在 preprocess() 中用到了
            return load_signals_CHBMIT(self.settings['datadir'], self.target, self.type)  #
        elif self.settings['dataset'] == 'FB':
            self.samp_freq = 256
            self.freq = 256
            self.global_proj = np.array([0.0]*114)
            return load_signals_FB(self.settings['datadir'], self.target, self.type)
        elif self.settings['dataset'] == 'Kaggle2014Pred':
            if self.type == 'ictal':
                data_type = 'preictal'
            else:
                data_type = self.type
            return load_signals_Kaggle2014Pred(self.settings['datadir'], self.target, data_type)

        return 'array, freq, misc'  # ？


    def preprocess_Kaggle(self, data_):
        ictal = self.type == 'ictal'  # 是发作期数据
        interictal = self.type == 'interictal'  # 是发作间期数据
        if 'Dog_' in self.target:  # 狗的脑电数据
            targetFrequency = 200  # re-sample to target frequency
            DataSampleSize = targetFrequency
            numts = 29.97
        else:
            targetFrequency = 1000
            DataSampleSize = int(targetFrequency / 5)
            numts = 60
        sampleSizeinSecond = 600

        df_sampling = pd.read_csv(
            'sampling_%s.csv' % self.settings['dataset'],
            header=0, index_col=None)  # 读取sampling_CHBMIT.csv中数据
        trg = self.target  # 患者编号列表？
        print(df_sampling)
        print(df_sampling[df_sampling.Subject == trg].ictal_ovl.values)
        ictal_ovl_pt = \
            df_sampling[df_sampling.Subject == trg].ictal_ovl.values[0]
        ictal_ovl_len = int(targetFrequency * ictal_ovl_pt * numts)

        def process_raw_data(mat_data):
            print('Loading data')
            X = []
            y = []
            sequences = []
            #scale_ = scale_coef[target]
            for segment in mat_data:
                for skey in segment.keys():  # https://www.w3school.com.cn/python/ref_dictionary_keys.asp
                    if "_segment_" in skey.lower():  # https://www.runoob.com/python/att-string-lower.html
                        mykey = skey
                if ictal:
                    y_value = 1
                    sequence = segment[mykey][0][0][4][0][0]
                else:
                    y_value = 0

                data = segment[mykey][0][0][0]
                sampleFrequency = segment[mykey][0][0][2][0][0]

                if sampleFrequency > targetFrequency:  # resample to target frequency
                    data = resample(data, targetFrequency*sampleSizeinSecond, axis=-1)

                data = data.transpose()

                from mne.filter import notch_filter

                totalSample = int(data.shape[0]/DataSampleSize/numts) + 1
                window_len = int(DataSampleSize*numts)
                #print ('DEBUG: window_len, totalSample', window_len, totalSample)
                for i in range(totalSample):

                    if (i+1)*window_len <= data.shape[0]:
                        s = data[i*window_len:(i+1)*window_len, :]
                        stft_data = stft.spectrogram(s, framelength=DataSampleSize, centered=False)
                        stft_data = stft_data[1:, :, :]
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0
                        stft_data = np.transpose(stft_data, (2, 1, 0))
                        stft_data = np.abs(stft_data)+1e-6


                        stft_data = stft_data.reshape(-1, 1, stft_data.shape[0], stft_data.shape[1], stft_data.shape[2])

                        X.append(stft_data)
                        y.append(y_value)
                        if ictal:
                            sequences.append(sequence)

                if ictal:
                    #print ('Generating more preictal samples....')
                    #overlapped window
                    i = 1
                    while window_len + (i + 1) * ictal_ovl_len <= data.shape[0]:  # 原来为：while(window_len + (i + 1) * ictal_ovl_len <= data.shape[0]):
                        s = data[i*ictal_ovl_len:i*ictal_ovl_len + window_len, :]

                        stft_data = stft.spectrogram(s, framelength=DataSampleSize, centered=False)

                        stft_data = stft_data[1:, :, :]
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0
                        stft_data = np.transpose(stft_data, (2, 1, 0))
                        stft_data = np.abs(stft_data)+1e-6

                        stft_data = stft_data.reshape(-1, 1, stft_data.shape[0], stft_data.shape[1], stft_data.shape[2])

                        X.append(stft_data)
                        y.append(2)
                        sequences.append(sequence)
                        i += 1

            if ictal:
                assert len(X) == len(y)
                assert len(X) == len(sequences)
                X, y = group_seizure(X, y, sequences)
                print('X', len(X), X[0].shape)
                return X, y
            elif interictal:
                X = np.concatenate(X)
                y = np.array(y)
                print('X', X.shape, 'y', y.shape)
                return X, y
            else:
                X = np.concatenate(X)
                print('X', X.shape)
                return X, None

        data = process_raw_data(data_)
        return data

    def preprocess(self, data_):  # 从 load_signals_CHBMIT() 接受data_
        ictal = self.type == 'ictal'
        interictal = self.type == 'interictal'
        targetFrequency = self.freq  # re-sample to target frequency
        numts = 28
        
        df_sampling = pd.read_csv(
            'sampling_%s.csv' % self.settings['dataset'],
            header=0, index_col=None)  # 读取 sampling_CHBMIT.csv （数据采样相关）
        trg = int(self.target)  # 患者编号
        print(df_sampling)
        print(df_sampling[df_sampling.Subject == trg].ictal_ovl.values)  # 有点意思（类似循环） 打印 ictal_ovl---矩阵的形式
        # sch 原来没有这一句
        test = df_sampling[df_sampling.Subject == trg].ictal_ovl.values
        #
        ictal_ovl_pt = \
            df_sampling[df_sampling.Subject == trg].ictal_ovl.values[0]  # 获取ictal_ovl的值
        ictal_ovl_len = int(targetFrequency * ictal_ovl_pt * numts)  # 什么意思？

        def process_raw_data(mat_data):            
            print('Loading data')
            X = []
            y = []
            # scale_ = scale_coef[target]
            for data in mat_data:  # mat_data 是通过 load_signals_CHBMIT 迭代出来的（迭代的目的是为了减小内存消耗）（第一次执行for循环时有一段空档期就是在迭代处理第一次发作前期的癫痫数据）
                if self.settings['dataset'] == 'FB':
                    data = data.transpose()  # 交换维度 https://blog.csdn.net/qq_37377691/article/details/80086480
                if ictal:
                    y_value = 1  # 发作前期
                else:
                    y_value = 0  # 发作间期

                X_temp = []
                y_temp = []
    
                totalSample = int(data.shape[0] / targetFrequency / numts) + 1  # 采样点数 / 采样频率 / 窗长（秒数） + 1（ + 1 ？ 若最后一个不够 28 秒也把它当作一个样本）
                window_len = int(targetFrequency * numts)  # 28 秒的窗长？  int() 去尾取整
                for i in range(totalSample):
                    if (i + 1) * window_len <= data.shape[0]:  # data: 22 信道 460800个采样点 data[0] = 460800  最后一个窗如果不够 28 秒则直接去掉（没有对应的else）
                        s = data[i * window_len: (i+1) * window_len, :]

                        stft_data = stft.spectrogram(s, framelength=targetFrequency, centered=False)  # 短时傅里叶变换
                        stft_data = np.transpose(stft_data, (2, 1, 0))  # https://blog.csdn.net/u012762410/article/details/78912667
                        stft_data = np.abs(stft_data)+1e-6  # +1e-6 防止下溢？

                        if self.settings['dataset'] == 'FB':  # 陷波器的作用 FB 为 50 hz工频干扰
                            stft_data = np.concatenate((stft_data[:, :, 1:47],
                                                        stft_data[:, :, 54:97],
                                                        stft_data[:, :, 104:]),
                                                       axis=-1)  # -1表示最后一个，https://blog.csdn.net/xiewenbo/article/details/89281264?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link
                        elif self.settings['dataset'] == 'CHBMIT':  # 陷波器的作用 CHB 为 60 hz工频干扰
                            stft_data = np.concatenate((stft_data[:, :, 1:57],
                                                        stft_data[:, :, 64:117],
                                                        stft_data[:, :, 124:]),
                                                       axis=-1)  # 按第三维进行拼接  陷波器的作用？
                        stft_data = np.log10(stft_data)  # 数据搞集中一点（最大值与最小值差别太大）
                        indices = np.where(stft_data <= 0)  # https://www.cnblogs.com/massquantity/p/8908859.html
                        stft_data[indices] = 0  # 为什么小于 0 的部分要让它等于 0 ？

                        stft_data = stft_data.reshape(-1, 1, stft_data.shape[0],
                                                      stft_data.shape[1],
                                                      stft_data.shape[2])  # -1 1 22 55 114（转化成ndarray）(好像没有实质性变化)


                        X_temp.append(stft_data)
                        y_temp.append(y_value)

                # overdsampling ictal （ictal_ovl = 1 的患者数据岂不是重复了？）
                if ictal:
                    i = 1
                    print('ictal_ovl_len =', ictal_ovl_len)
                    while window_len + (i + 1) * ictal_ovl_len <= data.shape[0]:  # data.shape[0] 返回列数（采样长度）
                        s = data[i * ictal_ovl_len: i * ictal_ovl_len + window_len, :]  # 第一个窗不要？

                        stft_data = stft.spectrogram(s, framelength=targetFrequency, centered=False)
                        stft_data = np.transpose(stft_data, (2, 1, 0))
                        stft_data = np.abs(stft_data)+1e-6

                        if self.settings['dataset'] == 'FB':
                            stft_data = np.concatenate((stft_data[:, :, 1:47],
                                                        stft_data[:, :, 54:97],
                                                        stft_data[:, :, 104:]),
                                                       axis=-1)
                        elif self.settings['dataset'] == 'CHBMIT':
                            stft_data = np.concatenate((stft_data[:, :, 1:57],
                                                        stft_data[:, :, 64:117],
                                                        stft_data[:, :, 124:]),
                                                       axis=-1)
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0

                        proj = np.sum(stft_data, axis=(0, 1), keepdims=False)  # 列相加，压缩行。 https://zhuanlan.zhihu.com/p/85790648
                        self.global_proj += proj / 1000.0  # ？

                        stft_data = stft_data.reshape(-1, 1, stft_data.shape[0],
                                                      stft_data.shape[1],
                                                      stft_data.shape[2])

                        X_temp.append(stft_data)
                        # differentiate between non-overlapped and overlapped
                        # samples. Testing only uses non-overlapped ones.
                        y_temp.append(2)
                        i += 1

                X_temp = np.concatenate(X_temp, axis=0)
                y_temp = np.array(y_temp)
                X.append(X_temp)
                y.append(y_temp)

            if ictal or interictal:
                #y = np.array(y)
                print('X', len(X), X[0].shape, 'y', len(y), y[0].shape)
                return X, y
            else:  # 不是发作前期也不是发作间期？（难道是发作期？）
                print('X', X.shape)
                return X

        data = process_raw_data(data_)

        return data

    def apply(self):
        filename = '%s_%s' % (self.type, self.target)
        cache = load_hickle_file(
            os.path.join(self.settings['cachedir'], filename))
        if cache is not None:
            return cache

        data = self.read_raw_signal()  # 还能这么用？
        if self.settings['dataset'] == 'Kaggle2014Pred':
            X, y = self.preprocess_Kaggle(data)
        else:
            X, y = self.preprocess(data)
        save_hickle_file(
            os.path.join(self.settings['cachedir'], filename),
            [X, y])
        return X, y


