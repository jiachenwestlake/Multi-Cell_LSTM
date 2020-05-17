import math
import random

class Multi_Task_Dataset:
    def __init__(self, train_instances, batch_size, train_rate=1.0, shuffle=True):
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._train_instances = self._split_train_instances(train_instances, train_rate) #instant list[instance1, instance2, ...]
        self._train_batches = self._sample_batches(self._train_instances, shuffle=self._shuffle) # dynamic list of batches [batch1[], batch2[], ...]


    @staticmethod
    def _split_train_instances(train_instances, train_rate):
        """
        Using part of train data for training
        """
        if train_rate < 1.0:
            train_size = int(len(train_instances) * train_rate)
            return train_instances[: train_size]
        else:
            return train_instances

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batches(self):
        """
        the maximum number of batches
        """
        return math.ceil(float(len(self._train_instances)) / float(self._batch_size))

    @property
    def train_size(self):
        return len(self._train_instances)

    @property
    def train_instances(self):
        return self._train_instances

    def train_batches(self):
        if self._train_batches is None or len(self._train_batches) < self.num_batches:
            self._train_batches = self._sample_batches(self._train_instances, shuffle=self._shuffle)
        return self._train_batches

    def next_batch(self):
        """
        return a batch and pop it, no end !
        :return: batch of training instance [instance1, instance2, ...]
                 end_flag: True if data is None after training this batch
        """
        if self._train_batches is None or len(self._train_batches) == 0:
            self._train_batches = self._sample_batches(self._train_instances, shuffle=self._shuffle)
        batch = self._train_batches.pop(0)
        end_flag = True if len(self._train_batches) == 0 else False
        return batch, end_flag

    def _sample_batches(self, dataset, shuffle=True):
        """
        :param dataset: instant list of training instances
        :param shuffle: shuffle the dataset before spliting into batches
        :return: list of batches[batch1[], batch2[], ...]
        """
        if shuffle:
            random.shuffle(dataset)
        data_batches = []
        dataset_size = len(dataset)
        for i in range(0, dataset_size, self._batch_size):
            batch_data = dataset[i: i + self._batch_size]
            data_batches.append(batch_data)
        return data_batches

def batch_arrange(num_batch_T, num_batch_S, mix_rate_S=1.0, mix_scale='2max', mix_pattern='rotation'):
    """
    :param num_batch_T:
    :param num_batch_S:
    :param mix_rate_S: number of source_data in 1 epoch
    :param mix_pattern: 'rotation': 'random':
           mix_scale:(only for random pattern) all, 2max
    :return:
    """
    num_batch_S = int(num_batch_S * mix_rate_S)
    batches = []
    if mix_pattern == 'rotation':
        paired_num = max(num_batch_S, num_batch_T)
        batches = ['Target', 'Source'] * paired_num
    elif mix_pattern == 'random':
        if mix_scale == 'all':
            batches += ['Target'] * num_batch_T
            batches += ['Source'] * num_batch_S
            random.shuffle(batches)
        elif mix_scale == '2max':
            paired_num = max(num_batch_S, num_batch_T)
            batches += ['Target'] * paired_num
            batches += ['Source'] * paired_num
            random.shuffle(batches)
    return batches




