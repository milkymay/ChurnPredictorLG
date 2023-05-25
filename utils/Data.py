import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class Data:
    def __init__(self, file_path, upsample=False, text_analysis=None):
        text_analysis = [] if text_analysis is None else text_analysis
        kids_lessons_monthly = pd.read_csv(file_path)
        lines = ['days_per_month', 'days_this_year', 'group_size', 'age', 'age_mean',
                 'group_size_rolling', 'teachers_num_rolling', 'unique_teachers_rolling', 'year_at_lgeg']

        if "dostoevsky" in text_analysis and "chatGPT" in text_analysis:
            lines.remove('teachers_num_rolling')
            lines += ['comment_negative_rolling',
                      'embedding_0',
                      'embedding_1',
                      'embedding_2',
                      'embedding_3',
                      'embedding_4',
                      'embedding_5',
                      'embedding_6',
                      'embedding_7',
                      'embedding_8',
                      'embedding_9',
                      'embedding_10',
                      'embedding_11',
                      'embedding_12',
                      'embedding_13',
                      'embedding_14',
                      'embedding_15',
                      'embedding_17',
                      'embedding_18',
                      'embedding_19']

        elif "dostoevsky" in text_analysis:
            lines += ['comment_count',
                      'comment_speech_rolling', 'comment_positive_rolling',
                      'comment_negative_rolling']

        elif "chatGPT" in text_analysis:
            lines += ['embedding_0',
                      'embedding_1',
                      'embedding_2',
                      'embedding_3',
                      'embedding_4',
                      'embedding_5',
                      'embedding_6',
                      'embedding_7',
                      'embedding_8',
                      'embedding_9',
                      'embedding_10',
                      'embedding_11',
                      'embedding_12',
                      'embedding_13',
                      'embedding_14',
                      'embedding_15',
                      'embedding_16',
                      'embedding_17',
                      'embedding_18',
                      'embedding_19']

        self.X_true = kids_lessons_monthly.reset_index()[lines].copy()

        self.y_true = kids_lessons_monthly['y'].copy()
        self.features_names = self.X_true.columns
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.X_true.values, self.y_true.values, random_state=42)
        self.X_train, self.y_train = self.prepare_trains(X_train, y_train, upsample)

    def prepare_trains(self, X_train, y_train, upsample=False):
        data = np.concatenate([X_train, y_train.reshape((len(y_train), 1))], axis=1)
        not_left = data[data[:, -1] == 0.]
        left = data[data[:, -1] == 1]
        left_upsampled = resample(left,
                                  replace=True,
                                  n_samples=len(not_left),
                                  random_state=42)
        unleft_downsampled = resample(not_left,
                                      replace=False,
                                      n_samples=len(left),
                                      random_state=42)
        upsampled = np.concatenate([not_left, left_upsampled])
        downsampled = np.concatenate([unleft_downsampled, left])
        X_train_up = upsampled[:, :-1]
        y_train_up = upsampled[:, -1]
        X_train_down = downsampled[:, :-1]
        y_train_down = downsampled[:, -1]
        if upsample:
            return X_train_up, y_train_up
        return X_train_down, y_train_down
