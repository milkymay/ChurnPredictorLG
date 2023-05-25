import csv
import json
import os

import pandas as pd
from scipy.stats import mode
import numpy as np
from os import walk

from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

from data_prepare.chatty.ChattyEmbeddingManager import ChattyEmbeddingManager
from data_prepare.chatty.ChattyPromptManager import ChattyPromptManager


def to_school_year(dt: pd._libs.tslibs.timestamps.Timestamp) -> str:
    """Convert a pd datetime into a school year"""
    return (str(dt.year) + '/' + str(dt.year + 1)
            if dt.month > 8
            else str(dt.year - 1) + '/' + str(dt.year))


def group_size_up_to(s, df: pd.DataFrame):
    df_tmp = (df.reset_index().set_index(['student_id', 'year'])
        .loc[s.name[:2]])
    return (df_tmp.loc[df_tmp['index'] <= s['date'], 'student_id_count']
            .mean())


def df_prepare_kids_lessons(raw_lessons: pd.DataFrame) -> pd.DataFrame:
    kids_lessons = (raw_lessons.loc[(raw_lessons['stage'] == 'D') | (raw_lessons['stage'] == 'E')]
                    .set_index('datetime')
                    .copy())
    kids_lessons = kids_lessons[kids_lessons['presence'] == 1]

    kids_lessons['datetime'] = kids_lessons.index
    kids_lessons['year'] = kids_lessons.loc[:, 'datetime'].apply(to_school_year)
    kids_lessons['month'] = (kids_lessons.index.month + 3) % 12 + 1
    kids_lessons['day'] = kids_lessons['datetime'].dt.day
    kids_lessons.index = kids_lessons.index.date

    kids_lessons['sentiment'] = np.log(kids_lessons['positive'] / kids_lessons['negative'])
    kids_lessons['comment_length'] = np.log(kids_lessons['comment_for_parent'].str.len() + 1)

    return kids_lessons


def df_prepare_lessons_monthly(kids_lessons: pd.DataFrame):
    kids_lessons_monthly = (kids_lessons.reset_index()
                            .groupby(['student_id', 'year', 'month'])
                            .agg({'index': 'nunique',
                                  'stage': 'first',
                                  'age': 'mean',
                                  'age_mean': 'mean',
                                  'student_id_count': lambda x: x.mean().round(1),
                                  'sentiment': lambda x: x.mean().round(1),
                                  'neutral': 'mean',
                                  'skip': 'mean',
                                  'speech': 'mean',
                                  'positive': 'mean',
                                  'negative': 'mean',
                                  'comment_length': lambda x: x.mean().round(1),
                                  'teacher_id': 'nunique',
                                  'comment_for_parent': lambda x: x.str.len().astype(bool).sum()}))

    kids_lessons_monthly['cumsum'] = (kids_lessons_monthly.reset_index()
                                      .groupby(['student_id', 'year'])
                                      .agg({'index': 'cumsum'})
                                      .values)

    kids_lessons_monthly.columns = ['days_per_month',
                                    'stage',
                                    'age',
                                    'age_mean',
                                    'group_size',
                                    'sentiment',
                                    'neutral',
                                    'skip',
                                    'speech',
                                    'positive',
                                    'negative',
                                    'comment_length',
                                    'teachers_num',
                                    'comment_count',
                                    'days_this_year']

    # Mean group size monthly up to this month

    df = (kids_lessons.reset_index()
          .groupby(['student_id', 'year', 'month'])
          .agg({'student_id_count': 'mean'}))
    df['cumsum'] = (df.reset_index()
                    .groupby(['student_id', 'year'])
                    .agg({'student_id_count': 'cumsum'})
                    .values)
    df['cumcount'] = (df.reset_index()
                      .groupby(['student_id', 'year'])
                      .agg({'student_id_count': 'cumcount'})
                      .values)
    kids_lessons_monthly['group_size_rolling'] = (df['cumsum'] / (df['cumcount'] + 1)).round(1)

    # Mean comment sentiment up to this month

    df = (kids_lessons.reset_index()
          .groupby(['student_id', 'year', 'month'])
          .agg({'sentiment': 'mean'}))
    df['cumsum'] = (df.reset_index()
                    .groupby(['student_id', 'year'])
                    .agg({'sentiment': 'cumsum'})
                    .values)
    df['cumcount'] = (df.reset_index()
                      .groupby(['student_id', 'year'])
                      .agg({'sentiment': 'cumcount'})
                      .values)
    kids_lessons_monthly['comment_sentiment_rolling'] = (df['cumsum'] / (df['cumcount'] + 1)).round(1)

    # Mean comment sentiments up to this month

    df = (kids_lessons.reset_index()
          .groupby(['student_id', 'year', 'month'])
          .agg({'neutral': 'mean',
                'skip': 'mean',
                'speech': 'mean',
                'positive': 'mean',
                'negative': 'mean', }))
    for i in df.columns:
        df[f'cumsum_{i}'] = (df.reset_index()
                             .groupby(['student_id', 'year'])
                             .agg({i: 'cumsum'})
                             .values)
        df[f'cumcount_{i}'] = (df.reset_index()
                               .groupby(['student_id', 'year'])
                               .agg({i: 'cumcount'})
                               .values)
        kids_lessons_monthly[f'comment_{i}_rolling'] = (df[f'cumsum_{i}'] / (df[f'cumcount_{i}'] + 1))

    # Mean comment length up to this month

    df = (kids_lessons.reset_index()
          .groupby(['student_id', 'year', 'month'])
          .agg({'comment_length': 'mean'}))
    df['cumsum'] = (df.reset_index()
                    .groupby(['student_id', 'year'])
                    .agg({'comment_length': 'cumsum'})
                    .values)
    df['cumcount'] = (df.reset_index()
                      .groupby(['student_id', 'year'])
                      .agg({'comment_length': 'cumcount'})
                      .values)
    kids_lessons_monthly['comment_length_rolling'] = (df['cumsum'] / (df['cumcount'] + 1)).round(1)

    # Mean number of teachers monthly up to this month

    df = (kids_lessons.reset_index()
          .groupby(['student_id', 'year', 'month'])
          .agg({'teacher_id': 'nunique'}))
    df['cumsum'] = (df.reset_index()
                    .groupby(['student_id', 'year'])
                    .agg({'teacher_id': 'cumsum'})
                    .values)
    df['cumcount'] = (df.reset_index()
                      .groupby(['student_id', 'year'])
                      .agg({'teacher_id': 'cumcount'})
                      .values)
    kids_lessons_monthly['teachers_num_rolling'] = (df['cumsum'] / (df['cumcount'] + 1)).round(1)

    # Unique teachers met this school year (up to this month)

    df = (kids_lessons.reset_index()
          .sort_values(by=['student_id', 'datetime']))
    unique_count = (df[['student_id', 'year', 'teacher_id']].drop_duplicates()
                    .groupby(['student_id', 'year'])
                    .cumcount()
                    .astype(int)) + 1
    df['cumnunique_teachers'] = (unique_count.reindex(df.index)
                                 .ffill())
    kids_lessons_monthly['unique_teachers_rolling'] = (df.groupby(['student_id', 'year', 'month'])
                                                       .agg({'cumnunique_teachers': 'max'})
                                                       .astype(int))

    # Age diff vs the group this month
    kids_lessons_monthly['age_mean'] = (kids_lessons_monthly['age'] - kids_lessons_monthly['age_mean']).round(1)

    kids_lessons_monthly = kids_lessons_monthly.dropna()

    # Change in number of teachers this month vs the average up until this month
    # Change in the group size this month vs the average up until this month
    # Change in the comment sentiment this month vs the average up until this month

    kids_lessons_monthly['teachers_num'] = kids_lessons_monthly['teachers_num'] - kids_lessons_monthly[
        'teachers_num_rolling']
    kids_lessons_monthly['group_size'] = kids_lessons_monthly['group_size'] - kids_lessons_monthly[
        'group_size_rolling']
    kids_lessons_monthly['comment_length'] = kids_lessons_monthly['comment_length'] - kids_lessons_monthly[
        'comment_length_rolling']

    # Year number for this month (how many years studied at LGEG up until now)

    df = kids_lessons_monthly.reset_index().loc[:, ['student_id', 'year']]
    kids_lessons_monthly['year_at_lgeg'] = ((df.drop_duplicates()
                                             .groupby('student_id')
                                             .agg({'year': 'cumcount'}) + 1).reindex(df.index)
                                            .ffill()
                                            .astype(int)
                                            .values)

    kids_lessons_monthly['y'] = 0
    # mark the last lesson as '2'
    (kids_lessons_monthly.loc[pd.Index(kids_lessons_monthly.reset_index()
                                       .groupby(['student_id', 'year'])
                                       .agg({'month': 'last'})
                                       .reset_index()
                                       .values), 'y']) = 2
    # then mark the next to last as '1'
    (kids_lessons_monthly.loc[pd.Index(kids_lessons_monthly[kids_lessons_monthly['y'] != 2].reset_index()
                                       .groupby(['student_id', 'year'])
                                       .agg({'month': 'last'})
                                       .reset_index()
                                       .values), 'y']) = 1

    kids_lessons_monthly = kids_lessons_monthly[kids_lessons_monthly['y'] != 2]

    # drop the last 5 months of the year
    kids_lessons_monthly = kids_lessons_monthly.drop(level=2, labels=[8, 9, 10, 11, 12])

    # drop the 2013/2014 year
    kids_lessons_monthly = kids_lessons_monthly.drop(level=1, labels=['2013/2014'])

    # sterilise the ages
    kids_lessons_monthly = kids_lessons_monthly[(kids_lessons_monthly['age'] > 5) &
                                                (kids_lessons_monthly['age'] < 15)]

    return kids_lessons_monthly


class PrepareData:
    def __init__(self, conf_path: str):
        data_json = json.load(open(conf_path, 'r'))
        self.data_store_path = data_json["data_store_path"]
        self.raw_xlsx_lessons_folder_path = data_json["raw_xlsx_lessons_folder_path"]
        self.raw_xlsx_attendance_folder_path = data_json["raw_xlsx_attendance_folder_path"]
        self.raw_xlsx_coursegroups_path = data_json["raw_xlsx_coursegroups"]
        self.raw_xlsx_lessons_master_path = data_json["raw_xlsx_lessons_master"]
        self.log_path = data_json["log_path"]
        self.conf_path = conf_path

    def df_prepare_raw_lessons(self, coursegroups: pd.DataFrame, chatty_emb, chatty_prompt) -> pd.DataFrame:
        for i in walk(self.raw_xlsx_lessons_folder_path):
            lessons_lst = i[2]
        raw_lessons = pd.DataFrame([])
        for l in lessons_lst:
            raw_lessons = pd.concat(
                [raw_lessons, pd.read_excel(self.raw_xlsx_lessons_folder_path + '/' + l, skiprows=2)])
        raw_lessons.columns = ['teacher_id',
                               'student_id',
                               'datetime',
                               'subject',
                               'coursegroup']
        raw_lessons = (raw_lessons.join(raw_lessons.groupby(['datetime',
                                                             'subject',
                                                             'coursegroup'])
                                        .agg({'student_id': 'count'}),
                                        on=['datetime',
                                            'subject',
                                            'coursegroup'],
                                        rsuffix='_count'))
        raw_lessons = raw_lessons.join(coursegroups['stage'], on='coursegroup', how='left')
        raw_lessons = raw_lessons.join(pd.read_excel(self.raw_xlsx_lessons_master_path, skiprows=2)
                                       .groupby('student_id')
                                       .agg({'birth_date': max}),
                                       on='student_id', how='left')
        raw_lessons['age'] = ((raw_lessons['datetime'] - raw_lessons['birth_date']).dt.days / 365).round(1)

        raw_lessons = (raw_lessons.join(raw_lessons.groupby(['datetime',
                                                             'subject',
                                                             'coursegroup'])
                                        .agg({'age': 'mean'}),
                                        on=['datetime',
                                            'subject',
                                            'coursegroup'],
                                        rsuffix='_mean'))

        for i in walk(self.raw_xlsx_attendance_folder_path):
            lessons_lst = i[2]

        attendance = pd.DataFrame([])
        for l in lessons_lst:
            attendance = pd.concat(
                [attendance, pd.read_excel(self.raw_xlsx_attendance_folder_path + '/' + l, skiprows=2)])

        attendance['homework_done'] = attendance['homework_done'].fillna(0.)
        attendance['presence'] = attendance['presence'].fillna(-1)
        attendance['comment_for_parent'] = attendance['comment_for_parent'].fillna('')
        attendance['comment_for_student'] = attendance['comment_for_parent'].fillna('')
        attendance['comment_for_supervisor'] = attendance['comment_for_supervisor'].fillna('')

        attendance = attendance[['teacher id', 'student_id',
                                 'SlotStartTime', 'presence', 'homework_done',
                                 'comment_for_parent', 'comment_for_student', 'comment_for_supervisor']]
        attendance.columns = ['teacher_id', 'student_id',
                              'datetime', 'presence', 'homework_done',
                              'comment_for_parent', 'comment_for_student', 'comment_for_supervisor']
        tokenizer = RegexTokenizer()
        nlp_model = FastTextSocialNetworkModel(tokenizer=tokenizer)
        nlp_result = nlp_model.predict(attendance['comment_for_parent'].tolist())

        attendance = pd.concat([attendance.reset_index(), pd.DataFrame(nlp_result)], axis=1)
        attendance = attendance.set_index('index')

        raw_lessons = pd.merge(raw_lessons, attendance,
                               on=['teacher_id', 'student_id', 'datetime']).dropna()

        if chatty_prompt:
            chattyPromptManager = ChattyPromptManager(self.conf_path, raw_lessons['comment_for_parent'].tolist())
            chattyPromptResults = chattyPromptManager.prepare_prompts_results()

            attendance = pd.concat([attendance.reset_index(), chattyPromptResults], axis=1)
            attendance = attendance.set_index('index')

            raw_lessons = pd.merge(raw_lessons, attendance,
                                   on=['teacher_id', 'student_id', 'datetime']).dropna()

        if chatty_emb:
            chattyEmbeddingManager = ChattyEmbeddingManager(self.conf_path, raw_lessons,
                                                            log_path=self.log_path + "/embeddings")
            raw_lessons = chattyEmbeddingManager.add_embeddings()

        return raw_lessons

    def df_prepare_coursegroups(self) -> pd.DataFrame:
        coursegroups = pd.read_excel(self.raw_xlsx_coursegroups_path, skiprows=2)
        coursegroups.columns = ['coursegroup_id', 'coursegroup', 'stage', 'stage_id']
        coursegroups = coursegroups.drop(['coursegroup_id', 'stage_id'], axis=1).groupby('coursegroup').agg(
            lambda s: mode(s.values, keepdims=True)[0])
        return coursegroups

    def df_prepare_all(self, chatty_emb, chatty_prompt):
        coursegroups = self.df_prepare_coursegroups()
        raw_lessons = self.df_prepare_raw_lessons(coursegroups, chatty_emb, chatty_prompt)
        kids_lessons = df_prepare_kids_lessons(raw_lessons)

        return raw_lessons, kids_lessons

    def prepare_data(self, force=False, chatty_emb=False, chatty_prompt=False):
        file_path = self.data_store_path + "/kids_lessons_monthly.csv"
        if os.path.isfile(file_path) and not force:
            print(
                f"File '{file_path}' already exists. Set 'force=True' to rewrite data.")
            return
        else:
            if force:
                print("Forced rewriting...")
            print("Evaluating raw_lessons...")
            raw_lessons, kids_lessons = self.df_prepare_all(chatty_emb, chatty_prompt)
            print("Evaluating kids_lessons_monthly...")
            kids_lessons_monthly = df_prepare_lessons_monthly(kids_lessons)

            datas = dict(raw_lessons=raw_lessons,
                         kids_lessons_monthly=kids_lessons_monthly)
            print("Uploading data to " + self.data_store_path)
            for name in datas.keys():
                file_path = self.data_store_path + "/" + name + ".csv"
                datas[name].to_csv(file_path, sep=',', index=True, encoding='utf-8')

        print("Done")
