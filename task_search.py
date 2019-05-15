import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import re
import random

from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

class TaskSearch:
    def __init__(self):
        self.load_all_data('/home/dmitry/Development/geekhubds/dv_tasks_project/data')
        self.set_unbillable_field()
        self.n_neighbors = 10
        self.metric = 'cosine'
        self.initialize_knn()

    def search(self, search_word, n_neighbour_search=10):
        print('here (in search')
        vectorizer2 = CountVectorizer(stop_words='english', vocabulary=self.vocabulary)
        test_word_vectorized = vectorizer2.fit_transform([search_word])
        neighbors = self.knn.kneighbors(X=test_word_vectorized, n_neighbors=n_neighbour_search)
        neighbors_task_numbers = neighbors[1][0]
        neighbors_distances = neighbors[0][0]
        neighbor_estimates = []
        for neighbor_i in range(len(neighbors_task_numbers)):
            neighbor_task_number = neighbors_task_numbers[neighbor_i]
            neighbor_distance = neighbors_distances[neighbor_i]
            if neighbor_distance > (np.sum(np.abs(test_word_vectorized)) + np.sum(np.abs(self.vect.getrow(neighbor_task_number)))):
                continue
            task_data = self.project_data.loc[neighbor_task_number]
            task_id = task_data['#']
            spent_time = task_data['TotalSpentWithUnbillable']
            task_title = task_data['Subject']
            task_developer = task_data['Assignee'] # @todo implement Developer identification
            task_unbillable = task_data['UnbillableTotal']
            neighbor_estimates.append({'id': task_id, 'title': task_title, 'time': spent_time, 'developer': task_developer, 'unbillable': task_unbillable})
        return neighbor_estimates

    def load_all_data(self, path_to_csv_files, session_length=10):
        files = os.scandir(path_to_csv_files)
        sites = dict()
        max_code = 1
        self.project_data = pd.DataFrame()
        for file in files:
            if (file.is_file()):
                filename = file.name
                print(filename)
                project_data = pd.read_csv(path_to_csv_files + '/' + filename, error_bad_lines=False)
                self.project_data = self.project_data.append(project_data, ignore_index=True)
    def process_unbillable(self):
        unbillable_tasks = self.project_data[self.project_data['Subject'].str.contains('#\d{1,6}', regex=True) & ((self.project_data['Project'] == '[DVB] DV Billable') | (self.project_data['Project'] == '[DV] Default Value'))]
        self.unbillable_tasks_data = {}
        for index, task in unbillable_tasks.iterrows():
            res = task
            task_subject = task['Subject']
            regexp = re.compile('#\d{1,6}')
            matches = regexp.search(task_subject)
            real_task_id = matches.group(0)
            real_task_id = real_task_id[1:]
            real_task_id = int(real_task_id)
            unbillable_spent = task['Spent time']
            unbillable_total_spent = task['Total spent time']
            self.unbillable_tasks_data[real_task_id] = [unbillable_spent, unbillable_total_spent]
    def get_unbillable_by_task_id(self, task_id):
        if task_id in self.unbillable_tasks_data.keys():
            return self.unbillable_tasks_data[task_id][0]
        else:
            return 0
    def get_total_unbillable_by_task_id(self, task_id):
        if task_id in self.unbillable_tasks_data.keys():
            return self.unbillable_tasks_data[task_id][1]
        else:
            return 0

    def set_unbillable_field(self):
        self.process_unbillable()
        self.project_data['Unbillable'] = self.project_data['#'].apply(self.get_unbillable_by_task_id)
        self.project_data['UnbillableTotal'] = self.project_data['#'].apply(self.get_total_unbillable_by_task_id)
        self.project_data['TotalSpentWithUnbillable'] = self.project_data['Total spent time'] + self.project_data['UnbillableTotal']
    def spent_time_hours(self, time):
        return int(time)

    def initialize_knn(self):
        vectorizer = CountVectorizer(stop_words='english')
        self.vect = vectorizer.fit_transform(self.project_data['Subject'])
        self.vocabulary = vectorizer.vocabulary_
        self.knn=KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric)
        y = self.project_data['Spent time'].apply(self.spent_time_hours)
        self.knn.fit(self.vect, y)

    def aggregate(self, tasks):
        tasks_records = self.project_data[self.project_data['#'].isin(tasks)]
        tasks_records = tasks_records['Spent time'] #with unbillable???
        tasks_records.dropna()
        random_hash = random.getrandbits(10)
        image_file = f'images/{random_hash}hist.png'
        fig = plt.figure()
        tasks_records.hist(bins=30)
        fig.savefig('/home/dmitry/Development/geekhubds/dv_tasks_project/estimates_webapp/static/' + image_file)
        task_statistics = {
            'mean': tasks_records.mean(),
            'min': tasks_records.min(),
            'max': tasks_records.max(),
            'std': tasks_records.std(),
            'gauss_image': image_file
        }

        return task_statistics
