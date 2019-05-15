import os

from task_search import TaskSearch
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
current_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = '/home/dmitriych/Documents/dv_tasks/data';
taskSearch = TaskSearch(current_directory, data_directory)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', n_neighbour=10)


@app.route('/tasks/', methods=['POST'])
def tasks():
    keywords = request.form.get('task_keywords', '')
    n_neighbours = int(request.form.get('n_neighbour', 10))
    similar_tasks = taskSearch.search(keywords, n_neighbours)
    return render_template('index.html', tasks=similar_tasks, search_query=keywords, n_neighbour=n_neighbours)


@app.route('/aggregate/', methods=['POST'])
def aggregate():
    print(request.form)
    tasks_requested_string = request.form.get('tasks_selected', '');
    tasks_requested = []
    if tasks_requested_string:
        tasks_requested = [int(i) for i in request.form.get('tasks_selected', '').split(',')]
    print(tasks_requested)
    similar_tasks_stats = taskSearch.aggregate(tasks_requested)
    return render_template('aggregate.html', tasks_stats=similar_tasks_stats)


if __name__ == '__main__':
    app.run()
