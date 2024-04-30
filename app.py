import json

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
from sklearn.manifold import TSNE

import utils
from model.models import RGCN

app = Flask(__name__)

MODEL_FOLDER = 'model'
SCRIPT_FOLDER = 'script'
MODEL_EXTENSIONS = {'pt', 'pth', 'bin', 'onnx', 't7', 'pkl'}
SCRIPT_EXTENSIONS = {'py'}

CORS(app, origins='*', resources=r'/*')

app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['SCRIPT_FOLDER'] = SCRIPT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['SUBGRAPH'] = {}
app.config['ALL_TRIPLETS'] = []


@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'file error'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'no file name'})

    if os.path.exists(os.path.join(app.config['MODEL_FOLDER'], file.filename)):
        os.remove(os.path.join(app.config['MODEL_FOLDER'], file.filename))

    if file and utils.allowed_file(file.filename, ext_list=MODEL_EXTENSIONS):
        print(file.filename)
        file.save(os.path.join(app.config['MODEL_FOLDER'], file.filename))
        app.config['MODEL_NAME'] = file.filename
        return jsonify({'message': 'upload successfully'})
    else:
        return jsonify({'error': 'invalid file'})


@app.route('/upload_script', methods=['POST'])
def upload_script():
    if 'file' not in request.files:
        return jsonify({'error': 'file error'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'no file name'})

    if os.path.exists(os.path.join(app.config['SCRIPT_FOLDER'], file.filename)):
        os.remove(os.path.join(app.config['SCRIPT_FOLDER'], file.filename))

    if file and utils.allowed_file(file.filename, ext_list=SCRIPT_EXTENSIONS):
        print(file.filename)
        file.save(os.path.join(app.config['SCRIPT_FOLDER'], file.filename))
        app.config['SCRIPT_NAME'] = file.filename
        return jsonify({'message': 'upload successfully'})
    else:
        return jsonify({'error': 'invalid file'})


@app.route('/select', methods=['GET'])
def select_points():
    id2entity, entity2id, id2relation, relation2id, all_triplets = utils.load_data('./data/wn18')
    app.config['ALL_TRIPLETS'] = all_triplets
    edge_type_count = utils.count_edge_types(all_triplets)

    with open('model/emb2d.json', 'r') as file:
        embedding_2d = json.load(file)

    print(edge_type_count)

    return jsonify(
        {
            'id2entity': id2entity,
            'id2relation': id2relation,
            'embedding': embedding_2d,
            'edge_type_count': edge_type_count
        }
    )


@app.route('/filter_by_link_type', methods=['POST'])
def filter_by_link_type():
    if len(app.config['ALL_TRIPLETS']) == 0:
        id2entity, entity2id, id2relation, relation2id, all_triplets = utils.load_data('./data/wn18')
    else:
        all_triplets = app.config['ALL_TRIPLETS']
    filter_links = request.form.get('filter_links').split(',')
    filter_links = [int(linkT) for linkT in filter_links]
    filtered_triplets = [(s, r, o) for s, r, o in all_triplets if r in filter_links]
    entities = set(s for s, _, o in filtered_triplets) | set(o for _, _, o in filtered_triplets)
    entities_list = list(entities)

    print(entities_list)

    return jsonify(
        {
            'filtered_index': entities_list
        }
    )


@app.route('/vis', methods=['GET'])
def visualization():
    id2entity, entity2id, id2relation, relation2id, all_triplets = utils.load_data('data/wn18')
    adj_list = utils.triples_to_adj(all_triplets)

    with open('model/emb2d.json', 'r') as file:
        embedding_2d = json.load(file)

    subgraph_adj = utils.get_k_hop_subgraph(adj_list, nodes=[1], k=3)
    in_degree, out_degree = utils.calculate_in_out_degree(subgraph_adj)

    node_list = in_degree.keys()
    embedding_2d = {i: embedding_2d[i] for i in node_list}
    app.config['SUBGRAPH'] = subgraph_adj
    return jsonify(
        {
            'id2entity': id2entity,
            'id2relation': id2relation,
            'in_degree': json.dumps(in_degree),
            'out_degree': json.dumps(out_degree),
            'embedding': embedding_2d,
            'graph': json.dumps(subgraph_adj)
        }
    )


@app.route('/pathfind', methods=['POST'])
def get_path():
    start = request.form.get('start')
    end = request.form.get('end')
    metapath = request.form.get('metapath').split(',')
    metapath = [int(item) for item in metapath]
    paths = utils.find_paths(app.config['SUBGRAPH'], int(start), int(end), 4)
    return jsonify({'path': paths})


if __name__ == '__main__':
    app.run()
