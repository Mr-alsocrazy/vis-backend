import json

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import networkx as nx

import utils
import torch
import numpy as np
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

    graph = nx.DiGraph()
    for s, r, o in all_triplets:
        graph.add_edge(s, o)

    pagerank = nx.pagerank(graph, alpha=0.85)
    pagerank = sorted(pagerank.items(), key=lambda item: -item[1])
    pagerank = [{'node': node, 'pagerank': round(pagerank, 4)} for node, pagerank in pagerank]

    degree_centrality = nx.degree_centrality(graph)
    degree_centrality = sorted(degree_centrality.items(), key=lambda item: -item[1])
    degree_centrality = [{'node': node, 'degree_centrality': round(dc, 4)} for node, dc in degree_centrality]

    return jsonify(
        {
            'id2entity': id2entity,
            'id2relation': id2relation,
            'embedding': embedding_2d,
            'edge_type_count': edge_type_count,
            'pagerank': pagerank,
            'degree_centrality': degree_centrality,
            'pagerank_min': pagerank[len(pagerank) - 1]['pagerank'],
            'pagerank_max': pagerank[0]['pagerank'],
            'degree_centrality_min': degree_centrality[len(degree_centrality) - 1]['degree_centrality'],
            'degree_centrality_max': degree_centrality[0]['degree_centrality']
        }
    )


@app.route('/selection', methods=['POST'])
def selection():
    nodes_to_render = request.form.get('selection').split(',')
    nodes_to_render = [int(node) for node in nodes_to_render]
    app.config['nodes_to_render'] = nodes_to_render
    return jsonify({
        'message': 200
    })


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

    subgraph_adj = utils.get_k_hop_subgraph(adj_list, nodes=app.config['nodes_to_render'], k=3)
    in_degree, out_degree = utils.calculate_in_out_degree(subgraph_adj)

    print(len(subgraph_adj.keys()))

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
            'graph': json.dumps(subgraph_adj),
            'chosen': app.config['nodes_to_render']
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


@app.route('/predict', methods=['POST'])
def predict():
    start = int(request.form.get('start'))
    end = int(request.form.get('end'))
    id2entity, entity2id, id2relation, relation2id, all_triplets = utils.load_data('./data/wn18')
    test_graph = utils.build_test_graph(len(entity2id), len(relation2id), torch.tensor(np.array(all_triplets)))

    model = RGCN(len(entity2id), len(relation2id), num_bases=4, dropout=0.2)
    model.load_state_dict(torch.load('model/best_mrr_model.pth')['state_dict'])

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    relation_embedding = model.relation_embedding

    result = utils.predict_link(entity_embedding, relation_embedding, start, end, len(relation2id))

    result = [{'relation': relation, 'prob': prob} for relation, prob in result]
    print(result)

    return jsonify(
        {
            'result': result
        }
    )


if __name__ == '__main__':
    app.run()
