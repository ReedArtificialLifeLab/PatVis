# TODO:
# - remove references to bh_sne, we no longer install it with this package
# - generalize code to plot tsne visualizations of any embedding with any color scheme

# utilities for visualizing word2vec clusters.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim import models,corpora
from alife.visualize.discrete_color import discrete_color_scheme
from collections import defaultdict
import csv
from itertools import islice
from alife.visualize.util import cosine_dist
import time
from pymongo import MongoClient
from getpass import getpass

# def embedding_fig(w2v_model, kmeans_model, n=500, embed_style = 'tsne', savefn = None, show=False):
#     """
#     Save or show a matplotlib figure, displaying word embeddings colored
#     according to cluster assignment. The parameter n gives the number of
#     words to show, in decreasing order of frequency in the dataset.

#     """
#     freqs = {x: y.count for x,y in list(w2v_model.vocab.items())}
#     words,word_vecs = w2v_model.index2word, w2v_model.syn0
#     srtd_indices, srtd_words = list(zip(*sorted(list(enumerate(words)), key = lambda x: freqs[x[1]], reverse=True)))
#     srtd_vecs = np.array([word_vecs[i] for i in srtd_indices])
#     subset_words, subset_vecs = srtd_words[:n], srtd_vecs[:n]
#     # map cluster assignment to integer.
#     subset_clusters = np.array([int(kmeans_model.predict(x)) for x in subset_vecs])
#     unique_clusters = set(subset_clusters)
#     print("number of unique clusters in top {} words: {}.".format(n, len(unique_clusters)))
#     colormap = discrete_color_scheme(len(unique_clusters))
#     int_to_color = {idx: colormap[i] for (i,idx) in enumerate(list(unique_clusters))}
#     subset_colors = [int_to_color[i] for i in subset_clusters]
#     if embed_style == 'pca':
#         pca = PCA(n_components=2)
#         pca_embeddings = pca.fit_transform(subset_vecs)
#         embed_xs, embed_ys = pca_embeddings.transpose()
#     elif embed_style == 'tsne':
#         tsne_embeddings = bh_sne(np.asarray(subset_vecs, dtype=np.float64),d=2, perplexity=10)
#         embed_xs, embed_ys = tsne_embeddings.transpose()
#     fig = plt.figure()
#     fig.set_size_inches(50,28.4)
#     ax = fig.add_subplot(111)
#     ax.scatter(embed_xs, embed_ys, color = subset_colors, s=1000)
#     for (x,y,word) in zip(embed_xs, embed_ys, subset_words):
#         ax.annotate(word, xy=(x,y), textcoords='offset points', fontsize=20)
#     plt.title('{} 2d word embeddings'.format(embed_style))
#     if savefn is not None:
#         plt.savefig(savefn, dpi=120)
#     if show:
#         plt.show()

# def embedding_fig_nclusters(w2v_model, kmeans_model, n=[1,2,3,4,5], n_nearest_words=100, embed_style = 'tsne', savefn = None, show=False):
#     """
#     Save or show a matplotlib figure, displaying word embeddings colored
#     according to cluster assignment. The parameter n gives the number of
#     words to show, in decreasing order of frequency in the dataset.

#     """
#     freqs = {x: y.count for x,y in list(w2v_model.vocab.items())}
#     words,word_vecs = w2v_model.index2word, w2v_model.syn0
#     clusterdict = defaultdict(list)
#     for w,v in zip(words,word_vecs):
#         cluster = int(kmeans_model.predict(v))
#         if cluster in n:
#             clusterdict[cluster].append((w,v))

#     print("Pruning...")
#     words,vecs,clusters = [],[],[]
#     for cluster,tuples in list(clusterdict.items()):
#         sorted_tuples = sorted(tuples,key=lambda x: cosine_dist(kmeans_model.cluster_centers_[cluster],x[1]))
#         clusters.extend([ cluster for i in range(len(sorted_tuples))][:n_nearest_words])
#         words.extend([ x[0] for x in sorted_tuples][:n_nearest_words])
#         vecs.extend([ x[1] for x in sorted_tuples][:n_nearest_words])

#     # map cluster assignment to integer.
#     unique_clusters = set(clusters)
#     print("number of unique clusters in top {} words: {}.".format(n, len(unique_clusters)))
#     colormap = discrete_color_scheme(len(unique_clusters))
#     int_to_color = {idx: colormap[i] for (i,idx) in enumerate(list(unique_clusters))}
#     subset_colors = [int_to_color[i] for i in clusters]
#     if embed_style == 'pca':
#         pca = PCA(n_components=2)
#         pca_embeddings = pca.fit_transform(vecs)
#         embed_xs, embed_ys = pca_embeddings.transpose()
#     elif embed_style == 'tsne':
#         tsne_embeddings = bh_sne(np.asarray(vecs, dtype=np.float64),d=2, perplexity=10, theta=0)
#         embed_xs, embed_ys = tsne_embeddings.transpose()
#     fig = plt.figure()
#     fig.set_size_inches(50,28.4)
#     ax = fig.add_subplot(111)
#     ax.scatter(embed_xs, embed_ys, color = subset_colors, s=100)
#     period = n_nearest_words / 10
#     numwords = 0
#     for (x,y,word) in zip(embed_xs, embed_ys, words):
#         if numwords % period == 0:
#             ax.annotate(word, xy=(x,y), textcoords='offset points', fontsize=20)
#         numwords += 1
#     plt.title('{} 2d word embeddings'.format(embed_style))
#     if savefn is not None:
#         plt.savefig(savefn, dpi=120)
#     if show:
#         plt.show()

# def embedding_fig_term_buckets(w2v_model, kmeans_model, term_buckets_file, embed_style = 'tsne', savefn = None, show=False):
#     """
#     Save or show a matplotlib figure, displaying word embeddings colored
#     according to cluster assignment. The parameter n gives the number of
#     words to show, in decreasing order of frequency in the dataset.

#     """
#     with open(term_buckets_file,"rb") as infile:
#         reader = csv.DictReader(infile)
#         term_buckets = defaultdict(list)
#         for row in reader:
#             for (k,v) in list(row.items()):
#                 if k in [ 'chemical','medical','computers','electrical','mechanical','null' ]:
#                     term_buckets[k].append(v)

#     stems = {}
#     words,word_vecs = w2v_model.index2word, w2v_model.syn0
#     for w,v in zip(words,word_vecs):
#         stem = stemmer(w)
#         for name,bucket in list(term_buckets.items()):
#             if stem in bucket:
#                     b = name
#                     normalized = v/np.linalg.norm(v)
#                     if stem in stems:
#                         stems[stem][0].append(normalized)
#                     else:
#                         stems[stem] = [[normalized],b]
#     chosen_words, chosen_vecs, chosen_buckets = list(zip( *[ (w,np.mean(info[0],axis=0),info[1]) for w,info in list(stems.items()) ] ))

#     # map cluster assignment to integer.
#     colormap = discrete_color_scheme(len(list(term_buckets.keys())))
#     bucket_to_color = { bucket : colormap[i] for (i,bucket) in enumerate(list(term_buckets.keys())) if bucket != 'null'}
#     bucket_to_color['null'] = [0,0,0]
#     colors = [ bucket_to_color[bucket] for bucket in chosen_buckets ]
#     if embed_style == 'pca':
#         pca = PCA(n_components=2)
#         pca_embeddings = pca.fit_transform(chosen_vecs)
#         embed_xs, embed_ys = pca_embeddings.transpose()
#     elif embed_style == 'tsne':
#         tsne_embeddings = bh_sne(np.asarray(chosen_vecs, dtype=np.float64),d=2, perplexity=10, theta=0)
#         embed_xs, embed_ys = tsne_embeddings.transpose()
#     fig = plt.figure()
#     fig.set_size_inches(50,28.4)
#     ax = fig.add_subplot(111)
#     ax.scatter(embed_xs, embed_ys, color = colors, s=1000)
#     for (x,y,word) in zip(embed_xs, embed_ys, chosen_words):
#         ax.annotate(word, xy=(x,y), textcoords='data', fontsize=20)
#     plt.title('{} 2d word embeddings'.format(embed_style))
#     if savefn is not None:
#         plt.savefig(savefn, dpi=120)
#     if show:
#         plt.show()

def embedding_fig_by_category(d2v_model, n=500, embed_style = 'tsne', savefn = None, show=False):
    """
    Save or show a matplotlib figure, displaying word embeddings colored
    according to cluster assignment. The parameter n gives the number of
    words to show, in decreasing order of frequency in the dataset.

    """
    conn = MongoClient("localhost",27017)
    conn.the_database.authenticate(input('Please Enter Your DB Username:'), getpass('Please Enter Your DB Password:'), source='admin')
    db = conn['patents']
    pnos = []
    print("Reading File: ")
    with open("/Users/Shared/d2v model and pno list/abstracts_1976-2014_0010001_pnos.txt",'r') as pnofile:
        reader = csv.reader(pnofile)
        pnos = [ int(pno[0]) for pno in reader ]
        
    categories = []
    nummiss = 0
    categories_found = 0
    print("Fetching Categories: ")
    for pno in pnos:
        db_doc = db.patns.find_one({ 'pno' : pno, 'nber_main_category' :  { '$exists' : True } }, { 'nber_main_category' : 1 })
        if db_doc != None:
            categories.append(db_doc['nber_main_category'])
            categories_found += 1
        else:
            categories.append(None)
            nummiss += 1
        if categories_found == n:
            break

    pnos = [ pno for pno, category in zip(pnos,categories) if category != None ]
    print("Number of Patents skipped: " + str(nummiss))
    print("Fitting Model: ")
    
    X = np.asarray( [ vec for vec, category in zip(d2v_model.docvecs,categories) if category != None ], dtype=np.float64)
    categories = filter(lambda x: x != None, categories)
    
    model = TSNE()
    np.set_printoptions(suppress=True)
    model.fit_transform(X)
    
    fig = plt.figure()
    fig.set_size_inches(50,28.4)
    
    colormap = discrete_color_scheme(6)
    colors = [ colormap[category-1] for category in categories ]
    plt.legend(handles=[ mpatches.Patch(color=colormap[i], label=str(i+1)) for i in range(6) ])
    
    embed_xs = X[:, 0]
    embed_ys = X[:, 1]
    if n <= 100:
        for (x,y,pno) in zip(embed_xs, embed_ys, pnos):
            plt.annotate(pno, xy=(x,y), textcoords='data', fontsize=20)
        
    plt.scatter(embed_xs, embed_ys, c=colors, s=100)
    plt.savefig(str(n) + ".pdf")

def test():
    data_dir = '/Users/Shared/d2v model and pno list'
    docvec_fn = '/'.join([data_dir, 'model_1976-2014_dim=300_min=50_win=5_0010001'])
    kmeans_fn = '/'.join([data_dir, 'kmeans200.pkl'])
    d2v = models.doc2vec.Doc2Vec.load(docvec_fn)
    embedding_fig_by_category(d2v, n = 100)
    embedding_fig_by_category(d2v, n = 500)
    embedding_fig_by_category(d2v, n = 1000)
    embedding_fig_by_category(d2v, n = 5000)
    embedding_fig_by_category(d2v, n = 10000)
    embedding_fig_by_category(d2v, n = 50000)
    embedding_fig_by_category(d2v, n = 100000)
    embedding_fig_by_category(d2v, n = 1000000)





if __name__ == '__main__':
    test()
