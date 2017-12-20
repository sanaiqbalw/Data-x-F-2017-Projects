import multiprocessing
from tabulate import tabulate

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from embedding import load_all, load_word2vec, MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer

# TODO: try fasttext
if __name__ == '__main__':
	# load data
	data = load_all()
	model = load_word2vec()
	w2v = dict(zip(model.wv.index2word, model.wv.syn0))
	workers = multiprocessing.cpu_count()

	# start with the classics - naive bayes of the multinomial and bernoulli varieties
	# with either pure counts or tfidf features
	mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
			("multinomial nb", MultinomialNB())])
	bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
			("bernoulli nb", BernoulliNB())])
	mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
			("multinomial nb", MultinomialNB())])
	bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
			("bernoulli nb", BernoulliNB())])
	# SVM - which is supposed to be more or less state of the art
	# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
	svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
			("linear svc", SVC(kernel="linear"))])
	svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
			("linear svc", SVC(kernel="linear"))])

	config = {
	    "max_depth": 3,#3,
	    "n_estimators": 200,
	    "n_jobs": workers,
	}

	# Extra Trees classifier is almost universally great; stack it with our embeddings
	etree = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
			("extra_trees", ExtraTreesClassifier(**config))])
	etree_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
			("extra_trees", ExtraTreesClassifier(**config))])
	etree_w2v = Pipeline([("word2vec_vectorizer", MeanEmbeddingVectorizer(w2v)),
			("extra_trees", ExtraTreesClassifier(**config))])
	etree_w2v_tfidf = Pipeline([("word2vec_vectorizer", TfidfEmbeddingVectorizer(w2v)),
			("extra_trees", ExtraTreesClassifier(**config))])

	params = {
	    "learning_rate": 0.1,#0.15,
	    "max_depth": 5,#6,
	    "n_estimators": 200,
	    "nthread": workers,
	    "min_child_weight": 0,
	}

	xgb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
			('xbgoost', XGBClassifier(**params))])
	xgb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
			('xbgoost', XGBClassifier(**params))])
	xgb_w2v = Pipeline([("word2vec_vectorizer", MeanEmbeddingVectorizer(w2v)),
			('xbgoost', XGBClassifier(**params))])
	xgb_w2v_tfidf = Pipeline([("word2vec_vectorizer", TfidfEmbeddingVectorizer(w2v)),
			('xbgoost', XGBClassifier(**params))])

	# benchmark different models
	all_models = [
	    # classic models
	    ("mult_nb", mult_nb),
	    ("mult_nb_tfidf", mult_nb_tfidf),
	    ("bern_nb", bern_nb),
	    ("bern_nb_tfidf", bern_nb_tfidf),
	    ("svc", svc),
	    ("svc_tfidf", svc_tfidf),
	    # tree models with custom word2vec
	    ("etree", etree),
	    ("etree_tfidf", etree_tfidf),
	    ("etree_w2v", etree_w2v),
	    ("etree_w2v_tfidf", etree_w2v_tfidf),
	    ("xgb", xgb),
	    ("xgb_tfidf", xgb_tfidf),
	    ("xgb_w2v", xgb_w2v),
	    ("xgb_w2v_tfidf", xgb_w2v_tfidf),
	]

	# sort by descending order
	scores = sorted([(name, accuracy_score(data['y'], model.predict(data['X'])))
	        for name, model in all_models], key=lambda x: -x[1])

	print(tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
