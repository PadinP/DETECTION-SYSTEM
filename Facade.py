import numpy as np
import pandas
import pickle as pck
import warnings

from sklearn.metrics import accuracy_score

from utils import utils, LoadData
from models.proactive_forest_classifier import PFClassifier
from models.adaboost import ABClassifier
from models.bayessian_gaussian_mixture import *
from models.decision_tree import *
from models import GradientClassifier
from models.random_forest_classifier import *
from models.knn_classifier import *
from models.naive_bayes import *
from models.support_vector_machine import *
from models.kmeans_classifer import *
from models.bayessian_gaussian_mixture import *


warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def show(lista):
    uniq, count = np.unique(lista, return_counts=True)
    try:
        print("Number of human users in classification:", count[0])
    except:
        print("Number of human users in classification:", 0)
    try:
        print("Number of bots users in classification:", count[1])
    except:
        print("Number of bots users in classification:", 0)


# Fit Functions
def fit_pf_process(esc="3"):
    pfc_model = None
    pfc = PFClassifier(escenario=esc, k=5)
    train, test = pfc.prepareData()  # preprocesamiento
    if (len(train) != 0):
        pfc_model, avg, max_score = utils.cross_validation_train(pfc, train, test)

    # file = open ("./files/Para_tesis_model_grande.pckl","ab+")
    # pck.dump (pfc, file)
    # file.close ()
    return pfc_model


def fit_adaboost_process(esc='3'):
    abc_model = None
    abc = ABClassifier(escenario=esc, k=5)
    train, test = abc.prepareData()
    if len(train) != 0:
        abc_model, avg, max_score = utils.cross_validation_train(abc, train, test)
    return abc_model


def fit_decision_tree_process(esc='3'):
    dec_model = None
    dec = DecisionTree(escenario=esc, test_size=0.2)
    train, test = dec.prepareData2()
    if len(train) != 0:
        dec_model, avg, max_score = utils.cross_validation_train(dec, train, test)
    return dec_model


def fit_knn_process(esc='3'):
    knn_model = None
    knn = KNNClassifier(n_neighbors=5, escenario=esc)
    train, test = knn.prepareData2()
    if len(train) != 0:
        knn_model, avg, max_score = utils.cross_validation_train(knn, train, test)
    return knn_model


def fit_naive_bayes_process(esc="3"):
    nav_model = None
    nav = NaiveBayes(escenario=esc)
    train, test = nav.prepareData2()
    if len(train) != 0:
        nav_model, avg, max_score = utils.cross_validation_train(nav, train, test)
    return nav_model


def fit_gbt_process(esc='3'):
    gbdt = GradientClassifier.GClassifier(n_estimators=250, escenario=esc, k=5)
    train, test = gbdt.prepareData()
    gbdt_model, avg, max_score = utils.cross_validation_train(gbdt, train, test) if train else (None, None, None)
    return gbdt_model


def fit_random_forest_process(esc='3'):
    ran_model = None
    ran = RFClassifier(escenario=esc, k=5)
    train, test = ran.prepareData()
    if len(train) != 0:
        ran_model, avg, max_score = utils.cross_validation_train(ran, train, test)
    return ran_model


def fit_support_vector_machine(esc='3'):
    sup_model = None
    sup = SVMClassifier(kernel='linear', random_state=0, escenario=esc)
    train, test = sup.prepareData2()
    if len(train) != 0:
        sup_model, avg, max_score = utils.cross_validation_train(sup, train, test)
    return sup_model

# Fit Functions End


def classification_process(model, data, since=0, until=5000):
    x, test = utils.load_and_divide(data, since, until)
    y = model.predict(x)
    show(y)
    # print(model.score(x,test)*100)
    return x, y


def component_process(x, y, model):
    comp = model.start_component(x, y, model)
    comp.run_charact(x, y)


