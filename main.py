from Facade import *
from utils import utils, LoadData

def start(args=None):
    print("\nPick your option to continue")
    print("-------------------------------------")
    print("1. AdaBoost")
    print("2. Decision Tree")
    print("3. Gradient Boosting Tree")
    print("4. KNN")
    print("5. Naive Bayes")
    print("6. Proactive Forest")
    print("7. Random Forest")
    print("8. Support Vector Machines")
    print("------------------------------------\n")

    option = '6' #input("Enter your option: ")
    
    print('Scenarios range from 0 to 13\n')
    e = '0' #input("Pick the scenery you want to use: ")

    if int(e) < 0 or int(e) > 13:
        print('Scenary is not valid, defaulting to 3...')
        e = '3'

    match option:
        case '1':
            # AdaBoost
            print("Using AdaBoost")
            print('Scenary: ' + e)
            model = fit_adaboost_process(esc=e)
            if model is not None:
                x_clasf, y_clasf = classification_process(model, "./database-preprosesing/smote/"+e+"/minmax/"+e+".minmax_smote.pickle", 250, 1000)
        case '2':
            # Decision Tree
            print("Using Decision Tree")
            print('Scenary: ' + e)
            model = fit_decision_tree_process(esc=e)
            if model is not None:
                x_clasf, y_clasf = classification_process(model, "./database-preprosesing/smote/"+e+"/minmax/"+e+".minmax_smote.pickle", 250, 1000)
        case '3':
            # Gradient Tree Boosting
            print("Using Gradient Tree Boosting")
            print('Scenary: ' + e)
            model = fit_gbt_process(esc=e)
            if model is not None:
                x_clasf, y_clasf = classification_process(model,"./database-preprosesing/smote/"+e+"/minmax/"+e+".minmax_smote.pickle", 250, 1000)
        case '4':
            # KNN
            print("Using KNN")
            print('Scenary: ' + e)
            model = fit_knn_process(esc=e)
            if model is not None:
                x_clasf, y_clasf = classification_process(model,"./database-preprosesing/smote/"+e+"/minmax/"+e+".minmax_smote.pickle", 250, 1000)
        case '5':
            # Naive Bayes
            print("Using Naive Bayes")
            print('Scenary: ' + e)
            model = fit_naive_bayes_process(esc=e)
            if model is not None:
                x_clasf, y_clasf = classification_process(model,"./database-preprosesing/smote/"+e+"/minmax/"+e+".minmax_smote.pickle", 250, 1000)
        case '6':
            # Proactive Forest
            print("Using Proactive Forest")
            print('Scenary: ' + e)
            model = fit_pf_process(esc=e)
            #if model is not None:
            #    x_clasf, y_clasf = classification_process(model, "./database-preprosesing/smote/"+e+"/minmax/"+e+".minmax_smote.pickle", 250, 1000)
            #    component_process(x_clasf, y_clasf, model)
        case '7':
            # Random Forest
            print("Using Random Forest")
            print('Scenary: ' + e)
            model = fit_random_forest_process(esc=e)
            if model is not None:
                x_clasf, y_clasf = classification_process(model, "./database-preprosesing/smote/"+e+"/minmax/"+e+".minmax_smote.pickle", 250, 1000)
        case '8':
            # Support Vector Machines
            print("Using Support Vector Machines")
            print('Scenary: ' + e)
            model = fit_support_vector_machine(esc=e)
            if model is not None:
                x_clasf, y_clasf = classification_process(model, "./database-preprosesing/smote/"+e+"/minmax/"+e+".minmax_smote.pickle", 250, 1000)
        case _:
            print("Error, classification model does not exists")

if __name__ == "__main__":
    start()
    
# if __name__ == "__main__":
#     e = 0
#     k = 0 
#     for i in range(10):
#         with open ('resultados escenario 0 v2 .txt', 'a') as f:
#             k += 1
#             print(f"Escenario {e} , iteracion {k} \n", file=f)
#         start()