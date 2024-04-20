##### Importação das Bibliotecas #####
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import setup, create_model, predict_model, save_model, plot_model
import warnings
import pandas as pd
import mlflow

##### LOcal de Gravação dos modelos #####
warnings.filterwarnings('ignore')
rlSaveModelLocal = '../models/rl_final_model'
dtSaveModelLocal = '../models/dt_final_model'

##### Carga dos dados de Treino & Teste #####
train_data = pd.read_parquet("../data/processed/base_train.parquet")
test_data = pd.read_parquet("../data/processed/base_test.parquet")

##### Declaração do Modelo #####
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Kobe")

# experiment_name = 'Kobe'
# experiment = mlflow.get_experiment_by_name(experiment_name)
# if experiment is None:
#     experiment_id = mlflow.create_experiment(experiment_name)
#     experiment = mlflow.get_experiment(experiment_id)
# experiment_id = experiment.experiment_id

with mlflow.start_run(run_name='Treinamento'):
    #Configurando o setup do PyCaret
    setup(data=train_data, target='shot_made_flag')

    #### Regressão Logística #####
    logistic_model = create_model('lr')
    # Faz previsões no conjunto de teste e calcula o log loss e F1 Score
    logistic_predictions = predict_model(logistic_model, data=test_data)
    #Salva as metricas no MLFlow
    mlflow.log_metric("logistic_regression_log_loss", log_loss(test_data['shot_made_flag'], logistic_predictions['prediction_score']))
    mlflow.log_metric("logistic_regression_f1_score", f1_score(test_data['shot_made_flag'], logistic_predictions['prediction_label']))
    # Tags no MLFlow
    mlflow.set_tag('model', 'Logistic Regression')
    mlflow.set_tag('algorithm', 'PyCaret')
    # Registro do modelo no MLFlow
    mlflow.sklearn.log_model(logistic_model, "model_logistic_regression")
    # Salvar o modelo 
    save_model(logistic_model, rlSaveModelLocal)
    # Salvar o arquivo pickle 
    mlflow.log_artifact(rlSaveModelLocal + '.pkl')

    ##### Decision Tree #####
    tree_model = create_model('dt')
     # Faz previsões no conjunto de teste e calcula o log loss e F1 Score
    tree_predictions = predict_model(tree_model, data=test_data)
    # Salvar as metricas no MLFlow
    mlflow.log_metric("decision_tree_log_loss", log_loss(test_data['shot_made_flag'], tree_predictions['prediction_score']))
    mlflow.log_metric("decision_tree_f1_score", f1_score(test_data['shot_made_flag'], tree_predictions['prediction_label']))
    # Tags no MLFlow
    mlflow.set_tag('model', 'Decision Tree Classifier')
    mlflow.set_tag('algorithm', 'PyCaret')
    # Registrar o modelo no MLFlow
    mlflow.sklearn.log_model(tree_model, "model_decision_tree")
    # Salvar o modelo 
    save_model(tree_model, dtSaveModelLocal)
    # Salvar o arquivo pickle 
    mlflow.log_artifact(dtSaveModelLocal + '.pkl')
