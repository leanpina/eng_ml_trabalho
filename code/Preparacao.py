from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
import mlflow
import pandas as pd
import warnings

##### Declaração do Experimento #####
warnings.filterwarnings('ignore')

# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_tracking_uri("sqlite:///mlruns.db")
# mlflow.set_tracking_uri("https://localhost:5000")
mlflow.set_experiment("Kobe")

# experiment_name = 'Kobe'
# experiment = mlflow.get_experiment_by_name(experiment_name)
# if experiment is None:
#     experiment_id = mlflow.create_experiment(experiment_name)
#     experiment = mlflow.get_experiment(experiment_id)
# experiment_id = experiment.experiment_id''
#####

##### Parâmetros #####

TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
SEED = 42

##### Carga dos Dados #####
df = pd.read_csv("../data/raw/data.csv")

##### Tratamento dos dados com remoção dos dados ausentes  e filtro por 'shot_type' #####
df_filtered = df.dropna(subset='shot_made_flag')
df_filtered = df_filtered[df_filtered['shot_type'] == '2PT Field Goal']
df_filtered = df_filtered[['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']]

##### Comparando o tamanho da base original e a filtradas #####
print(f"Tamanho do DataFrame Original: {len(df)}")
print(f"Tamanho do DataFrame Filtrado: {len(df_filtered)}")

##### Salvando o dataset filtrado #####
df_filtered.to_parquet("../data/processed/data_filtered.parquet")

#####  Separando os dados nos datasets de treino (80%) e teste (20%) #####
X_train, X_test, y_train, y_test = train_test_split(
    df_filtered.drop('shot_made_flag', axis=1), 
    df_filtered['shot_made_flag'], 
    test_size=TEST_SIZE, 
    stratify=df_filtered['shot_made_flag'], 
    random_state=SEED
)

##### Salvando os datasets de treino e teste #####
X_train.join(y_train).to_parquet("../data/processed/base_train.parquet")
X_test.join(y_test).to_parquet("../data/processed/base_test.parquet")

##### Iniciando uma run do MlFlow para o pipeline de preparação de dados #####
with mlflow.start_run(run_name='Preparacao'):
    mlflow.log_param("teste_percentual", TEST_SIZE)
    mlflow.log_metric("base_treino_tamanho", len(X_train))
    mlflow.log_metric("base_teste_tamanho", len(X_test))