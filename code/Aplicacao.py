######
# Antes de rodar a aplicação é necessário levantar o Served Model :
# mlflow models serve -m "runs:/7b3e7c7eea034a8ea712550a66b0fee5/model_decision_tree" --no-conda -p 8081
#####

##### Instalação das bibliotecas #####
import mlflow
import warnings
import pandas as pd
import requests
from sklearn.metrics import log_loss, f1_score

##### Especificar o Experimeto #####
warnings.filterwarnings('ignore')
mlflow.set_experiment("Kobe")

###### Carrga da base de produção #####
df = pd.read_parquet("../data/raw/dataset_kobe_prod.parquet")

##### Tratamento dos dados com remoção dos dados ausentes  e filtro por 'shot_type' #####
df = df.dropna(subset=['shot_made_flag'])
df_filtered = df[['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']]

#print(df)

##### Transformar os dados do dataframe em json
dados_json = df_filtered.to_json(orient='split')

##### Rquisição da API do MLFlow
responseMLFlow = requests.post(
    'http://127.0.0.1:8081/invocations',
    headers={'Content-Type':'application/json'},
    json={
        "dataframe_split": 
        {
            "columns": df_filtered.columns.tolist(),
            "data": df_filtered.values.tolist()
        }
    }
)

# Verificar a resposta for válida = 200
if responseMLFlow.status_code == 200:
    print("[OK] Dados recebidos do MLFlow")
    predicoes = responseMLFlow.json()
    
    #print(predicoes)

    # Calcula as métricas
    log_loss = log_loss(df['shot_made_flag'].values, predicoes['predictions'])
    f1_score = f1_score(df['shot_made_flag'].values, predicoes['predictions'])
    
    print(f"Predição Log Loss: {log_loss}")
    print(f"Predição F1 Score: {f1_score}")

    
    ##### Registro da métricas no MLFlow
    with mlflow.start_run(run_name="PipelineAplicacao"):
        # Registro das métricas
        mlflow.log_metric("log_loss", log_loss)
        mlflow.log_metric("f1_score", f1_score)

        mlflow.set_tag('model', 'Decision Tree Classifier')
        
        
else:
    print("[ERRO] Reason:", responseMLFlow.text)