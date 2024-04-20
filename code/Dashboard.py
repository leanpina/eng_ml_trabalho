##### Inicialização das Bibliotecas #####
import streamlit as st
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

nomeExperimento = "Kobe"

##### Set do experimento no MLFlow #####
mlflow.set_experiment(nomeExperimento)

###### Título do dashboard ######
st.title('Resultados da Modelagem')

##### Seção de experimento #####
st.header('Experimento: ' + nomeExperimento)

###### Recupera os dados do experimento #####
experiment = mlflow.get_experiment_by_name(nomeExperimento)
experiment_id = experiment.experiment_id
runs = mlflow.search_runs(experiment_ids=[experiment_id])
runs['experiment_name'] = nomeExperimento
runs = runs[runs['experiment_name'] == nomeExperimento]

#print(runs.columns)

st.write('ID no MLFlow:', experiment_id)
st.write(''.ljust(10, '-'))

st.header('Runs disponíveis no Experimento')
st.write('Resumo:', runs)
st.write(''.ljust(10, '-'))

##### Exibição das métricas #####
st.header('Modelo / Métricas')
runPipelineAplicacao = runs[runs['tags.mlflow.runName'] == 'PipelineAplicacao']

runIdPipelineAplicacao = runPipelineAplicacao['run_id'].values[0]
st.markdown('<b>ID:</b> ' + runIdPipelineAplicacao, unsafe_allow_html=True)
st.markdown(f"<b>Modelo:</b> {runPipelineAplicacao['tags.model'].values[0]}", unsafe_allow_html=True)

st.markdown(f"<b>Métrica - F1 Score:</b> {runPipelineAplicacao['metrics.f1_score'].values[0]}", unsafe_allow_html=True)
st.markdown(f"<b>Métrica - Log Loss:</b> {runPipelineAplicacao['metrics.log_loss'].values[0]}", unsafe_allow_html=True)

###### Recupera o local do arquivo final #####
uriFile = runPipelineAplicacao['artifact_uri'].values[0] + "/result.parquet"


