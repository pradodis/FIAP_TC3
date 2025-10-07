# Projeto - Pasta "entrega"

## Visão Geral
A pasta "entrega" contém os scripts, modelos e artefatos finais.  
O tratamento (limpeza, transformação, feature engineering e split) é feito exclusivamente no notebook indicado (abrir em Jupyter ou Colab).  
Após o tratamento, os arquivos processados alimentam o pipeline de treino local (Python) ou remoto (Google Colab).

## Dependências
Python 3.10+ recomendado.

Dependências principais:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- tqdm
- ipykernel

Opcionais (apenas se usados):
- plotly (visualização interativa)
- tensorflow ou torch (modelos deep learning)
- scikit-optimize ou optuna (otimização de hiperparâmetros)

## Instalação (Ambiente Local)
python -m venv .venv
(Windows) .venv\Scripts\activate
(Linux/Mac) source .venv/bin/activate
pip install --upgrade pip

Se existir requirements.txt:
pip install -r requirements.txt

Ou manual:
pip install pandas numpy scikit-learn matplotlib seaborn joblib tqdm ipykernel

Registrar kernel (opcional):
python -m ipykernel install --user --name fiap_tc3

## Uso do Notebook (Tratamento de Dados)
1. Abrir o notebook principal.
2. Executar todas as células até a geração dos datasets tratados (ex: dados_treinamento.csv / dados_teste.csv).
3. Confirmar diretórios de saída (ex: /entrega/data/processed).

## Treinamento Local (Python)
1. Garantir que os dados tratados foram gerados pelo notebook.
2. Rodar script de treino (exemplo):
   python treino.py --input entrega/data/processed/dados_treinamento.csv --modelo saida/modelo.pkl
3. Avaliar métricas geradas (logs ou relatório em /entrega/metrics).

## Treinamento Remoto (Google Colab)
1. Fazer upload do notebook e (se necessário) dos dados tratados ou montar Google Drive.
2. Instalar dependências no início do notebook:
   !pip install pandas numpy scikit-learn matplotlib seaborn joblib tqdm
3. Opcional: ativar GPU (Runtime > Change runtime type > GPU).
4. Executar células de treino e salvar modelo em drive.

## Estrutura Sugerida
entrega/
  data/
    raw/            (dados brutos)
    processed/      (dados tratados gerados no notebook)
  models/           (modelos salvos)
  metrics/          (relatórios / logs)
  scripts/          (scripts auxiliares)
  notebook.ipynb    (tratamento + experimentos)

## Atualização de Dependências
pip freeze > requirements.txt

## Reprodutibilidade
- Fixar versão de pacotes críticos (pandas, scikit-learn).
- Definir seeds em scripts (ex: numpy.random.seed, random.seed).

## Suporte
Executar primeiro o notebook para garantir consistência dos dados antes de qualquer treino local ou remoto.
