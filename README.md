# House Price Prediction (Machine Learning)

Projeto de Machine Learning para previsão de preços de imóveis utilizando a base de dados de Boston.

---

## Sobre o projeto

Este projeto tem como objetivo analisar dados imobiliários e construir modelos preditivos para estimar o valor médio de casas (`MEDV`) com base em variáveis como número de quartos, proporção aluno/professor, entre outras.

---

## Etapas do projeto

- Coleta de dados
- Análise exploratória (EDA)
- Limpeza e tratamento dos dados
- Criação de baseline
- Treinamento de modelos de Machine Learning
- Avaliação com RMSE

---

## Modelos utilizados

- Baseline (regra baseada em número de quartos)
- Regressão Linear
- Árvore de Decisão
- Random Forest

---

## Métrica de avaliação

- RMSE (Root Mean Squared Error)

---

## Tecnologias utilizadas

- Python --> 3.12.10
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Seaborn / Matplotlib
- streamlit
- ngrok

---

## Como executar

1. Clonar repositório:

```
git clone https://github.com/DouglaasPH/house-price-prediction.git
```

2. Criar variável de ambiente:

```
python -m venv venv
```

3. Executar variável de ambiente:

```
venv/Scripts/activate
```

4. Instalar dependências:

```
pip install -r requirements.txt
```

5. Executar projeto:

```
python main.py
```

6. Executar aplicação com streamlit:

```
streamlit run app.py
```

Após executar o comando, o streamlit irá gerar uma URL localhost para acessar a aplicação.

7. Expor aplicação com Ngrok:

```
ngrok http --url={seu dominio do ngrok} 8501
```

Após executar o comando, o Ngrok irá gerar uma URL pública para acessar a aplicação.

---

## Estrutura do projeto (sugestão)

```
├── main.py
├── app.py
├── data.csv (após executar main.py)
├── requirements.txt
├── .gitignore
└── README.md
```
