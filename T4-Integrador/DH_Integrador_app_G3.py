# Importando bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn import tree
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

import streamlit as st

########################
# Leitura do Dataframe #
########################

#DF = pd.read_csv('C:\\Users\\cesar\\Desktop\\Data Science\\Digital-House-G3-main\\T4-Integrador\\EWF_DF.csv')
DF = pd.read_csv('EWF_DF.csv')
#DF.drop('Unnamed: 0', axis=1, inplace=True)

#DF_std_ext2 = pd.read_csv('C:\\Users\\cesar\\Desktop\\Data Science\\Digital-House-G3-main\\T4-Integrador\\DF_std_ext2.csv')
DF_std_ext2 = pd.read_csv('DF_std_ext2.csv')
#DF_std_ext2.drop('Unnamed: 0', axis=1, inplace=True)

DF_best_features = pd.read_csv('DF_best_features.csv')



# Preparando dados para Modelos

DF_std = DF
feature_cols_base = list(DF_std.iloc[:,1:].columns)
feature_cols = feature_cols_base
feature_cols.remove('Startups_Ranking_2020')
feature_cols.remove('Startups_Score_2020')

X = DF_std[DF_std['Startups_Score_2020'] > 0 ][feature_cols].copy()
y = DF_std[DF_std['Startups_Score_2020'] > 0 ].Startups_Score_2020

#normalizando os dados
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
SEED = 10
TS = 0.10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TS, random_state = SEED)

## MODELOS

# Elastic Net

alpha = 1.08
model_en = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True, max_iter=6, tol=0.0001, random_state=SEED)
model_en.fit(X_train, y_train)
ypred_en = model_en.predict(X_test)

# Decision Tree

model_dt = tree.DecisionTreeRegressor(criterion='squared_error', splitter='best', ccp_alpha=0.11, max_depth=3, min_samples_split=3,
                                  min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None)
model_dt.fit(X_train, y_train)
ypred_dt = model_dt.predict(X_test)

# XGBoost

data = X
label = y
data_dmatrix = xgb.DMatrix(data=data,label=label)
model_xg_reg = xgb.XGBRegressor(objective ='reg:linear', alpha = 0.01, colsample_bytree = 0.3,
                                learning_rate = 0.14, max_depth = 2, n_estimators = 10)
model_xg_reg.fit(X_train,y_train)
ypred_xgb = model_xg_reg.predict(X_test)


####################
# Início do Layout #
####################


# Text/Title
st.title("Startups X Liberdade Econômica no Mundo")

st.header("""
O quanto os índices de qualidade de um país podem ajudar na decisão de investir em Startups?
""")

# Selectbox - País de Comparação
Countries = list(DF_std.Country)
Country = st.sidebar.selectbox("Escolha um País", Countries)

st.write("Você escolheu:")
st.success(Country)

# Seleção dos Top Ranking dos Países
top_rank = st.sidebar.slider("Top Rank", 1, 20)
DF_top = DF_best_features
cond1 = DF_top['Startups Ranking 2020'] <= top_rank
cond2 = DF_top['Startups Ranking 2020'] > 0

Top = DF_top[(cond1) & (cond2)].sort_values(by='Startups Ranking 2020')

#####################
# Destaques do País #
#####################
st.sidebar.write("Índices do", Country)
IDH = DF[DF['Country'] == Country].EFW1_HDI__10_20.iloc[0].round(3)
st.sidebar.write("IDH -",(IDH))

GPD = DF[DF['Country'] == Country].WB_GDP_10_20.iloc[0].round(0)
st.sidebar.write("PIB - $",(GPD), "bilhões")

POP = DF[DF['Country'] == Country].WB_Popu_10_20.iloc[0].round(0)
st.sidebar.write("População -",(POP),"milhões")

Ranking = int(DF[DF['Country'] == Country].Startups_Ranking_2020.iloc[0].round(0))
st.sidebar.write("Posição -",(Ranking),"lugar")

# Gráfico em torta:

df_pie = DF_std.loc[:, ['Country', 'WB_Agri_10_20', 'WB_Indu_00_10', 'WB_Serv_10_20']]
df_pie.columns = ['Country', '% Agro', '% Ind', '% Serv']
pie = df_pie[df_pie.Country == Country].transpose().iloc[1:,:]
pie = pie.reset_index()
pie.columns = ['Index', 'Country']
Outros = {'Outros': (1 - (pie.Country.sum()/100))}
DF_Outros = pd.DataFrame.from_dict(Outros, orient='index')
DF_Outros = DF_Outros.reset_index()
DF_Outros.columns = ['Index', 'Country']
pie = pie.append(DF_Outros, ignore_index=True)

fig = px.pie(pie, values='Country', names='Index', title='Distribuição do PIB')
st.sidebar.plotly_chart(fig, use_container_width=True, use_container_height=True)

#########
# Plots #
#########

DF_plot = DF_std_ext2.copy()

top_score = DF_std[(DF_std['Startups_Ranking_2020'] <= top_rank) & (DF_std['Startups_Ranking_2020'] > 0)].sort_values(by='Startups_Ranking_2020').copy()
lista_score = list(top_score.Country)
lista_score.append(Country)
close = DF_plot.Countries.isin(lista_score)
data_plot = DF_plot[DF_plot['Year'] >= 2000].loc[close]

# Função para plotar linha:

def plotline(y, titulo):
    fig = px.line(data_plot, x='Year', y=y, color='Countries', #title='Tamanho do Governo',
                     width=900, height=400, symbol="Countries", markers=True, line_shape='spline')
    fig.update_yaxes(title=titulo, showgrid=False, linecolor='black', minor_griddash="dot", mirror=True,
                     tickfont=dict(family='Arial',  color='maroon',  size=15),
                     )
    fig.update_xaxes(title='Anos', mirror=True, showgrid=False, linecolor='black',
                     )
    fig.update_layout(title=titulo, xaxis_title="Anos", yaxis_title="", legend_title="Países",
                      font=dict(family="Arial, monospace", size=20, color="black")
                     )
    st.plotly_chart(fig, use_container_width=True, use_container_height=True)
    return y

st.header("Principais Índices de Liberdade")
st.subheader("Comparação dos 5 componentes que compõem o índice de liberdade econômica entre os países selecionados.")

# Plot 'Tamanho do Governo':
plotline('1.0_size_government','Tamanho do Governo')
# Plot Direito de Propriedade
plotline('2.0_property_rights','Direito de Propriedade')
# Plot Estabilidade Financeira
plotline('3.0_sound_money','Estabilidade Financeira')
# Plot Comércio
plotline('4.0_trade','Comércio')
# Plot Regulamentação
plotline('5.0_regulation','Regulamentação')

# Funcção para plotar Barras:

Med_Mundial = pd.DataFrame(DF_best_features.iloc[-1, :]).transpose().reset_index(drop=True)
Ranking_startups = pd.concat([Med_Mundial, Top]).copy()
Ranking_startups = pd.concat([Ranking_startups, DF_best_features[DF_best_features['País'] == Country]])

def plotbar(y, titulo):
    fig = px.bar(Ranking_startups, y=y, x='País', text_auto='.3s', color="País", title=titulo, width=500, height=500,)
    fig.update_yaxes(title=titulo, showgrid=False, linecolor='black', mirror=True, minor_griddash="dot",
                     title_font=dict(size=25, family='Arial', color='black'))
    fig.update_xaxes(title='Paises', showgrid=False, linecolor='black', mirror=True, tickangle=40,
                     tickfont=dict(family='Arial',  color='maroon', size=1))
    fig.update_layout(title=titulo, xaxis_title='Países', yaxis_title="", legend_title="Países",
                      font=dict(family="Arial, monospace", size=20, color="black"))
    fig.update_traces(textfont_size=15,  textangle=0,  textposition="outside",  cliponaxis=False)

    st.plotly_chart(fig, use_container_width=True, use_container_height=True)
    return y

st.header("Índices Startups")
st.subheader("Comparação dos índices considerados mais relevantes para se prever o sucesso de uma "
             "startup entre a média mundial, os países mais bem colocados e o país selecionado.")


# Apresentação do DF_best_features
st.dataframe(DF_best_features)

# Plot dos gráficos em barra das melhores features escolhidas pelos modelos.

Features = list(DF_best_features.columns)[1:]
status = st.multiselect("Selecione as Features de interesse.", Features)
st.success(status)

for imp in status:
    plotbar(imp, imp)

# Considerações
#st.sidebar.header("Projeto em Andamento")

if st.button("Fim!"):
    st.balloons()
