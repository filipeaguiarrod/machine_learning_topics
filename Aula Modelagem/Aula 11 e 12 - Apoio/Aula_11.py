import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import streamlit as st

import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings("ignore")

# Fontes são meus próprios artigos e notebooks por isso não estou citando nada.
# https://github.com/sn3fru/pricing-predict/blob/master/notebook.ipynb
# https://www.kaggle.com/sn3fru/intepreting-machine-learning-linear-regression
# https://medium.com/data-hackers/interpretando-ml-modelos-lineares-bccbd9f5d6a0

# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# local_css("style.css")

# t = "<div>Hello there my <span class='highlight blue'>name <span class='bold'>yo</span> </span> is <span class='highlight red'>Fanilo <span class='bold'>Name</span></span></div>"
# st.markdown(t, unsafe_allow_html=True)

sns.set(style="whitegrid") # plotagens bonitinhas
rng = np.random.RandomState(42) # garantir que os meus números "aleatórios" sejam sempre os mesmos.

st.write('# Modelagem')

st.write('''
#### Conteúdo de Hoje:
    - Regressões Lineares - Guia de Uso
    - Modelagem Estatística - Viés e Causalidade
    - Modelagem Estatística - formas funcionais
    - Case

''')

st.write('Nessa aula vamos começar a aprender o lado "artistico" da Ciência de Dados. Começaremos a pensar nos problemas pelo prisma do viés e da causalidade que não é um grande foco de AI.')

st.write('Para aquecer os motores vamos usar um dataset em que temos o raro momento de conhecer exatamente como as variáveis impactam umas as outras. Como Saberemos? Vamos criar esse dataset do zero.')

'Equação da reta:'
st.latex(r'''
y = ax + b
''')

text = st.empty()
text.text_area('Como seria nossa função?', '')


st.write('### Passo 1 - Pessoas e Educação')

qtd  = st.sidebar.slider(
    'Tamanho da amostra em pessoas',
    10, 2000, (30)
)

var1 = st.slider(
    'Educação Máxima:',
    10.0, 20.0, (16.0))

education =var1*rng.rand(qtd) # variavel ale
df = pd.DataFrame({'Educ':education})
df['Educ'].plot.hist(figsize=(9,6))
st.pyplot()

'Exemplo dos dados:'
st.dataframe(df.head(10))

'Média:', round(df['Educ'].mean(),2), 'Desvio Padrão', round(df['Educ'].std(),2)
st.write('### Passo 2 - Salário dependendo da Educação')

st.write('O primeiro passo é definirmos como a educação impactará os salários. Esse valor representa quanto 1 ano a mais de educação impacta na média o salário do individuo.')

educ_wage = st.slider(
    'Selecione o efeito da Educação no Salario:',
    100.0, 1000.0, (300.0))

st.write('Como esse é um evento estocastico (nem sempre as pessoas terão o mesmo retorno para o salário), vamos adicionar um pouco de aleatoriedade:')

var2 = st.slider(
    'Selecione a Aleatoriedade dos Salarios:',
    500.0, 1500.0, (1200.0))

st.write('Por último vamos definir um salário minimo:')

salario_minimo = st.slider(
    'Selecione o efeito da Educação no Salario:',
    500.0, 2000.0, (1000.0))


education = var1 * rng.rand(qtd) # variavel ale
wage = educ_wage * education + var2*rng.randn(qtd) + salario_minimo # y = ax + b

df = pd.DataFrame({'Educ':education, 'Salaries':wage})

st.write('Distribuição dos Salários:')
df['Salaries'].plot.hist(figsize=(9,6))
st.pyplot()


st.write('5 linhas do dataset:')
st.dataframe(df.head().round(0))

st.write('Algumas estatisticas Descritivas:')
st.dataframe(df.describe())

st.write('Correlação: ', df.corr())

st.write('Agora vamos rodar uma regressão Simples entre Educação e Salário:')

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(df[['Educ']], df[['Salaries']])

xfit = np.linspace(0, 16, 100)
yhat = model.predict(xfit[:, np.newaxis])

plt.scatter(education, wage)
plt.plot(xfit, yhat, color='red')

st.pyplot()

st.write('Vamos analisar a distribuição dos residuos (azul) com uma distribuição normal (laranja):')

st.write('''
As Curvas são parecidas?
- Para que serve essas duas curvas serem parecidas?
- Qual métodos vocês usariam para checar se essas curvas são parecidas?
''')


from scipy.stats import norm

df['yhat'] = model.predict(df[['Educ']])
df['erro'] = df['yhat']-df['Salaries']

sns.distplot(df['erro'] , fit=stats.norm);
(mu, sigma) = stats.norm.fit(df['erro'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')
st.pyplot()

# st.write(
# "Intercepto beta_0 Calculado: ", round(model.intercept_[0],2)
# )
# st.write(
# "Inclinação beta_1 Calculado: ", round(model.coef_[0][0],2)
# )


st.write('### Rodando uma regressão e analisando o output:')
'O que r2 tem com a correlação?'
function = '''
Salaries ~ Educ
'''


model = smf.ols(function, df).fit()
st.text(model.summary())


import statsmodels.api as sm

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 1, ax=ax)
ax.set_ylabel("Salario")
ax.set_xlabel("Nível da Educação")
ax.set_title("Linear Regression")
st.pyplot()


st.write('### Etapa 3. Adicionando Covariáveis')


st.write('Legal, motores aquecidos, agora vamos tornar o nosso exemplo um pouco mais Real. Lá no mundo real, além de existir muitas variáveis impactando o salário, elas ainda se relacionam entre si e isso aumenta muito a complexidade da análise.')

# education = var1 * rng.rand(qtd) # variavel ale
# wage = educ_wage * education + var2*rng.randn(qtd) + salario_minimo # y = ax + b

'Representação de uma variável aleatória com uma distruição normal:'
st.latex(
'''
X \sim \mathcal{N}(\mu,\,\sigma^{2})\,.
''')

'E nossas variaveis:'
st.latex('''Esforço \sim \mathcal{N}(0,3)''')
st.latex(r'''
Educação \sim \frac{7}{10} \mathcal{N}(10,4) + \frac{3}{10} Esforço + 5
''')
st.latex(r'''
Experiência \sim 25\mathcal{N}(17,6)
''')

st.latex(r'''
Salário \sim 500\mathcal{N}(0,1) + 300Educação + 200experiencia + 200esforço + 1000
''')

esforço = rng.normal(0, 3, qtd)
educação = (.7)*(rng.normal(10, 4, qtd)) + 5*(.3)*esforço + 5
experiencia = rng.normal(17, 6, qtd)

# target = st.number_input('Efeito', min_value=0.0, max_value=0.5, value=0.3, step=0.01, format=None, key=None)      
salario = 500 * rng.randn(qtd) + \
          300 * educação + \
          200 * experiencia + \
          200 * esforço + \
          1000

df = pd.DataFrame({'Salaries':salario,
                   'Educ':educação,
                   'Expe': experiencia,
                   'Grit': esforço})

df[['Educ', 'Expe', 'Grit']].plot.hist(bins=50, alpha=.5)
st.pyplot()

'Matriz de Correlação:'
corrmat = df.corr()
k = 10
f, ax = plt.subplots(figsize=(8, 6))
cols = corrmat.nlargest(k, 'Salaries')['Salaries'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.15)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
st.pyplot()

"Uma outra forma de visualizar correlações que acabei desenvolvendo ao longo do tempo é através de um grafo. Onde cada nó desse grafo é uma variável e a cor das *edges* que ligam os nós é a força da correlação. (note que não exibimos todas as relações, apenas as principais) "

def plotCorrGraph(df):
    import networkx as nx
    corr = df.corr().abs()
    links = corr.stack().reset_index()
    links.columns = ['var1', 'var2','value']
    links_filtered=links.loc[(links['var1'] != links['var2'])]
    temp = []
    checkvalues= []
    for row in links_filtered.values:
        if row[2]>.1 and row[2] not in checkvalues:
            temp.append(row)
            checkvalues.append(row[2])
    links_filtered = pd.DataFrame(temp, columns=['var1', 'var2','value'])
    G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2', )
    nx.draw(G, with_labels=True, font_size=10, alpha=.75, node_color='#A0CBE2',edge_color=links_filtered.value.values, width=4, edge_cmap=plt.cm.Blues)
    plt.show()
    st.pyplot()

plotCorrGraph(df)


'Por fim, a forma mas didatica, os diagramas de Venn (que está no conteúdo pré-aula)'

'''
    - Como ler (covariância & R2)
    - O que são correlações parciais.
    - Correlações Parciais -> Causalidade
    - O que acontece se retirarmos/esquecermos/não estiver deponivel uma das variáveis?

'''
# Import the library
from matplotlib_venn import venn3, venn3_circles # A, B, AB, C, AC, BC, ABC

values = (1000, 900, 300, 900, 200, 300, 100)
v=venn3(subsets = values, set_labels = ('Salaries', 'Educ', 'Grit'))
c=venn3_circles(subsets = values, linestyle='dashed', linewidth=1, color="grey")
plt.show()
st.pyplot()


function = ''' Salaries ~ Educ + Expe + Grit'''

model = smf.ols(function, df).fit()
st.text(model.summary())

'Por que não parece uma reta? Não era uma regressão *linear*??'

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 1, ax=ax)
ax.set_ylabel("Salario")
ax.set_xlabel("Nível da Educação")
ax.set_title("Linear Regression")
st.pyplot()

'E como está nossa distribuição dos residuos?'

model = LinearRegression()
model.fit(df[['Educ', 'Expe', 'Grit']], df[['Salaries']])
df['yhat'] = model.predict(df[['Educ', 'Expe', 'Grit']])
df['erro'] = df['yhat']-df['Salaries']

sns.distplot(df['erro'] , fit=stats.norm);
(mu, sigma) = stats.norm.fit(df['erro'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Distribuição do Erro')
st.pyplot()

'''
Plus:
é erro ou residuo?

Erro: valor observado - valor verdadeiro

Residuo: valor observado - valor previsto

Como "nunca" (exceto hoje) temos acesso ao valor verdadeiro, nunca podemos ter acesso aos erros, apenas aos residuos.

Mas na real, ninguém pensa nisso e é considerado como sinônimos.
'''

st.write("### ETAPA 4 - Viés")
'''E se soubermos (via outras ciências) que algumas variaveis são importantes e não tivermos as variáveis disponiveis? Dá para rodar? Funciona?? Como serão nossos Betas???'''

'''Vamos pegar o modelo anterior e rtodar apenas os Salarios contra Educação, o que acontece?'''
function = ''' Salaries ~ Educ'''

'''O que podemos fazer para corrigir? Aumentar o tamanho da amostra resolve?'''
model = smf.ols(function, df).fit()
st.text(model.summary())

'E como está nossa distribuição dos residuos?'

model = LinearRegression()
model.fit(df[['Educ']], df[['Salaries']])
df['yhat'] = model.predict(df[['Educ']])
df['erro'] = df['yhat']-df['Salaries']

sns.distplot(df['erro'] , fit=stats.norm);
(mu, sigma) = stats.norm.fit(df['erro'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Distribuição do Erro')
st.pyplot()


'''E se rodarmos uma regressão com uma variável independente? O que acontece? Por que?'''
function = ''' Salaries ~ Expe'''
model = smf.ols(function, df).fit()
st.text(model.summary())

values = (1000, 900, 200, 900, 300, 0, 0) # A, B, AB, C, AC, BC, ABC
v=venn3(subsets = values, set_labels = ('Salaries', 'Exp', 'Educ'))
c=venn3_circles(subsets = values, linestyle='dashed', linewidth=1, color="grey")
plt.show()
st.pyplot()

'E como está nossa distribuição dos residuos?'

model = LinearRegression()
model.fit(df[['Expe']], df[['Salaries']])
df['yhat'] = model.predict(df[['Expe']])
df['erro'] = df['yhat']-df['Salaries']

sns.distplot(df['erro'] , fit=stats.norm);
(mu, sigma) = stats.norm.fit(df['erro'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Distribuição do Erro')
st.pyplot()



st.write('## ETAPA 5 - Modelando e interpretando variáveis categóricas')

'Vamos adicionar uma variável ao modelo categórica que representará se o trabalhador nasceu ou não no Brasil e vamos atribuir a ela um Peso.'

'Note que essa variável tem uma distribuição de Bernoulli e no nosso exemplo a chance é de 50%'

brasil = st.slider(
    'Selecione o efeito da Educação no Salario:',
    -10000, 10000, (-5000))

size = qtd

esforço = 16 * rng.normal(0, 1, size)

educação = (.6)*(16 * rng.normal(0, 1, size)) + (.4)*esforço

experiencia = 25 * rng.normal(0, 1, size)

nasceu_brasil = rng.choice([0, 1], size)

salario = 500 * rng.randn(size) + \
          300 * educação + \
          200 * experiencia + \
          100 * esforço + \
          brasil * nasceu_brasil + \
          1000

df = pd.DataFrame({'salario':salario,
                   'educação':educação,
                   'exp': experiencia,
                   'esforço': esforço,
                   'br': nasceu_brasil})

function = '''salario ~ educação + br'''
df['br'].plot.hist(density=1)
df['br'].plot.kde()
st.pyplot()

model = smf.ols(function, df).fit()
st.text(model.summary())

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 1, ax=ax)
ax.set_ylabel("Salario")
ax.set_xlabel("Nível da Educação")
ax.set_title("Linear Regression")
st.pyplot()


st.write('## ETAPA 6 - Modelando variáveis não lineares com modelos Lineares o.O')

'Imagine agora que nossos dados não são exatamente lineares para rodarmos nosso modelo. O que fazer nesse caso?'

size = 400

esforço = 16 * rng.normal(0, 1, size)

educação = (.6)*(16 * rng.normal(0, 1, size)) + (.4)*esforço

experiencia = 25 * rng.normal(0, 1, size)

nasceu_brasil = rng.choice([0, 1], size)

salario = 500 * rng.randn(size) + \
          600 * educação + \
          -30 * educação**2 + \
          200 * experiencia + \
          100 * esforço + \
         -5000 * nasceu_brasil + \
          1000

df = pd.DataFrame({'Salaries':salario,
                   'Educ':educação,
                   'Exp': experiencia,
                   'Grit': esforço,
                   'br': nasceu_brasil})


function = '''Salaries ~ Educ'''

model = smf.ols(function, df).fit()


fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 1, ax=ax)
ax.set_ylabel("Salaries")
ax.set_xlabel("Education Level")
ax.set_title("Linear Regression")
fig.set_size_inches(8, 5)
st.pyplot()

'Como resolver? A primeira coisa que me vem a mente é fingir que é uma reta e rodar assim mesmo ...'
'E funciona ...'
st.text(model.summary())


'Mas dá para fazer melhor ... Com que se parece essa curva estranha? Será que podemos achar uma função matematica com esse formato?'

function = '''salario ~ educação + I(educação**2)'''

model = smf.ols(function, df).fit()
st.text(model.summary())

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model, 1, ax=ax)
ax.set_ylabel("Salaries")
ax.set_xlabel("Education Level")
ax.set_title("Linear Regression")
fig.set_size_inches(8, 5)
st.pyplot()


' O que é linear aqui são os parâmetros !! o modelo pode entender não linearidades complexas se você modelar ;) '

st.write('## ETAPA 7 - Modelos de elasticidade')

size = 1000

esforço = 16 * rng.poisson(6, size)

educação = (.6)*(16 * rng.poisson(6, size)) + (.4)*esforço

experiencia = 25 * rng.normal(0, 1, size)

nasceu_brasil = rng.choice([0, 1], size)

salario = rng.exponential(30000,size) + \
          300 * educação + \
          1000

df = pd.DataFrame({'salario':salario,
                   'educação':educação})


function = '''salario ~ educação'''

model = smf.ols(function, df).fit()
st.text(model.summary())

df['salario'].plot.hist(bins=50)
st.pyplot()

'Média:', df['salario'].mean(), 'Mediana:', df['salario'].median()

df.plot.scatter('salario', 'educação')
st.pyplot()

'Isso é um problema? se sim, como Resolvemos??'

df['log_salario'] = np.log1p(df['salario'])
df['log_salario'].plot.hist(bins=30)
st.write()

'E como é a interpretação dessa modelagem?'

function = '''log_salario ~ educação'''

model = smf.ols(function, df).fit()
st.text(model.summary())

st.write('## CASE - Mundo Real')

'Agora, no nosso exemplo de dados reais. Vamos usar um antigo arquivo do IBGE (já tem uns 30 anos) para tentar descobrir se existe alguma forma de discriminação de gênero. Como modelariamos esse problema?'

df = pd.read_csv('demografiav2.zip')
df.drop(['gestante'], axis=1, inplace=True)
df=df[df['salario']>2]
# df['gestante'] = (df['sexo']=='gestante')
# df['sexo'].str.replace('gestante', 'mulher')
# df = df[df.sexo.isin(['homem', 'mulher'])]
# df = df[(df.salario<=200000)&df.salario>=0]
# del df['id']
# df.to_csv('demografiav2.csv', index=False)
df=df.dropna()
# df['gestante']  = df['gestante'].astype(int)
# del df['seq']

st.dataframe(df.head(50))

st.dataframe(df.describe())

sns.heatmap(df.corr(), annot=True)
st.pyplot()

# sns.pairplot(df.sample(500), hue="sexo")
st.pyplot()

plotCorrGraph(df)

st.dataframe(df.groupby(['sexo']).mean())

df[df['sexo']=='homem']['salario'].plot.hist(bins=100, alpha=.5, xlim=(-1, 100000))
df[df['sexo']=='mulher']['salario'].plot.hist(bins=100, alpha=.5)
st.pyplot()


sns.distplot(df['salario'] , fit=stats.norm);
st.pyplot()

(mu, sigma) = stats.norm.fit(df['salario'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Salaries distribution')
fig = plt.figure()
res = stats.probplot(df['salario'], plot=plt)
plt.show()
st.pyplot()

'Vamos aplicar log no salario e ver como fica ...'



sns.distplot(np.log1p(df['salario']) , fit=stats.norm);
st.pyplot()

(mu, sigma) = stats.norm.fit(np.log1p(df['salario']))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Salaries distribution')

fig = plt.figure()
res = stats.probplot(np.log1p(df['salario']), plot=plt)
plt.show()
st.pyplot()



'Dá para saber se todas as variáveis importantes estão aqui? Quem nos diz isso?'

'Na próxima aula vamos aprender a resolver esse problema através de experimentos (ou quase-experimentos ...)'

st.write('## Fim')

st.write('by Marcos Silva')

