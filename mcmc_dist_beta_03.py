# =====================================================
# imports
import numpy as np
import scipy
import scipy.stats
import matplotlib as mpl   
import matplotlib.pyplot as plt
import math 

# =====================================================
# Dados de entrada
input_file_path = "C:\\Users\\gusta\\Dropbox\\PC\\Documents\\mestrado\\etapa_de_programacao\\programacao_beta_mcmc\\time_dnit.csv"

# =====================================================
#trabalhando com os dados ## tempo de espera para acesso a via
separador = ';'

z1 = []
x = []

with open(input_file_path, 'r', newline='') as csv_file:
    for line_number, content in enumerate(csv_file):
        if line_number:  # pula cabeçalho
            colunas = content.strip().split(separador)
            z1.append( float(colunas[0]) )
            x.append( float(colunas[8]) )

mean = np.mean(x) # media
sd = np.std(x)  # desvio padrao
#print(sd)
#print(mean)
# =====================================================
# =====================================================

# Vamos aplicar uma distribuicao beta aos dados  
modelo=lambda t:np.random.beta(mean,sd,t)

# Formar a populacao, embasado nos dados de entrada ## media e desvio padrao observados
populacao = modelo(30000)#30000
# Assumir uma pop a ser observada
observado = populacao[np.random.randint(0, 30000, 1000)]
# Observacao por grafico
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.hist( observado,bins=35 ,)
ax.set_xlabel("Dados")
ax.set_ylabel("Frequencia")
ax.set_title("Figura 1: Distribuicao das 1000 amostras observadas da populacao total de 30000")
mu_obs=observado.mean()
mu_obs

# =====================================================
# Modelo de transicao 
modelo_trans = lambda x: [x[0],np.random.beta(x[1],0.5,(1,))[0]]

# sigma = desvio padrao; mu = media;
def prior(x):
    #x[0] = mu, x[1]=sigma 
    # Retorna 1 para os valores validados de desvio padrao
    if(x[1] <=0):
        return 0
    return 1
# =====================================================
#Computar a probabilidade da data dado o valor do desvio padrao

# Mesma que a usada anteriormente, mas testaremos com a biblioteca scipy.
def prob_beta_log(x,data):
    #x[0]=mu, x[1]=sigma 
    #data = a mesma observada
    return np.sum(np.log(scipy.stats.beta(x[0],x[1]).pdf(data)))

# =====================================================
# Aceitacao ou nao da amostrada criada 
def aceitacao(x, x_novo):
    if x_novo>x:
        return True
    else:
        aceitar=np.random.uniform(0,1)
        return (aceitar < (np.exp(x_novo-x)))

# =====================================================
# Acelerador do metodo 
def metropolis_hastings(computar_prob,prior, modelo_trans, param_init,iterations,data,regra_de_aceite):
    x = param_init 
    aceitado = []
    rejeitado = []   
    for i in range(iterations):
        x_novo =  modelo_trans(x)    
        x_lik = computar_prob(x,data)
        x_novo_prob = computar_prob(x_novo,data) 
        if (regra_de_aceite(x_lik + np.log(prior(x)),x_novo_prob+np.log(prior(x_novo)))):            
            x = x_novo
            aceitado.append(x_novo)
        else:
            rejeitado.append(x_novo)            
                
    return np.array(aceitado), np.array(rejeitado)
    
aceitado, rejeitado = metropolis_hastings(prob_beta_log,prior,modelo_trans,[mu_obs,0.1], 10000,observado,aceitacao)# 50000,observado,aceitacao)
aceitado[-10:,1]
print(aceitado.shape)
# =====================================================
# Amostragem para alguns valores ## Visualizacao

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(2,1,1)
ax.plot( rejeitado[0:50,1], 'rx', label='Rejeitados',alpha=0.5)
ax.plot( aceitado[0:50,1], 'b.', label='Aceitos',alpha=0.5)
ax.set_xlabel("Interacoes")
ax.set_ylabel("$\sigma$")
ax.set_title("Figura 2: Amostragem MCMC para 50 amostras.") 
ax.grid()
ax.legend()
# =====================================================
# Amostragem completa

ax2 = fig.add_subplot(2,1,2)
to_show=-aceitado.shape[0]
ax2.plot( rejeitado[to_show:,1], 'rx', label='Rejeitados',alpha=0.5)
ax2.plot( aceitado[to_show:,1], 'b.', label='Aceitos',alpha=0.5)
ax2.set_xlabel("Interacoes")
ax2.set_ylabel("$\sigma$")
ax2.set_title("Figura 3: Amostragem MCMC para o desvio padrao. Contem todas as amostras.")
ax2.grid()
ax2.legend()
# =====================================================
# Histograma
fig.tight_layout()
aceitado.shape


show=int(-0.75*aceitado.shape[0])
hist_show=int(-0.75*aceitado.shape[0])

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(1,2,1)
ax.plot(aceitado[show:,1])
ax.set_title("Figura 4: Traceplot MCMC")
ax.set_ylabel("$\sigma$")
ax.set_xlabel("Interacoes")
ax = fig.add_subplot(1,2,2)
ax.hist(aceitado[hist_show:,1], bins=20,density=True)
ax.set_ylabel("Frequencia (beta)")
ax.set_xlabel("$\sigma$")
ax.set_title("Figura 5: Histograma $\sigma$")
fig.tight_layout()


ax.grid("off")
# =====================================================
# A posteriore 

mu=aceitado[show:,0].mean()
sigma=aceitado[show:,1].mean()
print(mu, sigma)
model = lambda t,mu,sigma:np.random.beta(mu,sigma,t)
observado_gen=model(populacao.shape[0],mu,sigma)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.hist( observado_gen,bins=70 ,label="Distribuicao prevista para os 30000 individuos") 
ax.hist( populacao,bins=70 ,alpha=0.5, label="Valores originais para os 30000 individuos")
ax.set_xlabel("Media")
ax.set_ylabel("Frequencia")
ax.set_title("Figura 6: Distribuicao posterior para as previsoes")
ax.legend()
plt.show()

# =====================================================
# Salvando os dados finais 
with open('dist_beta.txt', 'w') as output:
    output.write(str(aceitado))
    output.close()