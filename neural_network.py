import pandas as pd
import numpy as np


beta1,beta2 = 0.9,0.99
t =1
e = 10e-8
lambda_ = 0.001

#6stats,3neurons,8species

df = pd.read_excel("pokemon_iv_dataset_1000.xlsx")
x=df[["hp","attack","defense","sp_attack","sp_defense","speed"]].values
a=df["species"].values
y=np.eye(8)[a]
w1 = np.random.randn(6,3)*0.01
b1 = np.zeros((1,3))

w2 = np.random.randn(3,8)*0.01
b2 = np.zeros((1,8))  

x=x/100

learn=0.1

m=len(a)
mb1,vb1 = np.zeros_like(b1),np.zeros_like(b1)
mb2,vb2 = np.zeros_like(b2),np.zeros_like(b2)
mw1,vw1 = np.zeros_like(w1),np.zeros_like(w1)
mw2,vw2 = np.zeros_like(w2),np.zeros_like(w2)

cmb1,cvb1 = np.zeros_like(b1),np.zeros_like(b1)
cmb2,cvb2 = np.zeros_like(b2),np.zeros_like(b2)
cmw1,cvw1 = np.zeros_like(w1),np.zeros_like(w1)
cmw2,cvw2 = np.zeros_like(w2),np.zeros_like(w2)

def relu(z):
    return np.maximum(0,z)
def reluder(z):
    return(z>0).astype(float)


for i in range(60000):
    z1=x@w1+b1
    h=relu(z1)
    z2 = h@ w2+b2

    exp_z2 = np.exp(z2)
    p = exp_z2/(np.sum(exp_z2,axis = 1,keepdims=True))

    dz2 = p-y
    gradw2 = (h.T@dz2)/m
    gradb2 = (np.sum(dz2,axis=0,keepdims=True))/m

    dh = dz2@w2.T
    dz1=dh*reluder(z1)
    gradw1 = (x.T@dz1)/m
    gradb1 = (np.sum(dz1,axis=0,keepdims=True))/m

    l2 = l2 = (lambda_/2)*(np.sum(w1**2)+np.sum(w2**2))
    gradw1 = gradw1+lambda_*w1
    gradw2 = gradw2+lambda_*w2

    mw1 = ((1-beta1)*gradw1+beta1*mw1)
    mw2 = ((1-beta1)*gradw2+beta1*mw2)
    cmw1 = mw1/((1-(pow(beta1,t))))
    cmw2 = mw2/(1-(pow(beta1,t)))

    mb1 = ((1-beta1)*gradb1+beta1*mb1)
    mb2 = ((1-beta1)*gradb2+beta1*mb2)
    cmb1 = mb1/((1-(pow(beta1,t))))
    cmb2 = mb2/(1-(pow(beta1,t)))

    vw1 = ((1-beta2)*gradw1*gradw1+beta2*vw1)
    vw2 = ((1-beta2)*gradw2*gradw2+beta2*vw2)
    cvw1 = vw1/((1-(pow(beta2,t))))
    cvw2 = vw2/(1-(pow(beta2,t)))

    vb1 = ((1-beta2)*gradb1*gradb1+beta2*vb1)
    vb2 = ((1-beta2)*gradb2*gradb2+beta2*vb2)
    cvb1 = vb1/((1-(pow(beta2,t))))
    cvb2 = vb2/(1-(pow(beta2,t)))

    fingradw1 = (cmw1)/(np.sqrt(cvw1)+e)
    fingradw2 = (cmw2)/(np.sqrt(cvw2)+e)
    fingradb1 = (cmb1)/(np.sqrt(cvb1)+e)
    fingradb2 = (cmb2)/(np.sqrt(cvb2)+e)

    t=t+1

    w2 = w2 - learn*fingradw2
    b2 = b2 - learn*fingradb2  
    w1 = w1 - learn*fingradw1
    b1 = b1 - learn*fingradb1
    loss = -np.mean(np.sum(y * np.log(p + 1e-9), axis=1))+l2
    if(i%1000==0):
        print("loss:",loss)
       
    

print("the weights for neurons",w1)

print("the bias for neurons",b1)
print("the weights for softmax",w2)
print("the bias for neurons",b2)


np.savez("pokemonpredictneuralversion2",wneuron = w1,biasneuron = b1, wsoft = w2, bsoft = b2)





