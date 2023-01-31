import numpy as np

class FastICA():
  def __init__(self):
    print('ICA')
  
  def Fit_ICA(self,X,n_com,dif=1e-6,itmax=1000):
    # X is n_feature x n_samples
    self.X=X
    n_f=self.X.shape[0] #features
    m=self.X.shape[1]
    if n_com==0:
      n_com=n_f
    # W is n_features x n_components
    W=np.zeros((n_f,n_com),dtype=float)
    for it in range(n_com):
        if it==0:
            wp = np.random.rand(n_f, 1)
        else:
            wp = np.random.rand(n_f, 1)
            wp = orthogonalize(W, wp, it)
        for itx in range(itmax):
            last_wp=wp
            wp = opt_negentropia(X, wp)
            if it > 0:
                wp = orthogonalize(W, wp, it)
            wp = normalize(wp)
            W[:,it] = np.squeeze(wp)
            if np.max(np.abs(wp-last_wp))<dif:
                break
        print(f"Iteración para la {it} componente fue {itx}")
        
    #W is n_comp x n_features
    #S is n_comp x n_features
    S= W.T @ X
    return W.T, S

def opt_negentropia(X,wp):
  Y=wp.T @ X
  term1= np.mean(gprime(Y),axis=1,keepdims=True)*wp
  term2= np.mean(g(Y)*X,axis=1,keepdims=True)
  return term1 - term2

def orthogonalize(W, wp, i):
    w_i=W[:,:i]
    w_n=wp - np.sum((wp.T @ w_i)* w_i,axis=1,keepdims=True)
    return w_n

# wp normalization
def normalize(wp):
    return wp / np.linalg.norm(wp)     

def gprime(Y,a=1.):
  return a* (1-np.tanh(a*Y)**2)

def g(Y,a=1.):
  return np.tanh(a*Y)