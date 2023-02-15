import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.signal as signal
from sklearn.decomposition import FastICA 
from scipy.fft import fft, fftfreq
from scipy.stats import zscore

def standar_score(X):
        if len(X.shape)==1:
                # X 1-D array
                mu=X.mean(dtype=float)
                sigma=X.std(dtype=float)
                Z=(X-mu)/sigma
        elif len(X.shape) ==2 : 
                mu=np.mean(X,axis=1,keepdims=True,dtype=float)
                sigma=np.std(X,axis=1,keepdims=True,dtype=float)
                Z=(X-mu)/sigma
        elif len(X.shape)==3:
                mu=np.mean(X,axis=2,keepdims=True,dtype=float)
                sigma=np.std(X,axis=2,keepdims=True,dtype=float)
                Z=(X-mu)/sigma
        return Z        

def dkls(mu_p,mu_q,sigma_p,sigma_q):
    k=len(mu_p)
    
    inv_sigma_p=np.linalg.inv(sigma_p)
    inv_sigma_q=np.linalg.inv(sigma_q)
    
    term0_a=(mu_p - mu_q).T
    term0_b=inv_sigma_p + inv_sigma_q
    term0_c=mu_p - mu_q
    term1 = term0_a @ term0_b @ term0_c
    
    term2= np.trace( inv_sigma_q @ sigma_p )
    
    term3=np.trace( inv_sigma_p @ sigma_q )
    
    D_kls =(term1 + term2+ term3  - 2*k)/4
    
    return D_kls
#---------------------------------------------------------------
def Mean_Remove(X):
        if len(X.shape)>1:
                return X-X.mean(axis=1,keepdims=True)
        else:
                return X-X.mean

def Covariance(X):
        mu=np.mean(X,axis=1,keepdims=True)
        N=X.shape[1]
        return (1/N)*(X-mu)@(X-mu).T

def whitening(X,method='pca'):
        X_c=X-X.mean(axis=1,keepdims=True)
        X_cov=np.cov(X)
        lamdas, v = np.linalg.eigh(X_cov)
        e=0.1e-5
        diag_lambdas=np.diag(1/(lamdas+e)**0.5)
        if method=='pca':
                W_= diag_lambdas @ v.T
        elif method=='zca':
                W_= v @ diag_lambdas @ v.T
        X_w= W_ @ X_c
        return X_w, W_

#--------------------- DESCOMPOTITION ------------------------
class Pca():
    def __init__(self,X=[]):
        # X is features x samples
        if len(X)!=0:
            self.X_data=X
            self.Fit(X)
            print('PCA')

    def fit(self,X):  
        self.data=X
        self.mu=np.mean(self.data,axis=1,keepdims=True)
        self.N=X.shape[1]
        self.cov_matrix=(1/self.N)*(X-self.mu)@(X-self.mu).T
        self.eig_values,self.eig_vectors=np.linalg.eigh(self.cov_matrix)
        
        self.eig_values=self.eig_values[::-1]
        self.eig_vectors=self.eig_vectors[:,::-1]
        
        return self.eig_values,self.eig_vectors

    def project(self,X,n=2,plot=True):
        proj=self.eig_vectors[:,:n].T @ X
        if n==2 and plot==True:
            x1=proj[0,:]
            x2=proj[1,:]
            plt.scatter(x1,x2)
        return proj
    
    def fit_transform(X,n_comp=2):
        eig_val,eig_vec = self.Fit(X)
        X_proj = self.project(X,n=n_comp,plot=False)
        return eig_val, X_proj



def ica_descompotition(X,method='FastICA',random_seed=51):
        #X is whitening data X is n_features x n_samples
        X_=X.T.copy()
        if method=="FastICA":
                ica = FastICA(whiten=False,tol=1.e-4,max_iter=1000)
                ica.fit(X_)
                U=ica.components_
                A=ica.mixing_
                print(f"El número máximo de iteraciones hasta la convergencia fue {ica.n_iter_}")
        return U , A
        
                
                
        
        

def Index_Bad_Channel(data,threshold=200.e-6):
        
        #data is a matrix n_epochs x n_channels x samples
        
        #Obtenemos los maximos pico a pico para cada epoca y canal
        data_peak2peak=np.max(data,axis=2)-np.min(data,axis=2)
        
        #Obtenemos el promedio y el desvío estandar por epoca
        data_peak2peak_means=np.mean(data_peak2peak,axis=0)
        
        #Tenemos que medir distancias entre medias
        
        #Obtenemos aquellos indices donde se encuentra el outlayer
        idx=np.where(data_peak2peak_means>=threshold)
        
        return idx
        
def Mean_Distances(means):
        dist_means=np.zeros((len(means),len(means)))
        for it1 in range(len(means)):
                for it2 in range(len(means)):
                        dist_means[it1,it2]=np.abs(means[it1]-means[it2])
        
        
def Calc_Corr(X):
        return np.corrcoef(X)        
        
        
def Calc_STD(data):
        desv_std=np.std(data,axis=1)
        return desv_std


def Calc_DEE(signal,fs):
        n_rows=signal.shape[0] 
        n_times=signal.shape[1]
        df=fs/n_times
        t=np.linspace(0.,float(n_times/fs),n_times,dtype=float)
        signal_fft=[]
        signal_DEE=[]
        int_E=[]
        int_E_over50=[]
        for nch in range(n_rows):
                #Devuelve el espectro de Fourier para frecuencias positivas
                sigfft, freq=Calc_FFT(signal[nch,:],fs, n_times)
                #Obtenemos el valos de la suma acumulada para los elementos mayores a 50 Hz
                signal_fft.append((sigfft))
                #MUltiplicamos por 2 para contemplar a las potencias, no dividimos por 2 pi porque no estamos normalizando por omega
                signal_E=2*sigfft*np.conjugate(sigfft)
                signal_DEE.append(signal_E)
                #Plot_Subplot_2(signal[nch,:],signal_E,t,freq)
                int_E.append(df*np.sum(signal_E))
                int_E_over50.append(df*np.sum(signal_E[freq > 50.]))
        
        #Calculamos la varianza de la señal 
        var_signal=np.var(signal,axis=1)

        #Calculamos el error porcentual entre la varianza y la integral 
        err=(int_E-var_signal)/var_signal

        #Calculamos la integral de la densidad espectral por encima de los 50 HZ
        return int_E, int_E_over50, signal_DEE


def Calc_FFT(signal,fs, n):
        signal_fft=np.fft.fft(signal)
        freq=fftfreq(n, 1/fs)[:n//2]
        return signal_fft[0:n//2] , freq

def Calc_AUC(signal,dx):
        return dx*np.sum(np.abs(signal),axis=1)

def Calc_Der(signal,dx):
        n=int(signal.shape[1])
        der_signal=np.zeros((signal.shape[0],signal.shape[1]))
        for it in range(0,n-1):
                der_signal[:,it]=signal[:,it+1]-signal[:,it]     
        der_signal[:,-1]=signal[:,-1]-signal[:,-2]
        return der_signal/dx
        
def Calc_Int_Trap(y,dx):
        return np.trapz(y,dx=dx)        


def find_outliers(X,threshold=2.5,maxiter=4,tail=0):
        # X is 1D-array
        from scipy.stats import zscore
        my_mask = np.zeros(len(X), dtype=bool)
        for _ in range(maxiter):
                X = np.ma.masked_array(X, my_mask)
                if tail == 0:
                        this_z = np.abs(zscore(X))
                elif tail == 1:
                        this_z = zscore(X)
                elif tail == -1:
                        this_z = -zscore(X)
                else:
                        raise ValueError("Tail parameter %s not recognised." % tail)
                local_bad = this_z > threshold
                my_mask = np.max([my_mask, local_bad], 0)
                if not np.any(local_bad):
                        break
        bad_idx = np.where(my_mask)[0]
        return bad_idx
        




def Obtain_Index_Max(data,threshold=4.,axis=0,plot=True):
        #En este punto obtenemos los desvios de cada trial de imagineria
        desv_std=np.std(data,axis=axis)
        #Normalizamos el desvio
        mean_data=np.mean(data,axis=axis,keepdims=True)
        abs_data=np.abs(data-mean_data)
        
        if axis==None:
                data_norm=abs_data/ desv_std
        else:
                data_norm=abs_data/ desv_std[:,None] #Con este comando generamos que divida elemento a elemento una matriz por vector
        #Encontramos los indices donde el desvio es mayor que el umbral
        index=data_norm>=threshold
        
        #PLoteamos
        if plot:
                Plot_Desv(data_norm,title="Normalizado Registro")        
        return index

#--------------------------------------------Filters-------------------------------------------
def Construct_FIR_Filter(sfreq, freq, gain, filter_length,fir_window):
        from scipy.signal import firwin2 as fir_design
        #Normalizamos las frecuencias
        freq = np.array(freq) / (sfreq / 2.)
        h = fir_design(filter_length, freq, gain, window=fir_window)
        return h


#-------------------------------------------- PLOTS ----------------------------------------------------
def Plot_Bar(y,title="",xlabel="",ylabel=""):
        x=np.linspace(1,len(y),len(y))
        fig,ax=plt.subplots()
        fig.set_size_inches((8,8))
        ax.bar(x,y)
        ax.set_xlabel(xlabel,fontsize=14)
        ax.set_ylabel(ylabel,fontsize=14)
        plt.title(title)

def plot_desv(desv,title,threshold=4.,return_fig=False):
        x=np.linspace(1,len(desv),len(desv))
        fig, axs = plt.subplots(1,sharex=True)
        fig.set_size_inches(8,8)
        fig.suptitle(title)
        axs.scatter(x,desv,color='green')
        axs.axhline(y=threshold,  color="black", linestyle="--")
        axs.set(xlabel='N°',ylabel='Desvío')
        plt.show()
        if return_fig:
                return fig
               
def plot_register(data_plot,names,n_samples,n_epochs,ini,return_fig=False):
        nplots=len(names)
        fig, axs = plt.subplots(ncols=1,nrows=nplots,sharex=True)
        fig.set_size_inches((20,10))
        y_ticklabels = names

        positions_vlines= np.linspace(n_samples, n_samples*(n_epochs-1),n_epochs-1)
        positions_ticks = np.linspace(0, n_samples*(n_epochs-1),n_epochs)+500
        labels_ticks = np.linspace(ini+1,ini+n_epochs, n_epochs,dtype=int)
        for it in range(nplots): 
                axs[it].plot(data_plot[it,:],linewidth=0.2,color='k')
                ymin=-np.max(np.abs(data_plot[it,:]))
                ymax=np.max(np.abs(data_plot[it,:]))
                axs[it].vlines(positions_vlines,ymin,ymax,linestyle='dashed',colors='k',linewidth=0.5)
                axs[it].set_yticks([])
                axs[it].set_ylabel(y_ticklabels[it],rotation=45.,fontsize=8)
                axs[it].set_xlim((0,n_epochs*n_samples))
        axs[0].set_xticks(positions_ticks,labels=labels_ticks) 
        fig.subplots_adjust(wspace=0, hspace=0)
        if return_fig:
                return fig

def plot_scatter_register(X,ax,mark='o',c='', label=''):
        
        X_mu=X.mean(axis=1)
        X_sigma=np.cov(X)
        ax.scatter(X[0,:],X[1,:],marker=mark,cmap='tab20b',color=c,facecolors='none',
                        alpha=0.25)
        
        ax.scatter(X_mu[0],X_mu[1],marker=mark,cmap='tab20b',color=c,
                                label=label)
        ax = plot_ellipse2d_gaussian(X_mu, X_sigma, nstd =1.,ax=ax,color=c)
        
        return ax


def plot_scatter_register_bygender(data,mu,sigma, idx_f = None,idx_m = None, n_std =1., comp=[0,1], labels=['',''], title = ''):
        fig , ax = plt.subplots()
        fig.set_size_inches((8,8))
        for it_f, it_m in zip(idx_m, idx_f):
                        ax.scatter(data[it_f][0,:],data[it_f][1,:],
                                        marker='^',cmap='tab20b',color="C{}".format(it),facecolors='none',
                                        alpha=0.25)
                        
                        ax.scatter(data[it_m][0,:],data[it_m][1,:],
                                        marker='o',cmap='tab20b',color="C{}".format(it+1),facecolors='none',
                                        alpha=0.25)
                        it+=2
        if idx_f != None and idx_m != None:
                it=0
                for it_f, it_m in zip(idx_m, idx_f):
                        ax.scatter(data[it_f][0,:],data[it_f][1,:],
                                        marker='^',cmap='tab20b',color="C{}".format(it),facecolors='none',
                                        alpha=0.25)
                        
                        ax.scatter(data[it_m][0,:],data[it_m][1,:],
                                        marker='o',cmap='tab20b',color="C{}".format(it+1),facecolors='none',
                                        alpha=0.25)
                        it+=2

                it=0
                for it_f, it_f in zip(idx_m, idx_f):
                        ax.scatter(mu[it_f][0],mu[it_f][1],
                                marker='^',cmap='tab20b',color="C{}".format(it),
                                label=f"M{it_f}")
                        ax = plot_ellipse2d_gaussian(mu[it_f], sigma[it_f], nstd =1., 
                                                     ax=ax,color="C{}".format(it))
                        
                        
                        ax.scatter(mu[it_m][0],mu[it_m][1],marker='o',cmap='tab20b',color="C{}".format(it+1),
                                   label=f"H{it_m}")
                        ax = plot_ellipse2d_gaussian(mu[it_m], sigma[it_m], nstd =1., 
                                                     ax=ax,color="C{}".format(it+1))
                        it+=2
                        
        elif (idx_f != None) and (idx_m == None):
                for it in idx_f:
                        ax.scatter(data[it][0,:],data[it][1,:],
                                   marker='o',cmap='tab20b',color="C{}".format(it),facecolors='none',
                                   alpha=0.25)
                for it in idx_f:
                        ax.scatter(mu[it][0],mu[it][1],marker='o',cmap='tab20b',color="C{}".format(it),label=f"M{it}")
                        ax = plot_ellipse2d_gaussian(mu[it], sigma[it], nstd =n_std, ax=ax,color="C{}".format(it))
        elif (idx_m != None) and (idx_f == None):
                for it in idx_m:
                        ax.scatter(data[it][0,:],data[it][1,:],
                                   marker='o',cmap='tab20b',color="C{}".format(it),facecolors='none',
                                   alpha=0.25)
                for it in idx_m:
                        ax.scatter(mu[it][0],mu[it][1],marker='o',cmap='tab20b',color="C{}".format(it),label=f"H{it}")
                        ax = plot_ellipse2d_gaussian(mu[it], sigma[it], nstd =n_std, ax=ax,color="C{}".format(it))
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title)
        ax.legend()
        ax.set_aspect('equal', adjustable='datalim')

def plot_ellipse2d_gaussian(mu, sigma,nstd=3. , ax=None,color='red',cmap='viridiris'):
        
    eig_values, eig_vectors=np.linalg.eigh(sigma)
    order = eig_values.argsort()[::-1]
    eig_values=eig_values[order]
    eig_vectors=eig_vectors[:,order]
    
    #Hallamos el angulo de la ellipse
    theta = np.degrees(np.arctan2(*eig_vectors[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2. * nstd * np.sqrt(eig_values)
    
    
    ellip = Ellipse(xy=mu, width=width, height=height, angle=theta, linewidth=1,facecolor=None,edgecolor=color,fill=None)
    
    ax.add_artist(ellip)
    return ax

             
def Plot_PSD(X,t,labels=[],title="",xlab="",ylab=""): 
        fig=plt.figure(figsize=(8,8))
        n_data=X.shape[0]
        for it in range(n_data):
                plt.plot(t,X,label=labels[it])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.show()       
        
def Plot_Data(X, ):
        n_rows=X.shape[0]
        
        fig, axs = plt.subplots(ncols=1,nrows=n_rows,sharex=True)
        fig.set_size_inches((20,20))
        y_ticklabels = ['Fp1']*n_rows

        positions_vlines= np.linspace(n_samples, n_samples*(n_epochs_plot-1),n_epochs_plot-1)
        positions_ticks = np.linspace(0, n_samples*(n_epochs_plot-1),n_epochs_plot)+500
        labels_ticks = np.linspace(ini+1,ini+n_epochs_plot, n_epochs_plot)
        
        for it in range(n_rows): 
                axs[it].plot(X[it,:],linewidth=0.2,color='k')
                ymin=-np.max(np.abs(X[it,:]))
                ymax=np.max(np.abs(X[it,:]))
                axs[it].set_yticks([])
                axs[it].set_ylabel(y_ticklabels[it],rotation=45.,fontsize=8)
                axs[it].set_xlim((0,n_epochs_plot*n_samples))

        axs[0].set_xticks(positions_ticks,labels=labels_ticks) 
        plt.subplots_adjust(wspace=0, hspace=0)

def Plot_EEG(X,t,events=[],labels=[],xlab="",ylab=""): 
        n_data=X.shape[0]
        fig, axs = plt.subplots(n_data,sharex=True)
        fig.set_size_inches(8,8)
        
        for it in range(n_data):
                axs[it].plot(t,X[it])
                axs[it].set_title(labels[it])
                ymin=X[it].min()
                ymax=X[it].max()
                if len(events)>0:
                        for it2 in range(len(events)):
                                axs[it].axvline(events[it2],ymin=ymin,ymax=ymax)
                axs[it].set(ylabel=ylab)
                axs[it].grid(True)
        plt.xlabel(xlab)        
        plt.show()
                
def Plot_PSD(X,t,labels=[],title="",xlab="",ylab=""): 
        fig=plt.figure(figsize=(8,8))
        n_data=X.shape[0]
        for it in range(n_data):
                #plt.plot(t,X[it],label=labels[it])
                plt.plot(t,X[it])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.show()


