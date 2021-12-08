import numpy as np
import unittest
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS
from sklearn.gaussian_process.kernels import RBF

"""
class DiffusionDistance
    
    kernel
    X
    t_max
    stationnary_distrib
"""
class DiffusionDistance:
    kernel = None
    X = None
    t_max = None
    M = None
    P = None
    pi = None
    G = None
    def __init__(self, kernel, X, t_max) -> None:
        self.kernel= kernel
        self.X = X
        self.t_max = t_max
        
    def fit(self):
        compute_density_norm_matrix(self)
        compute_diffusion_Matrix(self)
        compute_stationnary_distrib(self)
        return
    
    def fit_transform(self):
        self.fit()
        embedding = MDS(n_components=2)  #Multidimentional scaling
        return embedding.fit_transform(self.X)
        
    
    def compute_density_norm_matrix(self):
        K = self.kernel(self.X)
        Q = np.diag(np.sum(K, axis= 1))
        self.M = np.linalg.inv(Q).dot(K).dot(np.linalg.inv(Q))
        return self.M
    
    def compute_diffusion_Matrix(self): 
        D = np.diag(np.sum(self.M, axis= 1))
        self.P = np.linalg.inv(D).dot(self.M)
        return self.P
    
    def compute_stationnary_distrib(self): 
        pi = np.sum(self.M, axis = 1)/np.sum(self.M)
        self.pi = pi
        return self.pi
    
    def distance_matrix_Pt(self, t): 
        Pt = np.linalg.matrix_power(self.P, 2**(self.t_max-t))
        return distance_matrix(Pt,Pt,1)
        
    def compute_custom_diffusion_distance(self): 
        G = np.zeros((self.X.shape[0], self.X.shape[0]))
                
        for t in range(0,self.t_max): 
            G = G + 2**(-(self.t_max-t)/2) * self.distance_matrix_Pt(t)
        G = G + 2**(-(self.t_max+1)/2) * distance_matrix(self.pi[:, None],self.pi[:, None],1)

        self.G = G
        return self.G
        
#for unit test
class testDiffusionDistance(unittest.TestCase):
    X=np.array([[0,0],[0,1], [1,0], [1,1]])
    epsilon = 2
    length_scale =np.sqrt(epsilon/2)
    kernel = 1.0 * RBF(length_scale)
    DD = DiffusionDistance(kernel, X, 1)
    
    def exemple(self):
        #example
        #self.assertEqual('foo'.upper(), 'FOO')
        #self.assertTrue('FOO'.isupper())
        #self.assertFalse('Foo'.isupper())
        return
    
    def test_compute_density_norm_matrix(self):
        M = self.DD.compute_density_norm_matrix()
        print("\n", self.DD.M)

    def test_compute_diffusion_Matrix(self): 
        M = self.DD.compute_density_norm_matrix()
        P = self.DD.compute_diffusion_Matrix()
        print("\n", self.DD.P)

    def test_compute_stationnary_distrib(self):
        M = self.DD.compute_density_norm_matrix()
        P = self.DD.compute_diffusion_Matrix() 
        pi = self.DD.compute_stationnary_distrib()
        print("\n",self.DD.pi)
        
    def test_compute_custom_diffusion_distance(self):
        M = self.DD.compute_density_norm_matrix()
        P = self.DD.compute_diffusion_Matrix() 
        pi = self.DD.compute_stationnary_distrib()
        
        G = self.DD.compute_custom_diffusion_distance()
        print("\n", self.DD.G)
        
if __name__ == '__main__':
    unittest.main()