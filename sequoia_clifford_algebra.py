import pdb
import numpy as np

def cos_alkashi(x,y,z):
    return (-z**2+(x**2+y**2))/(2*x*y)

def compute_cos_omega(x, y, a, b, c, d):
    cos_alpha = compute_cos_alpha(b, x, c) 
    cos_beta = compute_cos_beta(b, y, d)
    cos_gamma = compute_cos_gamma(x, y, a) 
    num = cos_gamma - cos_alpha*cos_beta 
    den = np.sqrt(1-cos_alpha**2)*np.sqrt(1-cos_beta**2)
    return num/den


def compute_cos_omega_from_angles(cos_alpha, cos_beta, cos_gamma):
    num = cos_gamma - cos_alpha*cos_beta 
    den = np.sqrt(1-cos_alpha**2)*np.sqrt(1-cos_beta**2)
    return num/den

def compute_cos_alpha(b, x, c):
    return cos_alkashi(b, x, c)

def compute_cos_beta(b, y, d):
    return cos_alkashi(b, y, d)

def compute_cos_gamma(x, y, a):
    return cos_alkashi(x, y, a)

def first_triple(x, b, c):
    return x**2 + b**2 - c**2

def second_triple(b, y, d):
    return b**2 + y**2 - d**2 


def Delta(x,y,a):
    return x**2 + y**2 - a**2

def correct_Gamma(x, y, a, b, c):
    val = 4*(b**2)*(x**2) - (first_triple(x, b, c))**2
    return np.sqrt(val)


def compute_formula(x,y, a,b,c,d):
    num = 2*(b**2)*Delta(x,y,a) - first_triple(x,b,c)*second_triple(b,y,d) 
    den = correct_Gamma(x, y, a, b, c)*np.sqrt(4*(y**2)*(b**2) - (second_triple(b,y,d))**2)
    return num/den

def compute_distance(alpha, beta):
    return np.linalg.norm(alpha - beta)

def generate_noise(mu, sigma, n):
    epsilon = sigma*np.random.randn(6*n, 1) + mu
    return epsilon

def unit_project(x):
    return max([-0.99, min([x,0.99])])

def projectMatrixPosDef(M):
    # M should be the matrix of pseudo distances
    (n_m, m_m) = M.shape
    J = np.eye(n_m) - (1/n_m)*np.ones((n_m,1))
    G = - 0.5*(J.dot(M**2)).dot(J.transpose())

    lambdas, eigvs = np.linalg.eig(G)
    
    lambdas_0 = np.max([lambdas, [0. for i in range(n_m)]], 0)
    D_projected = np.diag(lambdas_0)

    P = eigvs
    projected_G = P.dot(D_projected).dot(P.transpose()) 

    diag_projected_G = np.diag(projected_G).reshape((n_m, 1))
    projected_M = np.ones((n_m, 1)).dot(diag_projected_G.transpose()) - 2*projected_G + diag_projected_G.dot(np.ones((1, n_m)))

    return np.sqrt(projected_M)

