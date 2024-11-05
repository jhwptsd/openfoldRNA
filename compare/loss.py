import numpy as np

def RMSD(p1, p2):
    assert len(p1) == len(p2)
    out = 0
    for i in range(len(p1)):
        out += (np.linalg.norm(p1[i] - p2[i]))**2
        
    return np.sqrt(out/len(p1))

def tm_score(p1, p2, lt):
    d0 = lambda l: 1.24 * np.cbrt(l-15) - 1.8
    sum = 0
    for i in range(len(p1)):
        sum += 1/(1+np.power(np.abs(np.linalg.norm(p1[i]-p2[i]))/d0(lt),2))
    sum /= lt
    
    return sum