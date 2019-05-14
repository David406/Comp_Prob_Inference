import numpy as np

def compute_mutual_info(joint_prob_XY):
    """
    Computes the information divergence between the
    two distributions in joint_distribution
    
    Input
    -----
    joint_prob_XY: A 2D numpy array corresponding to the
        joint distribution of two random variables X and Y
        
    Output
    ------
    IXY: The information divergence between X and Y, 
        computed by I(X,Y) = D(P(x,y)||PxPy)
    """
    
    prob_X = joint_prob_XY.sum(axis=1)
    prob_Y = joint_prob_XY.sum(axis=0)
    
    joint_prob_XY_indep = np.outer(prob_X, prob_Y)
    
    info_divergence = lambda p, q: np.sum(p * np.log2(p/q))
    
    IXY = info_divergence(joint_prob_XY, joint_prob_XY_indep)
    
    return IXY

def main():
    joint_prob_XY = np.array([[0.1, 0.09, 0.11], \
                              [0.08, 0.07, 0.07], \
                              [0.18, 0.13, 0.17]])
    IXY = compute_mutual_info(joint_prob_XY)
    
    print(IXY)
    
if __name__ == '__main__':
    main()
    
    
    
    