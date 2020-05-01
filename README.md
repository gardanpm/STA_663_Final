# STA_663_Final
   ## This Repository shows the implementation of Latent Dirichlet Allocation with collapsed gibbs sampling in python by Cole Juracek and Pierre Gardan.
    
   ### - The detailed code can be found in document src 
   The sampler file containing the actual function for the gibbs sampler. The utility file contains functions to prepare the data into usable tokens and titles while inference is made of only one function to print top words. The test file tests that some of our functions return the desired output
       
  ### - Examples on two datasets are implemented in Examples
  The 20NewsGroup dataset from scikit-learn, a dataset of articles from Reuter and the NIPS dataset extracted from   https://archive.ics.uci.edu/ml/datasets/NIPS+Conference+Papers+1987-2015are used
      
  ### - The Data folder contains multiple Reuter data files like the one used in Examples
  
  ### - In comparisons, we compare our algorithm with existing ones using different methods such as sklearn (variational bayesian inference) and gensim (PLDA).
