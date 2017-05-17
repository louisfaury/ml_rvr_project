# AML_RVR-SVR_project

## Running the project 
The MATLAB project is decomposed in 9 sections. Simply comment / uncomment them regarding on which you desired to run. 
### S1 : Loads datasets 
Change the name of the dataset (*sinc* or *airfoils*) regarding which you desire to load.  
### S2 : Run SVR
Runs the SVR with the kernel, the SVR method you choose (nu,epsilon SVR), the hyperparameters indicated.
### S3 : Run RVR
Runs the RVR with the indicated kernel width. 
### S4 : BICSR validation 
Runs cross-validation on several models (arbitrary model as well as the *optimal models* according to different metrics). Plots the result on a MSE/sparsity graph. 
### S5 : CV for nu-SVR
Runs f-fold cross-validation for the nu-SVR (with RBF Gaussian kernel), with grid search over the hyperparameters. 
### S6 : CV for eps-SVR
Runs f-fold cross-validation for the eps-SVR (with RBF Gaussian kernel), with grid search over the hyperparameters.
### S7 : CV for RVR
Runs f-fold cross-validation for the RVR (with RBF Gaussian kernel), with grid search over the hyperparameters.
### S8 : BICSR validation (2)
Basically performs the same as S4, without different penalizing terms (stays with klnN)
### S9 : Model comparison 
Plots the different optimal models (according to BICSR and MSE) on the MSE / sparsity graph. 

## Library gestion
* Put libsvm source code in libsvm/ (and don't forget to compile !)
* Put sparseBayes source code in sparseBayes/ 
