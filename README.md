
# WHAT IS Genome Interpretation?#
Genome Interpretation (GI) is the challenging endeavor of modeling and understanding the genotype-phenotype relationship. Thanks to the flexibilit of Neural Networks (NN) methods, it is now possible to build ad-hoc models to tackle this challenge. 

In our vision, phenotype-specific NN-based GI methods can be build to directly learn from data the relationship between genotype and phenotype. This "genotype in, phenotype out" paradigm can be used for the binary classification of cases vs controls (https://academic.oup.com/nargab/article/2/1/lqaa011/5742219?login=false), or the multi-task prediction of quantitative traits (https://academic.oup.com/nar/article/50/3/e16/6430850?login=false).

# WHAT IS Federated Learning? #

Federated Learning (FL) is a recently proposed ML paradigm that allows advanced ML methods (including Deep Learning (DL)) to be trained and tested in a collaborative and parallel way that does not require the actual exchange of sensitive data between partners. FL was originally introduced as a distributed ML paradigm to allow the training of a centralized model with privacy-sensitive data coming from a large numbers of clients (i.e., millions of mobile devices). Recently, the FL paradigm showed promise in the application of ML on health records and medical imaging data.

# WHAT IS the goal of this repo?#

Here we present FedCrohn. FedCrohn is the Federated Learning implementation of the CDkoma exome-based Crohn's Disease risk predictor we published in (https://academic.oup.com/nargab/article/2/1/lqaa011/5742219?login=false).

# WHY IS FedCrohn relevant?#

FedCrohn constitutes a self-contained proof-of-concept that FL can be used to train NN-based GI interpretation methods that could be useful for clinical genetics and Precision Medicine. 

In particular, FedCrohn focuses on the binary classification of Crohn's Disease (CD) patients from just *their exome sequencing data*.

We analyse one NN-based GI methods for the classification of CD cases vs controls, and we compare the performances obtained in conventional ML settings with the ones that can be obtained in FL settings, *showing that they are extremely similar*.

This indicates that FL could be a viable way to develop new NN-based GI methods.

In particular, this repo contains the data and the code to run the two experiments described in FedCrohn paper (under review).


# WHY FL is important? #

Genome Interpretation is a very challening endeavor. After an initial surge of interesting results, due  to the large amounts of genomics data produced by "next generation sequencing" technologies, it started to clearly appear that the issues hindering our advances in understanding our genome shifted from the initial "data availability" issue (too few sequencing samples), to the current "data interpretation" issue. 

Genomic data are indeed extremely intricate, and we need to apply the latest data science methods (e.g. ML, DL) to have a chance to make sense of them. The approach of applying the latest ML/DL methods to Structural Biology, for example, lead to the astonishing results provided by AlphaFold. 

DL methods require *very large* dataset sizes, and when it comes to genomics data, it's not easy to gather such large datasets. Genomics and phenomics data are indeed extremely sensitive, and dealing with them requires solving a number of ethical, legal, infrastructural and privacy-related issues. Sharing genomics and phenomics data *is not as simple as sharing structural biology data*.

But what if we could apply the latest DL/ML methods to the GI problem *without having to share/move/aggregate the data in the first place?*

That is what FL can do for the advancement of the GI field of research.

### How do I set it up? ###

We recommend to use miniconda to create an environment to run FedCrohn.

Here we show how to create a miniconda environment containig all the required libraries. Similar instructions can be used to install the dependencies with `pip`. 
FedCrohn runs on python 3.x .

* Download and install miniconda from `https://docs.conda.io/en/latest/miniconda.html`
* Create a new conda environment by typing: `conda create -n FedCrohn -python=3.7`
* Enter the environment by typing: `conda activate FedCrohn`
* Install pytorch >= 1.0 with the command: `conda install pytorch -c pytorch` or refer to pytorch website https://pytorch.org
* Install scipy with the command: `conda install numpy scipy`
* Refer to `https://flower.dev/docs/ ` to install the FL library `flower	`

You can remove this environment at any time by typing: conda remove -n FedCrohn --all
 

### What this repository contains? ###

* `standaloneFL.py` -> the script needed to run Experiment 2
* `flClient.py` -> The FL client to run Exp1.
* `flServer.py` -> The FL server to run Exp1.
* `NXTfusion` -> folder containing core source code necessary to run NXTppi
* `marshalledP3/ , phenopediaCrohnGenesmodels/` -> folders containing data needed by the scripts
* `sources/` -> additional functions needed by the scripts

### How to run Exp1 ###


The `marshalledP3` folder contains 3 CD cases controls datasets. They are a processed binarized version derived from the ones used in the CAGI challenges (https://genomeinterpretation.org/challenges.html). For simplicity, we will call tehse datasets as 2, 3 and 4 from now on.

In the Exp1 we will run FedCrohn in a client-server settings. In particular, `flServer.py`will use one dataset to evaluate the performance of the learned FL model, and the other two clients `flClient.py` will train the shared model on their data independently.

To run Exp1, open 3 bash shells. Activete the right conda environment in all of them by typing `conda activate FedCrohn`

Then launch the server in one of the shells:

`$ python flServer.py 4`

Then you can launch the clients in the remaining two shells:

`$ python flClient.py 3` 

`$ python flClient.py 2`


These simple commands will ensure that the flServer starts, and waits for flClients to connect.

Once both of them they are connecteed, the server initializes an empy NN model and sends it to the clients. The clients train it on their data, and send the resulting model to the server. The server averages these models, obtaining a consensus model. This is 1 *round* of FL training. This script repeats this process for 5 rounds, and prints the final validation performances, computed by the server on its locally controlled dataset.

Feel free to assign different combinations of 2,3,4 datasets to the server and the clients, and to expxeriment with the code to see what happens!

Exp1 shows that a NN GI model specifically designed for CD prediction can be trained in FL settings. Results are comparable with the same model on centralized data.

### How to run Exp2 ###

In Exp2 we have a single script simulating the server-clients interaction `standaloneFL.py`. 

This script reads ALL the 2,3,4 datasets, combines them into a single one, and let you select and arbitrary (recommended 3-31) number of clieints. Exp2 simulates the behavior of FL GI methods when tens of centers are collaborating, even if each of the controls very small datasets (with 31 clients, less than 10 samples are controlled by FL partner (e.g. each client)).

To run Exp2, just type:


`$ standaloneFl.py 4` 

It will automatically randomly split the data, assigning them to 3 centers (one dataset is used by the server for performance evaluation).

Exp2 simulates the performance of a FL GI methods for CD prediction when several small datasets are combined into a single GI FL effort.


### Who do I talk to? ###

Please report bugs at the following mail address:
daniele DoT raimondi aT kuleuven DoT be

## Dataset Setup

This project requires external datasets that are NOT included in this repository.

Download from:
- Dataset for FedEnhanced (Kaggle dataset): https://www.kaggle.com/datasets/evanlukedsouza/datasets-for-fedenhanced

Place them in:

data/
  marshalledP3/
  phenopediaCrohnGenes/
  string_db/