## Title
Learning Efficient Multi-view Graph Embedding Features for Single-view Data Clustering

## Usage
Before training, please download the models [edge predictor (extraction code is <strong>kgzy</strong>)](https://pan.baidu.com/s/1PftfxtWd5esBABocOmBNEA) and put them into the floder ./data/YOUR_DATASETS_NAME/edge_probabilities.
### Training
For example, an example run on the Cora dataset is:
```
python3 ./Code/main_cora.py
```
## Results
The visualization clustering comparisons on the Cora, Citeseer and Pubmed of our proposed approach with
R-DGAE, MAGCN and GMM-VGAE.
### Cora 
<p align="center">
  <img width="960" height="190" src="/images/image1.png"/>
</p>

### Citeseer 
<p align="center">
  <img width="960" height="190" src="/images/image2.png"/>
</p>

### Pubmed
<p align="center">
  <img width="960" height="190" src="/images/image3.png"/>
</p>

## Requirements
The code is buildt with:

* Python 3.7.6
* Pytorch 1.10.0
* Scikit-learn 1.0.2
* Scipy 1.4.1
* Torch-cluster 1.6.0
* Torch-geometric 1.4.3
* Torch-scatter 2.0.9
* Torch-sparse 0.6.13
* Torch-spline-conv 1.2.1
* Networkx 2.1

## Datasets

### Cora (https://linqs.org/datasets/#cora):
The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links.

### Citeseer (https://linqs.org/datasets/#citeseer-doc-classification):
The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. The citation network consists of 4732 links.

### Pubmed (https://linqs.org/datasets/#pubmed-diabetes):
The Pubmed Diabetes dataset consists of 19717 scientific publications from PubMed database pertaining to diabetes classified into one of three classes. The citation network consists of 44338 links.


