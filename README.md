# ODKL-residential-load-forecasting
## 🌟 Dataset:

3 publicly available residential load datasets are applied:\
*[Ausgrid Resident](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data)*\
*[UMass Smart](https://traces.cs.umass.edu/index.php/Smart/Smart)*\
*[SGSC Customer Trial](https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data)*

## 🎯 Implementation:

APLF folder contains the Python file that contains all the scripts required to execute the method:\
model.py: This file contains a Tensorflow implementation of the Sparse Online Gaussian Process, which is added to a deep Soft Spiking Neuron Networks in ODKL. To implement the ODKL, just apply it directly to the project [OSTL](https://github.com/IBM/ostl).

## 📚 Citation
If you use ODKL in your research, please consider citing us.
@article{li2023residential,
  title={Residential load forecasting: An online-offline deep kernel learning method},
  author={Li, Yuanzheng and Zhang, Fushen and Liu, Yun and Liao, Huilian and Zhang, Hai-Tao and Chung, Chiyung},
  journal={IEEE Transactions on Power Systems},
  year={2023},
  publisher={IEEE}
}
