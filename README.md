<p align="center">
   <h2 align="center">Residential Load Forecasting: An Online-Offline Deep Kernel Learning Method</h2>
 <p align="center">
 <h3 align="center">IEEE Transactions on Power Systems ( Volume: 39, Issue: 2, March 2024) </h3>

A deep kernel is proposed by integrating the deep soft Spiking Neural Networks, which is then applied to
perform Gaussian Process (GP) regression. The constructed regressor investigates the temporal dynamics within the time-series and retains the probabilistic advantages for uncertainty estimates.

## 🌟 Dataset:

 3 publicly available residential load datasets are applied:
- *[Ausgrid Resident](https://github.com/pierre-haessig/ausgrid-solar-data?tab=readme-ov-file#:~:text=Personal%20repository%20on%20the%20analysis%20of%20the%20Solar%20home%20electricity)*: Loads of 300 households in the Australian distribution network are released for public utilization.
- *[UMass Smart](https://traces.cs.umass.edu/index.php/Smart/Smart)*: Multiple smart meters’ readings
of 7 homes are collected by the UMass Smart Home project in America from 2014 to 2016.
- *[SGSC Customer Trial](https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data): This dataset stems from the Smart Grid Smart City (SGSC) project in Australia since 2010.*

## 🎯 Implementation:

- Install the packages in requirement.txt
- The demo.py shows how to apply the proposed model for load forecasting
  - download file '2012-2013 Solar home electricity data v2.csv' from ausgrid resident dataset
  - run demo.py

## 🤗 Citation

If you use ODKL in your research, please consider citing us.
```bibtex
@article{li2023residential,
  title={Residential load forecasting: An online-offline deep kernel learning method},
  author={Li, Yuanzheng and Zhang, Fushen and Liu, Yun and Liao, Huilian and Zhang, Hai-Tao and Chung, Chiyung},
  journal={IEEE Transactions on Power Systems},
  year={2023},
  publisher={IEEE}
}
```

## 📚 Acknowledgement:
This repositories is based on the work https://github.com/IBM/ostl.
