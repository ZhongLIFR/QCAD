# QCAD
Paper submitted to DAMI

## Repo Structure

the QCAD repo includes two folders, Code and Data.


### Code
Specifcially, the Code folder contains the following sub-folders:

- *Implementation*: which includes the implementations of contextual anomaly detection algorithms and traditional anomaly detection algorithms as follows:
  -  **QCAD**: algorithm proposed and implemented by us.
  -  **COD**: algorithm proposed by [Song, Xiuyao, et al. 2007](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=conditional+anomaly+detection&btnG=#d=gs_cit&t=1660120134736&u=%2Fscholar%3Fq%3Dinfo%3ANRj9x9XFmTIJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den); implemented by us.
  -  **ROCOD**: algorithm proposed by [Liang, Jiongqian, and Srinivasan Parthasarathy. 2016](https://dl.acm.org/doi/pdf/10.1145/2983323.2983660); implemented by us.
  -  **LoPAD**: algorithm proposed by [Lu, Sha, et al.](https://link.springer.com/chapter/10.1007/978-3-030-47436-2_50); implemented by us.
  -  **PyODtest**: other traditional anomaly detection algorithms such as KNN,LOF,SOD,IForest, HBOS, implemented by [Yue Zhao, Zain Nasrullah, and Zheng Li., 2019](https://www.jmlr.org/papers/volume20/19-011/19-011.pdf?ref=https://githubhelp.com); API written by us.
- *Utilities*: which contains some utility function/scripts as follows:
  -  **SynDataGen** is used to generate synthetic datasets.
  -  **ContextualAnomalyInject**  is used to inject contextual anomalies.
  - **FindMB** (R)  is used to find Markov Blankets for the **LoPAD** algorithm.
- *Examples*: which contains the following scripts used to generate examples in our paper. 
  - **ExampleFootball**  is used to generate the application example;
  -  **ExampleQuantileHeight**  is used to generate the figures in Introduction section;
  -   **ExampleBeanPlot**  is used to generate the Beanplot in the Method section;
- *AblationStuides*: (TD)
- *RuntimeAnalysis*: (TD)
- *SensitivityStudies*: (TD)
- *MultipleRunningAverage*: (TD)


### Data
Specifcially, the Data folder contains the following sub-folders:

- *RawData*: 20 real-world datasets without contextual anomalies (assumption)
- *SynData*: 10 synthetic datasets withous contextual anomalies
- *GenData*: 20 real-world datasets with injected contextual anomalies, 10 synthetic datasets with contextual anomalies, and the Markov Blankets of these 30 datasets in the subfolder ~/MB/
- *Examples*: the football dataset with unkown real-world contextual anomalies
