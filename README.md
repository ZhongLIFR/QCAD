# QCAD
Paper submitted to DAMI

## Repo Structure

the QCAD repo includes two folders, Code and Data.


### Code
Specifcially, the Code folder contains the following sub-folders:

- *Implementation*: which includes the implementations of contextual anomaly detection algorithms.
  -  **QCAD**: our own algorithm
  -  **COD**: algorithm by [Song, Xiuyao, et al. 2007](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=conditional+anomaly+detection&btnG=#d=gs_cit&t=1660120134736&u=%2Fscholar%3Fq%3Dinfo%3ANRj9x9XFmTIJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den); implemented by us.
  -  **ROCOD**, algorithm by [Liang, Jiongqian, and Srinivasan Parthasarathy. 2016](https://dl.acm.org/doi/pdf/10.1145/2983323.2983660); implemented by us.
  -  **LoPAD**, 
  -  and traditional anomaly detection algorithms **PyODtest**(based on PyOD);
- *Utilities*: which contains some utility function/scripts.
  -  **SynDataGen** is used to generate synthetic datasets ;
  -  **ContextualAnomalyInject**  is used to inject contextual anomalies;
  - **FindMB** (R)  is used to find Markov Blankets for the **LoPAD** algorithm;
- *Examples*: which contains the scripts used to generate examples in our paper. 
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
