# QCAD
Paper submitted to DAMI (Data Mining and Knowledge Discovery journal)

## Repo Structure

the QCAD repo includes two folders, Code and Data.


### Code
Specifcially, the Code folder contains the following sub-folders:

- *Implementation*: which includes the implementations of contextual anomaly detection algorithms and traditional anomaly detection algorithms as follows:
  -  **QCAD.py**: algorithm proposed and implemented by us.
  -  **CAD.py**: algorithm proposed by [Song, Xiuyao, et al. 2007](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=conditional+anomaly+detection&btnG=#d=gs_cit&t=1660120134736&u=%2Fscholar%3Fq%3Dinfo%3ANRj9x9XFmTIJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den); implemented by us.
  -  **ROCOD.py**: algorithm proposed by [Liang, Jiongqian, and Srinivasan Parthasarathy. 2016](https://dl.acm.org/doi/pdf/10.1145/2983323.2983660); implemented by us.
  -  **LoPAD.py**: algorithm proposed by [Lu, Sha, et al.](https://link.springer.com/chapter/10.1007/978-3-030-47436-2_50); implemented by us.
  -  **PyODtest.py**: other traditional anomaly detection algorithms such as KNN,LOF,SOD,IForest, HBOS, implemented by [Yue Zhao, Zain Nasrullah, and Zheng Li., 2019](https://www.jmlr.org/papers/volume20/19-011/19-011.pdf?ref=https://githubhelp.com); API written by us.
- *Utilities*: which contains some utility function/scripts as follows:
  -  **SynDataGen.py**: generate synthetic datasets.
  -  **ContextualAnomalyInject.py**: inject contextual anomalies.
  - **FindMB.R**: find Markov Blankets for the **LoPAD** algorithm.
- *Examples*: which contains the following scripts used to generate examples in our paper. 
  - **ExampleFootball.py**: generate the football application example in the Experiment Results section;
  -  **ExampleQuantileHeight.py**: generate the figures in the Introduction section;
  -   **ExampleBeanPlot.py**: generate the Beanplot in the Method section;
- *AblationStuides*: which investigate the impacts of different components on detection performance.
  - **AblationStudy.py**: conduct two ablation stuides.
- *RuntimeAnalysis*: which inspects the computational cost of **QCAD** and **CAD**.
  - **RuntimeBehave.py**: inspect the running time by varying the number of behaviroual features.
  - **RuntimeContext.py**: inspect the running time by varying the number of contextual features.
  - **RuntimeSample.py**: inspect the running time by varying the number of samples.
- *SensitivityStudies*: which investigate the impact of parameter *k*.
  - **SensitivityOfNeighbours.py**: inspect the detection accuracy in terms of RUC AUC, PR AUC, P@n by varying the number of neighbours.
- *MultipleRunningAverage*: which run all involved detection algothms 10 times independently.
  - **AverageTest.py**: execute all anomaly detection algorithms except **CAD** on 20 real-world datasets 10 times, respectively.
  - **AverageTestCOD.py**: execute **CAD** separately on 20 real-world datasets 10 times, respectively. This is because it takes a long time.
  - **SynAverageTest.py**: execute all anomaly detection algorithms except **CAD** on 10 synthetic datasets 10 times, respectively.
  - **SynAverageTestCOD.py**: execute **CAD** separately on 10 synthetic datasets 10 times, respectively. This is because it takes a long time.

### Data
Specifcially, the Data folder contains the following sub-folders:

- *RawData*: 20 real-world datasets without contextual anomalies (assumption)
- *SynData*: 10 synthetic datasets withous contextual anomalies
- *GenData*: 20 real-world datasets with injected contextual anomalies, 10 synthetic datasets with contextual anomalies, and the Markov Blankets of these 30 datasets in the subfolder ~/MB/
- *Examples*: the football dataset with unkown real-world contextual anomalies
- *TempFiles*: temporary results
