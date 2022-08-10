# QCAD
Paper submitted to DAMI (Zenodo)

## Repo Structure

the QCAD repo includes two folders, Code and Data.


### Code
Specifcially, the Code folder contains the following sub-folders:

- *Implementation*: which includes the implementations of our contextual anomaly detection algorithm **QCAD**, and other SOTA contextual anomaly detection algorithms such as **COD**, **ROCOD**, **LoPAD**, and traditional anomaly detection algorithms **PyODtest**(based on PyOD);
- *Utilities*: which contains some utility function/scripts.
  -  **SynDataGen** is used to generate synthetic datasets ;
  -  **ContextualAnomalyInject**  is used to inject contextual anomalies;
  - **FindMB** (R)  is used to find Markov Blankets for the **LoPAD** algorithm;
- *Examples*: which contains the scripts used to generate examples in our paper. **ExampleFootball**  is used to generate the application example; **ExampleQuantileHeight**  is used to generate the figures in Introduction section; **ExampleBeanPlot**  is used to generate the Beanplot in the Method section;
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
