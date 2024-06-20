# Systematic Review Data

These datasets were kindly provided by the Royal Australian College
of Surgeons systematic review team ASERNIP-S.

* [CARGEL](CARGEL) : Ligament substitution
* [CIDP](CIDP) : Immunoglobulin for Chronic Inflammatory Demyelinating Polyneuropathy
* [EMR](EMR) : Endoscopic Mucosal Resection
* [ESG](ESG) : Endoscopic Sleeve Gastroplasty (Class I and Class II obesity)
* [IORT](IORT) : Targeted Intraoperative Radiotherapy (T-IORT) Early-Stage Breast Cancer
* [LANB](LANB) : Local anaesthesia nerve blockade (LANB) for post-surgical analgesia 
* [LMTA](LMTA) : Lung MTA
* [PBRT](PBRT) : Partial Breast Radiotherapy
* [PID](PID) : Primary Immunodeficiency, Intervention: antibody infusion
* [VERT](VERT) : Stabilizing compression fractures in the spine.

These datasets are combined into an initial training dataset by combining the
research question text with the results of the database search using the script:
[JOIN_DATA_TO_QUESTIONS.sh](../scripts/JOIN_DATA_TO_QUESTIONS.sh).
However, to see all the processing required to prepare the data for the machine
learning experiments see the [Data Processing RUN script](../RUN_02_PREPARE_DATA.sh).

## Analysis

The final data was analysed and summarised to create tables for the publication
[Analyse Data](scripts/analyse_datasets.py)
 
