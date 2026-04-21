# Enzymado Benchmark

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# TODO
## Stage 1: Data
 - [x] Clean datasets (Manuel)
    - [x] pHopt
    - [pending] Azoreductase data (Ricardo)
    - [x] active sites
    - [x] add solubility
    - [x] add fluo

 - [x] Dataset cards (json): dataset name / task type / path
    - [x] Metadata (key per dataset): dataset origin / date / size / information
    - [x] Task type (key per property): regression / class / ranking / metrics
    - [x] Features (key per dataset): sequence / smiles / identifier (for proteins and smiles)
    - [x] Labels (key per property): ec number / etc...
    - [x] Path: final dataset path
    - [ ] Expand dataset cards with: (Rana/Max)
      - [ ] Amount of removed data
      - [ ] Statistics?
         - [ ] Ratio enzyme/proteins
         - [ ] Nans/Dups
         - [ ] Entire protein pool cluster with MMseqs2
            - [ ] Ratio unique/variants
         - [ ] Label distributions per dataset
      - [ ] Dataset card with data information
      - [ ] Separate file for stasts and computed data
      - [ ] revise this paper: https://arxiv.org/abs/2204.01075

 - [x] Embedding scripts (Ricardo)
    - [x] ESM2
    - [x] ESM3
    - [x] ProsT5
    - [x] ProT5
    - [x] ProFluent
    - [x] ESMC
    - [x] Ankh

 - [x] Access to PC in San Diego (Ricardo)
   - [x] Using Ricardo's account

 - [ ] One consolidated dataset (Max)
   - [ ] SQL database (Max proposal)

## Stage 2: Benchmark run
 - [x] Scripts for splits (Manuel)

   ### Hackathon:
 - [x] Dataloader and training scripts (Manuel)
   
   - [x] Dataloader process #1: Active sites and Binding Sites.
      - [x] Featurizer: Protein embeddings.
      - [x] Label: Sites binary vector.

   - [x] Dataloader process #2: Kcat and KM.
      - [x] Featurizer:Protein embeddings + substrate embeddings (RDKit)
      - [x] Label: continuous labels

   - [x] Dataloader process #3:  Solub
      - [x] Featurizer:Protein embeddings
      - [x] Label: binary labels (0/1)

   - [x] Dataloader process #4: ph_opt
      - [x] Featurizer: Protein embeddings
      - [x] Label: continuos labels

   - [x] Dataloader process #5: ec_number
      - [x] Protein embeddings and binary multi labels
      - [x] For EC do BCE for multiple EC number enzymes
      - [x] Binary vector

   - [x] Dataloader process #6: binding
      - [x] Featurizer: Protein embeddings + substrate embeddings (RDKit)
      - [x] Label: Binary labels (0/1)

   - [x] Dataloader process #7: dtm and ddg
      - [x] Featurizer: Protein embeddings + mut embeddings
      - [x] Label: continuos labels

 - [ ] Scripts for models (Ricardo)
   - [ ] Linear Regression/Logistic Regression
   - [ ] XGBoost
   - [ ] How do we reduce dimensionality better (question to answer)
 - [ ] Scripts for metrics (Manuel)

 - [ ] Rating script (Manuel)
   - [ ] GLICKO-2
   - [ ] Percentages

 - [ ] Code revision check (Will)
   - [ ] move notebooks to scripts
   - [ ] Revise scripts
   - [ ] Test scripts
   - [ ] Real case scenarios
      - [ ] Embeddings


## Stage 3: Presentation
 - [ ] Leaderboards (Ricardo/Manuel)
 
   - [ ] Real case scenarios stream
      - [ ] Per dataset graphs
      - [ ] Global graph

   - [ ] Datasets stream (principal)
      - Linear/Logistic Regression (Only regression and classification metrics)
         - [ ] Per dataset graphs
         - [ ] Global graph

      - XGBoostRegressor / Classifier (Only regression and classification metrics)
         - [ ] Per dataset graphs
         - [ ] Global graph

      - XGBoostRanker / Classifier (Only ranking and classification metrics)
         - [ ] Per dataset graphs
         - [ ] Global graph

 - [ ] Html Webpage (Ricardo/Manuel)
 - [ ] create a step by step cleaning process sheet or instruction sheet for the project (Will)
 - [ ] Add a tuneable weight option for the properties.
