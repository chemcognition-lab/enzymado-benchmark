# Enzymado Benchmark

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# TODO
 - [x] Clean datasets (Manuel)
    - [x] pHopt
    - [pending] Azoreductase data (Ricardo)
    - [x] active sites
    - [x] add solubility
    - [x] add fluo
    - [ ] move notebooks to scripts
    - [ ] create a step by step cleaning progress
 - [x] Dataset cards (json): dataset name / task type / path
    - [x] Metadata (key per dataset): dataset origin / date / size / information
    - [x] Task type (key per property): regression / class / ranking / metrics
    - [x] Features (key per dataset): sequence / smiles / identifier (for proteins and smiles)
    - [x] Labels (key per property): ec number / etc...
    - [x] Path: final dataset path
    - [ ] Add huggingface link
    - [ ] Expand dataset cards with: (Paralelizable)
      - [ ] Amount of removed data
      - [ ] Statistics?
         - [ ] Ratio enzyme/properties
         - [ ] Nans/Dups
         - [ ] Label distributions per dataset
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
 - [x] Scripts for splits (Manuel)
 - [ ] Dataloader and training scripts (Manuel)
 - [ ] Scripts for models (Ricardo)
   - [ ] Linear Regression/Logistic Regression
   - [ ] XGBoost
 - [ ] Scripts for metrics (Manuel)
 - [ ] Rating script (Manuel)
   - [ ] Packett-Luce
   - [ ] Percentages
 - [ ] Revision check (Paralelizable)
   - [ ] Max
   - [ ] Rana
   - [ ] Will
 - [ ] Leaderboards
   - [ ] Real case scenarios stream
      - [ ] Per dataset graphs
      - [ ] Global graph
   - [ ] Dataset stream
      - Linear Regression
         - [ ] Per dataset graphs
         - [ ] Global graph
      - XGBoost
         - [ ] Per dataset graphs
         - [ ] Global graph
 - [ ] Html Webpage
