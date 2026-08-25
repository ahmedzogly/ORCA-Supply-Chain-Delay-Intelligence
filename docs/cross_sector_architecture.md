# Cross-Sector Architecture

To support cross-domain evaluation, the repository uses explicit Dataset Adapters mapping raw files to a common feature ontology.

SCMS remains the core domain. DataCo and Olist are loaded independently via DataCoAdapter and OlistAdapter. No datasets are arbitrarily appended or merged.
