# Exploring the statistics of language via LLMs and random tree models

## Contents

- **`hierarchical_chunking_GPT4_*.ipynb`**  
  Notebook for generating token trees using GPT-4o. Computation can be expensive (for a fixed K, roughly \$50–\$80 for 26 stories), so all precomputed tree data is stored under `data/labov_trees`. This notebook relies on helper functions in **`hierarchical_chunker_utilities_GPT4_*.py`**.

- **`1-pt_function_statistics_*.ipynb`** 
  Notebook for analyzing the 1-pt functions of the generated token trees and comparing them with the theory of random trees. Computing the theoretical curves can also be time-consuming (about one hour per fixed K across different story lengths), so all precomputed theory curves are stored in `data`. This notebook depends on helper functions in **`hierarchical_chunker_utilities_GPT4.py`** and **`RTM_Theory_utilities.py`**.

- **`2-pt_function_statistics_*.ipynb`**
(Work in progress) Notebook for analyzing the 2-pt functions of the generated token trees and comparing them with the theory of random trees. Still has a mismatch between theory and experiment. This notebook depends on helper functions in **`hierarchical_chunker_utilities_GPT4.py`** and **`RTM_Theory_utilities.py`**.