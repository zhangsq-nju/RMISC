The codes related to TimesFM are adapted from https://github.com/google-research/timesfm.

First, download the official source code and place it under the TimesFM-train directory as follows:

official_sources/timesfm
official_sources/transformers

Clone the repositories:

cd ./official_sources
git clone https://github.com/google-research/timesfm.git

cd ./official_sources
git clone https://github.com/huggingface/transformers.git

Replace the code in:

/official_sources/transformers/src/transformers/models/timesfm2_5/modeling_timesfm2_5.py

with the corresponding code from the library.

Similarly, replace the code in:

/official_sources/transformers/src/transformers/models/timesfm2_5/modular_timesfm2_5.py

with the corresponding code from the library.

Then, you can start training TimesFM model with train.py.