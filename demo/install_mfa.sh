git clone https://github.com/RobinWitch/Montreal-Forced-Aligner.git
cd Montreal-Forced-Aligner
pip install -e .
conda install -c conda-forge kalpy
pip install pgvector
pip install Bio
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa