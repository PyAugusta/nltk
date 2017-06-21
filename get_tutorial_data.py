import os
from nltk import download

install_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(install_dir, 'tutorial_data')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
#download('twitter_samples', download_dir=data_dir)
download('all')