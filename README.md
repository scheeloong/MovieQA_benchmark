# To run all test
  `bash runAllTest.sh`

# To run actual TFIDF file
  `python main.py`

# Requires:
### Python 2.7
### NLTK
  pip install nltk
#### wordnet corpora
  run nltk.download() in python interpreter and download wordnet under the corpora tab. 

#### Plots Data
  Manually get the script movieqa_get_stories.sh from movieqa.cs.toronto.edu by signing an agreement. 
  cd MovieQA
  bash movieqa_get_stories.sh true

# The output files are:
  - output.log
  - correctFile.html 
  - wrongFile.html
  - testResults.txt

#### bABi dataset
sudo apt install luarocks
sudo luarocks --from=https://raw.githubusercontent.com/torch/rocks/master/ install class
sudo luarocks --from=https://raw.githubusercontent.com/torch/rocks/master/ install torch
sudo luarocks make babitasks-scm-1.rockspec 

# 15, 16, 17 should work, others do not work for now
babitasks 17 1
# Generate 1000 training data for each of 20 tasks
for i in `seq 1 20`; do babi-tasks $i 1000 > task_$i.txt; done
