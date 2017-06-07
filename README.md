# Quorus
Identifying question pairs that have the same intent. We make use of Quora dataset to solve this problem.

Run the create_dataset.py. You can adjust the size of the training and testing file by changing the appropriate lines.
Recommended: Generate full,half and quarter of the datasets.

Download the pretrained embeddings from https://nlp.stanford.edu/projects/glove/
Choose the 
"Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip" from the list. This is a huge file.

Edit the train.py and select your model from the models found in the model directory. Add your models to this directory.
Modify the train.sh script according to the parameters chosen.

After the model is trained, evaluate the model using the test.sh script



