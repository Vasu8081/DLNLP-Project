Download the glove vector embeddings to glove1/glove.twitter.27B.50d.txt and glove/glove.6B.50d.txt.
First run the DataPreprocessing.py file then run the createEmbeddings.py file which will create all required data then

Run the model1.ipynb cells it will run and start the training process
We can use evaluate method to test the sentence
LoadTest.ipynb is used to calculate the scores of predictions.
Can use checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)) to load to previous checkpoints
