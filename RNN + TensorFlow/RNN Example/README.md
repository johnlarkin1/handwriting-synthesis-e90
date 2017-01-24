### Background
##### John Larkin and Tom Wilmots 
This example is pulled from (https://www.tensorflow.org/tutorials/recurrent/)
It is about using the Penn Tree Bank (PTB) dataset for predicting the next words in a text given a history of previous words. It should be literally intuitive as to why this is going to be a good topic for a neural network with memory taken into account. This is exactly what the recurrent neural networks are giving us. 

In order to run this example, you can run it in one of three different ways:
`python ptb_word_lm.py --data_path=simple-examples/data/ --model small`
`python ptb_word_lm.py --data_path=simple-examples/data/ --model medium`
`python ptb_word_lm.py --data_path=simple-examples/data/ --model large`

Small should not take too long. The perplexity should reach below 120 for the test set with small, the large one below 80, *though it might take several hours to train*.
