from rakun2 import RakunKeyphraseDetector

doc = """
    A probabilistic neural network (PNN) [1] is a feedforward neural network, which is widely used in classification and pattern recognition problems.
    In the PNN algorithm, the parent probability distribution function (PDF) of each class is approximated by a Parzen window and a non-parametric function.
    Then, using PDF of each class, the class probability of a new input data is estimated and Bayes’ rule is then employed to allocate the class with highest posterior probability to new input data.
    By this method, the probability of mis-classification is minimized.[2] This type of artificial neural network (ANN) was derived from the Bayesian network[3] and a statistical algorithm called Kernel Fisher discriminant analysis.[4] 
    It was introduced by D.F. Specht in 1966.[5][6] In a PNN, the operations are organized into a multilayered feedforward network with four layers:
    """

hyperparameters = {"num_keywords": 10,
                   "merge_threshold": 1.1,
                   "alpha": 0.3,
                   "token_prune_len": 3,
                   "ngrams": 3
                   }

keyword_detector = RakunKeyphraseDetector(hyperparameters)
out_keywords = keyword_detector.find_keywords(doc, input_type="string")
print(out_keywords)

keywords = set(x[0] for x in out_keywords)
assert "probabilistic neural network" in keywords
assert "artificial neural network" in keywords
assert "multilayered feedforward network" in keywords
