"""mini_sklearn.naive_bayes — generative probability classifiers (s11 / v0.3.3).

Naive Bayes models each class as a **generator** of its features and predicts
with the Bayes rule, in log space to dodge underflow.  The "naive" assumption
is per-class feature independence, which keeps the learner simple, fast and
stable.
"""

from ._nb import BernoulliNB, GaussianNB, MultinomialNB

__all__ = ["GaussianNB", "MultinomialNB", "BernoulliNB"]
