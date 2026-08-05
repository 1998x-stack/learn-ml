from src.mini_sklearn.base import BaseEstimator, clone


class _Estimator(BaseEstimator):
    def __init__(self, alpha: float = 0.5, max_iter: int = 100):
        self.alpha = alpha
        self.max_iter = max_iter

def test_get_params_returns_only_hyperparameters():
    est = _Estimator(alpha=0.1, max_iter=50)
    params = est.get_params()
    assert params == {"alpha": 0.1, "max_iter": 50}

def test_set_params_returns_self_and_updates():
    est = _Estimator()
    out = est.set_params(alpha=2.0)
    assert out is est
    assert est.alpha == 2.0

def test_clone_returns_new_object_with_equal_hyperparams():
    est = _Estimator(alpha=0.3, max_iter=500)
    cloned = clone(est)
    assert cloned is not est
    assert cloned.get_params() == est.get_params()
