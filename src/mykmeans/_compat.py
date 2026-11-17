"""Independent estimator protocol and input validation.

No scikit-learn numerical or estimator implementation is imported. Optional
scikit-learn *protocol value objects* (tags and metadata requests) are created
only when a caller asks for integration with that library. NumPy/SciPy supply
array primitives, not clustering algorithms.
"""
from __future__ import annotations

import copy
import inspect
import sys
import warnings
from numbers import Integral
from types import SimpleNamespace

import numpy as np
from scipy import sparse


class _NotFittedError(ValueError, AttributeError):
    pass


class _ConvergenceWarning(UserWarning):
    pass


# Reuse exception identities when a host has already loaded sklearn; importing
# mykmeans on its own never imports sklearn merely to obtain two exception types.
_exceptions = sys.modules.get("sklearn.exceptions")
NotFittedError = getattr(_exceptions, "NotFittedError", _NotFittedError)
ConvergenceWarning = getattr(_exceptions, "ConvergenceWarning", _ConvergenceWarning)


def check_random_state(seed):
    """NumPy RandomState contract used by the public sklearn-style interface."""
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, Integral) and not isinstance(seed, (bool, np.bool_)):
        return np.random.RandomState(int(seed))
    raise ValueError("random_state must be None, an integer, or numpy.random.RandomState")


def check_array(array, *, accept_sparse=False, dtype=np.float64, order=None,
                copy=False, force_writeable=False, ensure_2d=True, allow_nd=False,
                ensure_min_samples=1, ensure_min_features=1, ensure_all_finite=True,
                **kwargs):
    """Validate numeric dense/sparse input without calling sklearn validators."""
    if kwargs:
        raise TypeError(f"Unsupported validation options: {sorted(kwargs)}")
    is_sparse = sparse.issparse(array)
    if is_sparse and not accept_sparse:
        raise TypeError("Sparse data was passed, but dense data is required")
    if is_sparse:
        a = array
        # Compiled kernels trust index buffers, so reject malformed user-created
        # compressed sparse matrices before format conversion touches them.
        if hasattr(a, "check_format"):
            a.check_format(full_check=True)
        if np.issubdtype(a.dtype, np.complexfloating):
            raise ValueError("Complex data not supported")
        if isinstance(accept_sparse, str):
            a = a.asformat(accept_sparse)
        elif isinstance(accept_sparse, (tuple, list)) and a.format not in accept_sparse:
            a = a.asformat(accept_sparse[0])
        raw_dtype = a.dtype
    else:
        a = np.asarray(array)
        raw_dtype = a.dtype
        if np.issubdtype(raw_dtype, np.complexfloating):
            raise ValueError("Complex data not supported")
    if isinstance(dtype, (tuple, list)):
        chosen = raw_dtype if raw_dtype in [np.dtype(t) for t in dtype] else dtype[0]
    elif dtype is None:
        chosen = raw_dtype
    elif dtype == "numeric":
        chosen = np.float64 if raw_dtype.kind not in "fiu" else raw_dtype
    else:
        chosen = dtype
    if is_sparse:
        a = a.astype(chosen, copy=copy)
        if force_writeable and not a.data.flags.writeable:
            a = a.copy()
        values = a.data
    else:
        a = np.asarray(a, dtype=chosen, order=order)
        if copy or (force_writeable and not a.flags.writeable):
            a = a.copy(order=order or "K")
        values = a
    if a.ndim == 0:
        raise ValueError("Expected 2D array, got scalar array instead. Reshape your data.")
    if ensure_2d and a.ndim == 1:
        raise ValueError("Expected 2D array, got 1D array instead. Reshape your data.")
    if not allow_nd and a.ndim > 2:
        raise ValueError(f"Found array with dim {a.ndim}, while dim <= 2 is required")
    if a.shape[0] < ensure_min_samples:
        raise ValueError(f"Found array with {a.shape[0]} sample(s) while a minimum of {ensure_min_samples} is required")
    if ensure_2d and a.shape[1] < ensure_min_features:
        raise ValueError(f"0 feature(s) (shape={a.shape}) while a minimum of {ensure_min_features} is required.")
    if ensure_all_finite and not np.all(np.isfinite(values)):
        raise ValueError("Input contains NaN, infinity or a value too large for its dtype")
    return a


def _feature_names(X):
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = np.asarray(columns, dtype=object)
    if names.ndim == 1 and all(isinstance(s, str) for s in names):
        return names
    if names.ndim == 1 and any(isinstance(s, str) for s in names):
        raise TypeError("Feature names must all be strings or all be non-string values")
    return None


def validate_data(estimator, X, *, reset=True, **kwargs):
    """Shape and feature-name contract for fit and prediction."""
    names = _feature_names(X)
    array = check_array(X, **kwargs)
    if reset:
        estimator.n_features_in_ = int(array.shape[1])
        if names is None:
            estimator.__dict__.pop("feature_names_in_", None)
        else:
            estimator.feature_names_in_ = names.copy()
    else:
        check_is_fitted(estimator, "n_features_in_")
        fitted_names = getattr(estimator, "feature_names_in_", None)
        if names is not None and fitted_names is not None and not np.array_equal(names, fitted_names):
            raise ValueError("The feature names should match those that were passed during fit, in the same order")
        if names is None and fitted_names is not None:
            warnings.warn(f"X does not have valid feature names, but {type(estimator).__name__} was fitted with feature names", UserWarning, stacklevel=2)
        elif names is not None and fitted_names is None:
            warnings.warn(f"X has feature names, but {type(estimator).__name__} was fitted without feature names", UserWarning, stacklevel=2)
        if array.shape[1] != estimator.n_features_in_:
            raise ValueError(f"X has {array.shape[1]} features, but {type(estimator).__name__} is expecting {estimator.n_features_in_} features as input.")
    return array


def check_is_fitted(estimator, attributes=None):
    if attributes is None:
        fitted = any(key.endswith("_") and not key.startswith("__") for key in vars(estimator))
    else:
        if isinstance(attributes, str):
            attributes = [attributes]
        fitted = all(hasattr(estimator, key) for key in attributes)
    if not fitted:
        exception = getattr(sys.modules.get("sklearn.exceptions"), "NotFittedError", NotFittedError)
        raise exception(f"This {type(estimator).__name__} instance is not fitted yet. Call fit before using this estimator.")


class BaseEstimator:
    """Small independent implementation of the public estimator protocol."""

    @classmethod
    def _get_param_names(cls):
        signature = inspect.signature(cls.__init__)
        return sorted(name for name, parameter in signature.parameters.items()
                      if name != "self" and parameter.kind not in
                      (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL))

    def get_params(self, deep=True):
        result = {name: getattr(self, name) for name in self._get_param_names()}
        if deep:
            for name, value in list(result.items()):
                if hasattr(value, "get_params"):
                    result.update({f"{name}__{key}": child for key, child in value.get_params().items()})
        return result

    def set_params(self, **params):
        valid = self.get_params(deep=True)
        grouped = {}
        for name, value in params.items():
            parent, separator, child = name.partition("__")
            if parent not in valid:
                raise ValueError(f"Invalid parameter {parent!r} for {type(self).__name__}")
            if separator:
                grouped.setdefault(parent, {})[child] = value
            else:
                setattr(self, parent, value)
        for name, children in grouped.items():
            getattr(self, name).set_params(**children)
        return self

    def __sklearn_clone__(self):
        result = type(self)(**copy.deepcopy(self.get_params(deep=False)))
        for attribute in ("_output_transform", "_metadata_requests"):
            if hasattr(self, attribute):
                setattr(result, attribute, copy.deepcopy(getattr(self, attribute)))
        return result

    def __repr__(self):
        parameters = ", ".join(f"{name}={value!r}" for name, value in self.get_params(deep=False).items())
        return f"{type(self).__name__}({parameters})"

    def __sklearn_tags__(self):
        # Tags are schema objects for the caller; they perform no data analysis.
        try:
            from sklearn.utils import Tags, TargetTags, TransformerTags, InputTags
        except ImportError:
            return SimpleNamespace(estimator_type=getattr(self, "_estimator_type", None),
                                   input_tags=SimpleNamespace(sparse=False, allow_nan=False),
                                   target_tags=SimpleNamespace(required=False), requires_fit=True)
        return Tags(estimator_type=getattr(self, "_estimator_type", None),
                    target_tags=TargetTags(required=False),
                    transformer_tags=TransformerTags(preserves_dtype=["float64"])
                    if hasattr(self, "transform") else None,
                    input_tags=InputTags())

    def set_output(self, *, transform=None):
        if transform is not None:
            if transform not in ("default", "pandas", "polars"):
                raise ValueError("transform output must be 'default', 'pandas', or 'polars'")
            self._output_transform = transform
        return self

    def _wrap_output(self, result, X):
        setting = getattr(self, "_output_transform", "default")
        if setting == "default":
            return result
        columns = self.get_feature_names_out()
        if setting == "pandas":
            import pandas as pd
            index = getattr(X, "index", None)
            return pd.DataFrame(result, columns=columns, index=None if callable(index) else index)
        import polars as pl
        return pl.DataFrame(result, schema=list(columns), orient="row")

    def _set_request(self, method, **requests):
        if set(requests).difference({"sample_weight"}):
            raise TypeError("Only sample_weight metadata is accepted")
        for value in requests.values():
            if value is not None and not isinstance(value, (str, bool)):
                raise ValueError("Metadata requests must be bool, None, or an alias string")
        state = dict(getattr(self, "_metadata_requests", {}))
        state[method] = dict(requests)
        self._metadata_requests = state
        return self

    def set_fit_request(self, **requests):
        return self._set_request("fit", **requests)

    def set_score_request(self, **requests):
        return self._set_request("score", **requests)

    def get_metadata_routing(self):
        # Optional protocol integration. Core fitting works with sklearn absent.
        from sklearn.utils.metadata_routing import MetadataRequest
        result = MetadataRequest(owner=type(self).__name__)
        for method in ("fit", "score"):
            if "sample_weight" in inspect.signature(getattr(self, method, lambda: None)).parameters:
                mapping = getattr(self, "_metadata_requests", {}).get(method, {"sample_weight": None})
                for parameter, value in mapping.items():
                    getattr(result, method).add_request(param=parameter, alias=value)
        return result


class ClusterMixin:
    _estimator_type = "clusterer"


class TransformerMixin:
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class ClassNamePrefixFeaturesOutMixin:
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "_n_features_out")
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
            if names.shape != (self.n_features_in_,):
                raise ValueError("input_features must contain one name per feature")
            if hasattr(self, "feature_names_in_") and not np.array_equal(names, self.feature_names_in_):
                raise ValueError("input_features must match feature_names_in_")
        return np.asarray([f"{type(self).__name__.lower()}{j}" for j in range(self._n_features_out)], dtype=object)
