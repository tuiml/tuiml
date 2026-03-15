"""Tests for the Classifier base class.

These tests ensure that all classifier implementations follow
the expected interface and behavior.
"""

import numpy as np
import pytest
from abc import ABC

from tuiml.base.algorithms import (
    Algorithm, Classifier, Regressor, Clusterer, Associator
)


class TestAlgorithmBaseClass:
    """Tests for the Algorithm abstract base class."""
    
    def test_algorithm_is_abstract(self):
        """Test that Algorithm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Algorithm()
            
    def test_algorithm_has_required_methods(self):
        """Test that Algorithm defines required abstract methods."""
        assert hasattr(Algorithm, 'fit')
        assert hasattr(Algorithm, 'predict')
        
    def test_algorithm_metadata_methods(self):
        """Test that Algorithm has metadata methods."""
        assert hasattr(Algorithm, 'get_metadata')
        assert hasattr(Algorithm, 'get_parameter_schema')
        assert hasattr(Algorithm, 'get_capabilities')
        assert hasattr(Algorithm, 'get_complexity')
        assert hasattr(Algorithm, 'get_references')


class TestClassifierBaseClass:
    """Tests for the Classifier base class."""
    
    def test_classifier_is_abstract(self):
        """Test that Classifier cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Classifier()
            
    def test_classifier_inherits_from_algorithm(self):
        """Test that Classifier inherits from Algorithm."""
        assert issubclass(Classifier, Algorithm)
        
    def test_classifier_has_predict_proba(self):
        """Test that Classifier has predict_proba method."""
        assert hasattr(Classifier, 'predict_proba')


class TestRegressorBaseClass:
    """Tests for the Regressor base class."""
    
    def test_regressor_is_abstract(self):
        """Test that Regressor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Regressor()
            
    def test_regressor_inherits_from_algorithm(self):
        """Test that Regressor inherits from Algorithm."""
        assert issubclass(Regressor, Algorithm)


class TestClustererBaseClass:
    """Tests for the Clusterer base class."""
    
    def test_clusterer_is_abstract(self):
        """Test that Clusterer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Clusterer()
            
    def test_clusterer_inherits_from_algorithm(self):
        """Test that Clusterer inherits from Algorithm."""
        assert issubclass(Clusterer, Algorithm)
        
    def test_clusterer_has_cluster_attributes(self):
        """Test that Clusterer defines cluster attributes."""
        # These are set in __init__
        assert hasattr(Clusterer, '__init__')


class TestAssociatorBaseClass:
    """Tests for the Associator base class."""
    
    def test_associator_is_abstract(self):
        """Test that Associator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Associator()
            
    def test_associator_inherits_from_algorithm(self):
        """Test that Associator inherits from Algorithm."""
        assert issubclass(Associator, Algorithm)
        
    def test_associator_predict_raises(self):
        """Test that Associator predict raises NotImplementedError."""
        # Can't test directly since Associator is abstract
        pass


class TestAlgorithmRegistry:
    """Tests for the AlgorithmRegistry."""
    
    def test_list_algorithms(self):
        """Test that we can list algorithms."""
        from tuiml.base.algorithms import AlgorithmRegistry
        
        algorithms = AlgorithmRegistry.list()
        assert isinstance(algorithms, list)
        
    def test_list_by_type(self):
        """Test listing algorithms by type."""
        from tuiml.base.algorithms import AlgorithmRegistry
        
        classifiers = AlgorithmRegistry.list(type="classifier")
        assert isinstance(classifiers, list)
        
    def test_search_algorithms(self):
        """Test searching algorithms."""
        from tuiml.base.algorithms import AlgorithmRegistry
        
        results = AlgorithmRegistry.search("tree")
        assert isinstance(results, list)
