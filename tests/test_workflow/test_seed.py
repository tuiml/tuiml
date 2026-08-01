import random
import numpy as np
import pandas as pd
from tuiml.utils.seed import set_global_seed, get_global_seed
from tuiml.training import train
from tuiml.agent.tools import execute_tool


def test_global_seed_utility():
    set_global_seed(42)
    assert get_global_seed() == 42

    val1 = random.random()
    np_val1 = np.random.rand()

    set_global_seed(42)
    val2 = random.random()
    np_val2 = np.random.rand()

    assert val1 == val2
    assert np_val1 == np_val2

    # Clean up
    set_global_seed(None)
    assert get_global_seed() is None


def test_api_train_seed_determinism():
    # Create a simple synthetic classification dataset
    from tuiml.datasets.generators import Blobs
    data = Blobs(n_samples=50, n_features=3, n_clusters=2, random_state=42).generate()
    df = pd.DataFrame(data.X, columns=['x1', 'x2', 'x3'])
    df['target'] = data.y

    # Run train with explicit random_seed
    res1 = train({"model": {"name": "RandomForestClassifier"}, "data": df,
                  "target": "target", "random_seed": 42})
    res2 = train({"model": {"name": "RandomForestClassifier"}, "data": df,
                  "target": "target", "random_seed": 42})
    assert res1.metrics_ == res2.metrics_

    # Set global seed and run train with no explicit seed (should fallback to global seed)
    set_global_seed(123)
    res3 = train({"model": {"name": "RandomForestClassifier"}, "data": df, "target": "target"})

    set_global_seed(123)
    res4 = train({"model": {"name": "RandomForestClassifier"}, "data": df, "target": "target"})
    assert res3.metrics_ == res4.metrics_

    set_global_seed(None)


def test_mcp_tool_execute_seed():
    # Create simple dataframe and save it temporarily
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        from tuiml.datasets.generators import Blobs
        data = Blobs(n_samples=50, n_features=3, n_clusters=2, random_state=42).generate()
        df = pd.DataFrame(data.X, columns=['x1', 'x2', 'x3'])
        df['target'] = data.y
        df.to_csv(csv_path, index=False)

        # Test tuiml_train via MCP
        res1 = execute_tool("tuiml_train", algorithm="RandomForestClassifier", data=csv_path, target="target", random_seed=777)
        assert res1['status'] == 'success'
        assert res1['random_seed'] == 777

        res2 = execute_tool("tuiml_train", algorithm="RandomForestClassifier", data=csv_path, target="target", random_seed=777)
        assert res2['status'] == 'success'
        assert res2['random_seed'] == 777

        # Verify determinism
        assert res1['metrics'] == res2['metrics']

        # Test tuiml_benchmark via MCP
        exp_res1 = execute_tool(
            "tuiml_benchmark",
            algorithms=["RandomForestClassifier"],
            data=csv_path,
            target="target",
            random_seed=888
        )
        assert exp_res1['status'] == 'success'
        assert exp_res1['random_seed'] == 888

        # Clean up global seed
        set_global_seed(None)
