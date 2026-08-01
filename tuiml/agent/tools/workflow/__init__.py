"""The tools that actually run machine learning.

One module per tool, each declaring its ``ToolSpec`` next to its executor.
These are the calls an agent chains together to get from a dataset to a
served model, and the only group whose results become notebook cells.

Tools
-----
- **tuiml_train:** Fit a model, with optional preprocessing, feature
  selection and cross-validation. Returns a ``model_id``.
- **tuiml_predict:** Predict with a previously trained model.
- **tuiml_evaluate:** Score a trained model on a dataset.
- **tuiml_benchmark:** Compare several algorithms over the same data.
- **tuiml_tune:** Grid or random hyperparameter search.
- **tuiml_plot:** Confusion matrix, ROC, PR, learning curve, CD diagram.
- **tuiml_save_model:** Write a trained model to a path of your choosing.

Notes
-----
Most of these take a ``model_id`` produced by ``tuiml_train``. Ids resolve
through an index rehydrated from ``~/.tuiml/models/`` at import, so one
survives an MCP server restart.
"""
