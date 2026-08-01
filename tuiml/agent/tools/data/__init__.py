"""The tools that get data in and ready.

Everything an agent needs before it can train: bringing a file into TuiML,
looking at it, reshaping it, or synthesising one when there is no real data
to hand.

Tools
-----
- **tuiml_upload_data:** Register a dataset from a file path or from inline
  text, giving it a name later tools can refer to.
- **tuiml_read_data:** Show rows (head, tail, sample, or specific indices)
  so the agent can see what it is working with.
- **tuiml_profile_data:** Shape, dtypes, missing values, class balance and
  summary statistics.
- **tuiml_generate_data:** Synthesise a dataset from a built-in generator
  (Blobs, Agrawal, Friedman, ...).
- **tuiml_preprocess:** Apply preprocessing steps and save the result.
- **tuiml_select_features:** Reduce to the most informative columns.

Notes
-----
Anywhere these take a ``data`` argument, it accepts a built-in dataset name
(``"iris"``), a path to a file, or the name of a previous upload.
"""
