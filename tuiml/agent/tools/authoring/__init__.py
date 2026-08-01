"""The tools that let an agent write its own algorithms.

An agent that can only call built-in algorithms is limited to what shipped.
These tools let it read the source of existing ones, then write, register and
iterate on a new algorithm of its own — which then behaves like any other
TuiML component and can be trained and benchmarked by name.

Tools
-----
- **tuiml_get_skeleton:** A correct starting template for a classifier or
  regressor, with the decorator and required methods in place.
- **tuiml_create_algorithm:** Validate and register new source, versioned.
- **tuiml_read_algorithm:** Read a user-authored or built-in algorithm.
- **tuiml_edit_algorithm:** Exact-string edit, optionally bumping the version.
- **tuiml_delete_algorithm:** Remove an authored algorithm, or one version.
- **tuiml_list_files:** List the algorithm source files available to read.
- **tuiml_search_source:** Regex search across that source.

Notes
-----
Authored algorithms live under ``~/.tuiml/user_algorithms/<name>/<version>/``
and are re-registered from disk at import, so they survive a server restart.
Source is AST-validated before it is accepted: it must define exactly one
``@classifier`` or ``@regressor`` class.
"""
