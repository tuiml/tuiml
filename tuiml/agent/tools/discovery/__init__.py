"""The tools an agent uses to find out what TuiML has.

An agent cannot train ``RandomForestClassifier`` unless it knows the name
exists and what it accepts. These two tools answer that, so nothing has to be
hardcoded into a prompt.

Tools
-----
- **tuiml_list:** Enumerate components by category (algorithms, datasets,
  preprocessors, metrics, ...), with keyword search, limit and offset.
- **tuiml_describe:** Full detail for one component: what it does, its
  parameters with types and defaults, and its capabilities.

Notes
-----
Neither tool is reproducible, so their calls never become notebook cells:
looking something up is not part of the analysis being exported.
"""
