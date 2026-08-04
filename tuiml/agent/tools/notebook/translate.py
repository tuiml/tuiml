"""Translate recorded MCP calls into notebook cells."""

import os
from typing import Dict, List, Optional

from .._state import _MODEL_ID_TO_VAR


def _call_to_kwargs_str(args: dict, skip: set = None, indent: int = 4) -> str:
    """Format a dict of args as indented keyword arguments.

    Parameters
    ----------
    args : dict
        Argument name -> value mapping to render.
    skip : set, default=None
        Keys to omit (None-valued entries are always omitted).
    indent : int, default=4
        Number of spaces to indent each line.

    Returns
    -------
    text : str
        Newline-joined ``name=value,`` lines.
    """
    pad = ' ' * indent
    skip = skip or set()
    lines = []
    for k, v in args.items():
        if k in skip or v is None:
            continue
        lines.append(f"{pad}{k}={v!r},")
    return '\n'.join(lines)


def _data_load_lines(data: str) -> List[str]:
    """Return notebook code lines that load `data` into variable `_dataset`.

    Parameters
    ----------
    data : str
        File path (loaded via pandas) or built-in dataset name (loaded
        via ``load_dataset``).

    Returns
    -------
    lines : list of str
        Source lines for a notebook code cell.
    """
    if data and os.path.isfile(os.path.expanduser(data)):
        return [
            f"import pandas as _pd\n",
            f"_df = _pd.read_csv({repr(data)})\n",
            f"_X = _df.iloc[:, :-1].values\n",
            f"_y = _df.iloc[:, -1].values",
        ]
    # assume builtin dataset name
    return [f"_dataset = load_dataset({repr(data)})"]


def _algorithm_source_cell(source: str, header: List[str]) -> List[str]:
    """Return code-cell lines: a comment header followed by algorithm source.

    Parameters
    ----------
    source : str
        Full Python source of a user-authored algorithm.
    header : list of str
        Comment lines (each newline-terminated) to place above the source.

    Returns
    -------
    lines : list of str
        Source lines for a notebook code cell, with no trailing blank line.
    """
    body = source.splitlines(keepends=True)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return list(header) + ["# (source unavailable)"]
    if body[-1].endswith('\n'):
        body[-1] = body[-1][:-1]
    return list(header) + body


def _read_user_algorithm_source(name: str, version: Optional[str] = None) -> tuple:
    """Read a user algorithm's current on-disk source.

    The import is local: notebook translation must not drag the
    user-algorithm storage layer into every import of this module.

    Parameters
    ----------
    name : str
        User algorithm name (directory / class name).
    version : str, default=None
        Pin a specific version; None reads the newest on disk.

    Returns
    -------
    source : str or None
        The full source, or None when it could not be read.
    error : str or None
        Reason the read failed, or None on success.
    """
    try:
        from tuiml.agent.user_algorithms import read_source
        res = read_source(name, version=version)
    except Exception as e:
        return None, str(e)
    if not isinstance(res, dict) or res.get('status') != 'success':
        return None, (res or {}).get('error', 'unknown error')
    return res.get('source'), None


def _already_emitted(emitted: Optional[Dict], name: str, source: str) -> bool:
    """Whether this exact source was already emitted for `name`.

    A session that edits an algorithm several times records one call per
    edit, but every edit cell re-reads the same final on-disk source;
    emitting them all would put N identical class definitions in the
    notebook.

    Parameters
    ----------
    emitted : dict or None
        Mutable ``name -> last emitted source`` map. None disables dedup.
    name : str
        Algorithm name the source belongs to.
    source : str
        Source about to be emitted.

    Returns
    -------
    duplicate : bool
        True when the caller should skip this cell.
    """
    if emitted is None:
        return False
    if emitted.get(name) == source:
        return True
    emitted[name] = source
    return False


def _resolve_model_var(model_id: Optional[str], fallback: str = "model_1") -> str:
    """Map a model_id back to the Python variable name used in the notebook.

    Parameters
    ----------
    model_id : str or None
        Model identifier recorded during the session.
    fallback : str, default='model_1'
        Variable name returned when the id is unknown.

    Returns
    -------
    var : str
        Notebook variable name (e.g. ``'result_2'``).
    """
    if model_id and model_id in _MODEL_ID_TO_VAR:
        return _MODEL_ID_TO_VAR[model_id]
    return fallback


def _translate_call(call: Dict, train_counter: List[int],
                    emitted_sources: Optional[Dict] = None) -> tuple:
    """Translate one recorded session call into notebook cells.

    Parameters
    ----------
    call : dict
        Recorded call with ``'tool'`` and ``'args'`` keys.
    train_counter : list of int
        Single-element mutable counter of train calls seen so far, used
        to number ``result_N`` / ``model_N`` variables.
    emitted_sources : dict, default=None
        Mutable ``algorithm name -> last emitted source`` map used to skip
        repeated identical definitions of a user-authored algorithm. None
        disables that dedup.

    Returns
    -------
    md_lines : list of str or None
        Markdown cell source, or None when the tool is not translatable.
    code_lines : list of str or None
        Code cell source, or None when the tool is not translatable.
    """
    tool = call['tool']
    args = call['args']

    # ── tuiml_profile_data ───────────────────────────────────────────────────
    if tool == 'tuiml_profile_data':
        data = args.get('data', '')
        target = args.get('target')
        md = [
            f"## Data Profiling, `{data}`\n",
            f"> `tuiml_profile_data(data={repr(data)}`",
        ]
        code = _data_load_lines(data) + [
            "\n",
            "import pandas as pd\n",
            "_df_profile = pd.DataFrame(_dataset.X, columns=_dataset.feature_names)\n",
        ]
        if target:
            code.append(f"_df_profile[{repr(target)}] = _dataset.y\n")
        code += [
            "print(f'Shape: {_df_profile.shape}')\n",
            "print(f'Missing values: {_df_profile.isnull().sum().sum()}')\n",
        ]
        if target:
            code.append(f"print('Class distribution:\\n', _df_profile[{repr(target)}].value_counts())\n")
        code.append("_df_profile.describe()")
        return md, code

    # ── tuiml_create_algorithm ───────────────────────────────────────────────
    if tool == 'tuiml_create_algorithm':
        name = args.get('name', 'UserAlgorithm')
        kind = args.get('kind', '')
        version = args.get('version', '1.0.0')
        description = args.get('description')
        source = args.get('code', '')
        if _already_emitted(emitted_sources, name, source):
            return None, None
        md = [
            f"## Define Algorithm `{name}` (v{version})\n",
            f"> `tuiml_create_algorithm(name={name!r}, kind={kind!r}, version={version!r})`\n",
        ]
        if description:
            md.append(f"\n{description}\n")
        # A user-authored algorithm lives in ~/.tuiml/user_algorithms/, not in
        # the installed package, so `pip install tuiml` alone would leave the
        # later tuiml.train(...) cells unable to resolve the name. Inlining the
        # source keeps the notebook self-contained.
        md.append(
            "\nThis algorithm was authored during the session, so it is not part of the "
            "installed package. Its source is inlined here: running the cell fires the "
            "`@classifier`/`@regressor` decorator, which registers the class under "
            f"`{name}` for the training cells below."
        )
        code = _algorithm_source_cell(source, [
            f"# User-authored algorithm `{name}` v{version}, registered on execution.\n",
        ])
        return md, code

    # ── tuiml_edit_algorithm ─────────────────────────────────────────────────
    if tool == 'tuiml_edit_algorithm':
        name = args.get('name', '')
        # The recorded args hold only the old->new fragment, which is not
        # runnable on its own, so re-read the full post-edit source from disk.
        # After a version bump the edit landed in a *new* version, so read the
        # latest rather than the version the edit targeted.
        version = None if args.get('bump_version') else args.get('version')
        source, read_err = _read_user_algorithm_source(name, version)
        if source is not None and _already_emitted(emitted_sources, name, source):
            return None, None
        md = [
            f"## Redefine Algorithm `{name}` (edited)\n",
            f"> `tuiml_edit_algorithm(name={name!r}, ...)`\n",
            "\nThe session edited this algorithm. The full post-edit source is inlined "
            "below so the notebook reproduces the edited version; re-running it "
            "re-registers the class over any earlier definition.",
        ]
        if source is None:
            code = [
                f"# Could not recover the post-edit source of `{name}`: {read_err}\n",
                "# It was edited during the session but is no longer readable on disk,\n",
                "# so this step cannot be reproduced automatically.\n",
                f"# The edit replaced {args.get('old_string', '')!r}\n",
                f"#            with   {args.get('new_string', '')!r}",
            ]
            return md, code
        code = _algorithm_source_cell(source, [
            f"# User-authored algorithm `{name}`, source after the session's edit.\n",
        ])
        return md, code

    # ── tuiml_train ──────────────────────────────────────────────────────────
    if tool == 'tuiml_train':
        train_counter[0] += 1
        n = train_counter[0]
        algo = args.get('algorithm', 'UnknownAlgorithm')
        data = args.get('data', '')
        target = args.get('target')
        # tuiml.train() uses the spec convention: the model is
        # {"name": algo, "params": {...}}, the data is {"source", "target"},
        # steps live in one ordered "pipeline" list, and evaluation options
        # are grouped into an "evaluation" dict. Translate the tool-level
        # vocabulary into that shape so the generated call is valid,
        # idiomatic Python.
        model_spec = {"name": algo}
        algo_params = args.get('algorithm_params')
        if isinstance(algo_params, dict) and algo_params:
            model_spec["params"] = algo_params
        data_spec = {"source": data}
        if target is not None:
            data_spec["target"] = target
        if args.get('features') is not None:
            data_spec["features"] = args['features']

        def _nest_step(step):
            """Convert a flat tool-level step ({"name", **params}) to spec form."""
            if isinstance(step, str):
                return {"name": step}
            step = dict(step)
            name = step.pop('name', None)
            params = step.pop('params', step)
            nested = {"name": name}
            if params:
                nested["params"] = params
            return nested

        steps = [_nest_step(s) for s in args.get('preprocessing') or []]
        if args.get('feature_selection'):
            steps.append(_nest_step(args['feature_selection']))
        pipeline = steps or args.get('preset')
        evaluation = {
            k: args[k]
            for k in ('cv', 'test_size', 'stratify', 'metrics')
            if args.get(k) is not None
        }
        lines = [
            f"result_{n} = tuiml.train({{\n",
            f"    \"model\": {model_spec!r},\n",
            f"    \"data\": {data_spec!r},\n",
        ]
        if pipeline:
            lines.append(f"    \"pipeline\": {pipeline!r},\n")
        if evaluation:
            lines.append(f"    \"evaluation\": {evaluation!r},\n")
        if args.get('random_seed') is not None:
            lines.append(f"    \"random_seed\": {args['random_seed']!r},\n")
        lines.append("})\n")
        md = [
            f"## Train `{algo}` (step {n})\n",
            f"> `tuiml_train(algorithm={repr(algo)}, data={repr(data)}, ...)`",
        ]
        code = lines + [
            f"model_{n} = result_{n}.model_\n",
            f"print('Metrics:', result_{n}.metrics_)",
        ]
        return md, code

    # ── tuiml_predict ────────────────────────────────────────────────────────
    if tool == 'tuiml_predict':
        model_id = args.get('model_id')
        var = _resolve_model_var(model_id)
        result_var = var  # e.g. result_1
        model_var = var.replace('result_', 'model_')
        data = args.get('data', '')
        md = [
            f"## Predict with `{result_var}`\n",
            f"> `tuiml_predict(model_id=..., data={repr(data)})`",
        ]
        code = _data_load_lines(data) + [
            "\n",
            # Predict via the fitted Workflow so the training-time
            # transformations are re-applied (a bare model would see raw,
            # untransformed inputs and produce wrong predictions).
            f"predictions = {result_var}.predict(_dataset.X)\n",
            "print('Predictions (first 10):', predictions[:10])",
        ]
        return md, code

    # ── tuiml_evaluate ───────────────────────────────────────────────────────
    if tool == 'tuiml_evaluate':
        model_id = args.get('model_id')
        var = _resolve_model_var(model_id)
        model_var = var.replace('result_', 'model_')
        data = args.get('data', '')
        target = args.get('target')
        metrics = args.get('metrics')
        md = [
            f"## Evaluate `{model_var}`\n",
            f"> `tuiml_evaluate(model_id=..., data={repr(data)})`",
        ]
        code = _data_load_lines(data) + [
            "\n",
            # Evaluate via the fitted Workflow: it re-applies the fitted
            # transformations before scoring, so the metrics reflect the real
            # pipeline.
            f"eval_metrics = {var}.evaluate(\n",
            f"    _dataset.X, _dataset.y,\n",
        ]
        if metrics:
            code.append(f"    metrics={repr(metrics)},\n")
        code += [")\n", "print('Eval metrics:', eval_metrics)"]
        return md, code

    # ── tuiml_benchmark ─────────────────────────────────────────────────────
    if tool == 'tuiml_benchmark':
        algos = args.get('algorithms', [])
        data_arg = args.get('data', [])
        if isinstance(data_arg, str):
            data_arg = [data_arg]
        cv = args.get('cv', 10)
        metrics = args.get('metrics')
        md = [
            f"## Benchmark, {', '.join(str(a) for a in algos)}\n",
            f"> `tuiml_benchmark(algorithms={algos}, data={data_arg}, cv={cv})`",
        ]
        code = []
        for ds_name in data_arg:
            safe = ds_name.replace('-', '_').replace('/', '_')
            code.append(f"_{safe} = load_dataset({repr(ds_name)})\n")
        model_specs = [
            a if isinstance(a, dict) else {"name": a} for a in algos
        ]
        dataset_specs = "[" + ", ".join(
            f'{{"name": {repr(d)}, '
            f'"X": _{d.replace("-","_").replace("/","_")}.X, '
            f'"y": _{d.replace("-","_").replace("/","_")}.y}}'
            for d in data_arg
        ) + "]"
        evaluation = {"cv": cv}
        if metrics:
            evaluation["metrics"] = list(metrics)
        code += [
            "\n",
            "bench = tuiml.Benchmark(\n",
            f"    models={model_specs!r},\n",
            f"    datasets={dataset_specs},\n",
            f"    evaluation={evaluation!r},\n",
        ]
        seed = args.get('random_seed')
        if seed is not None:
            code.append(f"    random_seed={seed!r},\n")
        code += [
            ").run()\n",
            "print(bench.summary())\n",
            "bench.table()",
        ]
        return md, code

    # ── tuiml_tune ───────────────────────────────────────────────────────────
    if tool == 'tuiml_tune':
        algo = args.get('algorithm', '')
        data = args.get('data', '')
        method = args.get('method', 'random')
        param_grid = args.get('param_grid', {})
        cv = args.get('cv', 5)
        scoring = args.get('scoring', 'accuracy_score')
        n_iter = args.get('n_iter', 10)
        n_iterations = args.get('n_iterations', 50)
        # Prefer an explicit random_state, then the effective session seed folded
        # in by record_session_call, then the default, so tuning reproduces.
        random_state = args.get('random_state', args.get('random_seed', 42))
        cls_map = {'grid': 'GridSearchCV', 'random': 'RandomSearchCV', 'bayesian': 'BayesianSearchCV'}
        tuner_cls = cls_map.get(method, 'RandomSearchCV')
        param_kw = 'param_grid' if method == 'grid' else ('param_space' if method == 'bayesian' else 'param_distributions')
        n_kw_line = (f"    n_iter={n_iter},\n" if method == 'random'
                     else f"    n_iterations={n_iterations},\n" if method == 'bayesian' else "")
        md = [
            f"## Hyperparameter Tuning, `{algo}` ({method} search)\n",
            f"> `tuiml_tune(algorithm={repr(algo)}, method={repr(method)}, ...)`",
        ]
        code = [
            "from tuiml.registry import registry as _registry\n",
            "import tuiml.algorithms as _\n",
            f"from tuiml.evaluation.tuning import {tuner_cls}\n",
            "\n",
            *_data_load_lines(data), "\n",
            f"_cls = _registry.get({repr(algo)})\n",
            f"tuner = {tuner_cls}(\n",
            f"    estimator=_cls(),\n",
            f"    {param_kw}={repr(param_grid)},\n",
            f"    cv={cv},\n",
            f"    scoring={repr(scoring)},\n",
            n_kw_line,
            f"    random_state={random_state},\n",
            ")\n",
            "tuner.fit(_dataset.X, _dataset.y)\n",
            "print('Best params:', tuner.best_params_)\n",
            "print('Best score: ', tuner.best_score_)",
        ]
        return md, code

    # ── tuiml_plot ───────────────────────────────────────────────────────────
    if tool == 'tuiml_plot':
        plot_type = args.get('plot_type', '')
        model_id = args.get('model_id')
        var = _resolve_model_var(model_id)
        model_var = var.replace('result_', 'model_')
        data = args.get('data', '')
        target = args.get('target', '')
        algo = args.get('algorithm', '')
        title = args.get('title') or f"{plot_type.replace('_', ' ').title()}"

        md = [
            f"## Plot, `{plot_type}`\n",
            f"> `tuiml_plot(plot_type={repr(plot_type)}, ...)`",
        ]

        if plot_type == 'confusion_matrix':
            code = (
                _data_load_lines(data) + ["\n",
                "from tuiml.evaluation.visualization import plot_confusion_matrix\n",
                f"_preds = {var}.predict(_dataset.X)\n",
                f"plot_confusion_matrix(_dataset.y, _preds, title={repr(title)})\n",
                "plt.show()",
            ])
        elif plot_type == 'roc_curve':
            code = (
                _data_load_lines(data) + ["\n",
                "from tuiml.evaluation.visualization import plot_roc_curve\n",
                f"_probas = {var}.predict_proba(_dataset.X)\n",
                f"plot_roc_curve(_dataset.y, _probas, title={repr(title)})\n",
                "plt.show()",
            ])
        elif plot_type == 'pr_curve':
            code = (
                _data_load_lines(data) + ["\n",
                "from tuiml.evaluation.visualization import plot_pr_curve\n",
                f"_probas = {var}.predict_proba(_dataset.X)\n",
                "# PR curve takes positive-class scores; pick column 1 if 2-D.\n",
                "_score = _probas[:, 1] if _probas.ndim == 2 else _probas\n",
                f"plot_pr_curve(_dataset.y, _score, title={repr(title)})\n",
                "plt.show()",
            ])
        elif plot_type == 'feature_importance':
            code = [
                f"_importances = getattr({model_var}, 'feature_importances_', None)\n",
                "if _importances is None:\n",
                "    raise ValueError('This model does not expose feature importances.')\n",
                "plt.figure(figsize=(10, 5))\n",
                "plt.bar(range(len(_importances)), _importances)\n",
                f"plt.title({repr(title)})\n",
                "plt.xlabel('Feature Index'); plt.ylabel('Importance')\n",
                "plt.tight_layout(); plt.show()",
            ]
        elif plot_type == 'learning_curve':
            # plot_learning_curve takes precomputed (train_sizes, train_scores,
            # test_scores), so compute them here over increasing train subsets.
            code = (
                _data_load_lines(data) + ["\n",
                "import numpy as np\n",
                "from tuiml.evaluation.visualization import plot_learning_curve\n",
                "from tuiml.registry import registry\n",
                "import tuiml.algorithms  # noqa: F401, registers algorithms\n",
                "from tuiml.evaluation.splitting import train_test_split\n",
                "from tuiml.evaluation.metrics import accuracy_score\n",
                f"_cls = registry.get({repr(algo)})\n",
                "_Xtr, _Xte, _ytr, _yte = train_test_split(\n",
                "    _dataset.X, _dataset.y, test_size=0.25, random_state=42)\n",
                "_sizes, _train_sc, _test_sc = [], [], []\n",
                "for _frac in np.linspace(0.2, 1.0, 5):\n",
                "    _n = max(2, int(len(_Xtr) * _frac))\n",
                "    _m = _cls(); _m.fit(_Xtr[:_n], _ytr[:_n])\n",
                "    _sizes.append(_n)\n",
                "    _train_sc.append(accuracy_score(_ytr[:_n], _m.predict(_Xtr[:_n])))\n",
                "    _test_sc.append(accuracy_score(_yte, _m.predict(_Xte)))\n",
                "plot_learning_curve(np.array(_sizes), np.array(_train_sc),\n",
                f"                    np.array(_test_sc), title={repr(title)})\n",
                "plt.show()",
            ])
        elif plot_type in ('cd_diagram', 'boxplot_comparison', 'heatmap', 'ranking_table'):
            exp_results = args.get('benchmark_results', {})
            # cd_diagram maps to plot_critical_difference; all take the scores
            # dict as the first positional argument (not 'benchmark_results=').
            fn = 'plot_critical_difference' if plot_type == 'cd_diagram' else f'plot_{plot_type}'
            code = [
                "import numpy as np\n",
                f"from tuiml.evaluation.visualization import {fn}\n",
                f"_exp = {{k: np.array(v, dtype=float) for k, v in {repr(exp_results)}.items()}}\n",
                f"{fn}(_exp)\n",
                "plt.show()",
            ]
        else:
            return None, None

        return md, code

    # ── tuiml_save_model ─────────────────────────────────────────────────────
    if tool == 'tuiml_save_model':
        model_id = args.get('model_id')
        dest = args.get('destination', './model.joblib')
        var = _resolve_model_var(model_id)
        model_var = var.replace('result_', 'model_')
        md = [
            f"## Save Model → `{dest}`\n",
            f"> `tuiml_save_model(model_id=..., destination={repr(dest)})`",
        ]
        # model.save()/Algorithm.load() are a matched pair; a fitted Workflow
        # round-trips through them too, carrying its steps along.
        code = [
            f"{model_var}.save({repr(dest)})\n",
            f"print('Model saved to {dest}')\n",
            "\n",
            f"# Verify reload\n",
            f"from tuiml.base.algorithms import Algorithm\n",
            f"_reloaded = Algorithm.load({repr(dest)})\n",
            f"print('Reloaded:', _reloaded)",
        ]
        return md, code

    # ── tuiml_generate_data ──────────────────────────────────────────────────
    if tool == 'tuiml_generate_data':
        gen = args.get('generator', '')
        n_samples = args.get('n_samples', 100)
        kw = _call_to_kwargs_str({k: v for k, v in args.items() if k != 'generator'})
        md = [
            f"## Generate Synthetic Data, `{gen}`\n",
            f"> `tuiml_generate_data(generator={repr(gen)}, n_samples={n_samples})`",
        ]
        code = [
            f"from tuiml.datasets.generators import {gen}\n",
            f"_gen = {gen}(\n",
            kw + "\n",
            ")\n",
            "_gen_dataset = _gen.generate()\n",
            "print(f'Generated: {_gen_dataset.X.shape}')",
        ]
        return md, code

    # ── tuiml_preprocess ─────────────────────────────────────────────────────
    if tool == 'tuiml_preprocess':
        data = args.get('data', '')
        steps = args.get('steps', [])
        target = args.get('target')
        # Normalize steps to a list of step names.
        step_list = steps if isinstance(steps, list) else [steps]
        step_list = [s for s in step_list if s]
        md = [
            f"## Preprocess Data, {step_list}\n",
            f"> `tuiml_preprocess(data={repr(data)}, steps={step_list})`",
        ]
        code = _data_load_lines(data) + ["\n", "_X_pre = _dataset.X\n"]
        for step in step_list:
            code += [
                f"from tuiml.preprocessing import {step}\n",
                f"_X_pre = {step}().fit_transform(_X_pre)\n",
            ]
        code.append("print(f'Preprocessed shape: {_X_pre.shape}')")
        return md, code

    # ── tuiml_select_features ────────────────────────────────────────────────
    if tool == 'tuiml_select_features':
        data = args.get('data', '')
        method = args.get('method', '')
        target = args.get('target', '')
        k = args.get('k')
        md = [
            f"## Feature Selection, `{method}`\n",
            f"> `tuiml_select_features(data={repr(data)}, method={repr(method)})`",
        ]
        init_args = {}
        if k:
            init_args['k'] = k
        init_str = ', '.join(f'{kk}={repr(vv)}' for kk, vv in init_args.items())
        code = (
            _data_load_lines(data) + ["\n",
            f"from tuiml.features.selection import {method}\n",
            f"_selector = {method}({init_str})\n",
            "_selector.fit(_dataset.X, _dataset.y)\n",
            "_X_selected = _selector.transform(_dataset.X)\n",
            "print(f'Features: {_dataset.X.shape[1]} → {_X_selected.shape[1]}')\n",
            "if hasattr(_selector, 'selected_indices_'):\n",
            "    print('Selected indices:', _selector.selected_indices_)",
        ])
        return md, code

    # ── tuiml_test_statistics ────────────────────────────────────────────────
    if tool == 'tuiml_test_statistics':
        test = args.get('test', '')
        results = args.get('results', {})
        alpha = args.get('significance_level', 0.05)
        md = [
            f"## Statistical Test, `{test}`\n",
            f"> `tuiml_test_statistics(test={repr(test)}, ...)`",
        ]
        # Statistical tests are functions in tuiml.evaluation.statistics, not
        # classes. Map each test to its function and call shape (mirrors the
        # tuiml_test_statistics executor).
        fn_map = {
            'friedman': 'friedman_test', 'nemenyi': 'nemenyi_post_hoc',
            'wilcoxon': 'wilcoxon_signed_rank_test', 'paired_t': 'paired_t_test',
            'anova': 'one_way_anova', 'friedman_aligned': 'friedman_aligned_ranks_test',
            'quade': 'quade_test',
        }
        fn = fn_map.get(test, 'friedman_test')
        code = [
            "import numpy as np\n",
            f"from tuiml.evaluation.statistics import {fn}\n",
            f"_results = {{k: np.array(v, dtype=float) for k, v in {repr(results)}.items()}}\n",
        ]
        if test in ('friedman', 'friedman_aligned', 'quade'):
            code += [
                f"statistic, p_value, significant = {fn}(_results, significance_level={alpha})\n",
                "print('Statistic:', statistic)\n",
                "print('p-value:  ', p_value)\n",
                "print('Significant:', significant)",
            ]
        elif test == 'anova':
            code += [
                f"f_stat, p_value, significant = {fn}(*_results.values(), significance_level={alpha})\n",
                "print('F-statistic:', f_stat)\n",
                "print('p-value:    ', p_value)\n",
                "print('Significant:', significant)",
            ]
        elif test in ('wilcoxon', 'paired_t'):
            code += [
                "_names = list(_results.keys())\n",
                "_x, _y = _results[_names[0]], _results[_names[1]]\n",
                f"_stats = {fn}(_x, _y, significance_level={alpha})\n",
                "print('Statistic:', _stats.t_statistic)\n",
                "print('p-value:  ', _stats.p_value)\n",
                "print('Significant:', _stats.is_significant())",
            ]
        else:  # nemenyi
            code += [
                f"_pairwise = {fn}(_results, significance_level={alpha})\n",
                "for _pair, _sig in _pairwise.items():\n",
                "    print(_pair, '→ significant:', bool(_sig))",
            ]
        return md, code

    # ── tuiml_upload_data ────────────────────────────────────────────────────
    if tool == 'tuiml_upload_data':
        file_path = args.get('file_path', '')
        name = args.get('name', '')
        md = [
            f"## Load Dataset, `{name or file_path}`\n",
            f"> `tuiml_upload_data(file_path={repr(file_path)})`",
        ]
        if file_path:
            code = [
                "import pandas as pd\n",
                f"_df = pd.read_csv({repr(file_path)})\n",
                "print(_df.shape)\n",
                "_df.head()",
            ]
        else:
            code = [f"# Dataset '{name}' was registered inline, recreate from source"]
        return md, code

    return None, None
