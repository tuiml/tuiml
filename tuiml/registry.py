"""Local component registry for TuiML.

Provides the local algorithm/feature/preprocessor registry used by
``@classifier``, ``@regressor``, ``@transformer`` decorators across the
library, plus the shared ``ComponentType`` enum and ``Registrable`` mixin.

Examples
--------
Local component registration::

    from tuiml.registry import registry

    @registry.register("classifier")
    class MyClassifier:
        pass

    model = registry.create("MyClassifier", param=value)
"""

import re
import sys
from abc import ABC
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union



class ComponentType(Enum):
    """Types of components that can be registered."""

    ALGORITHM = "algorithm"
    CLASSIFIER = "classifier"
    CLUSTERER = "clusterer"
    REGRESSOR = "regressor"
    ANOMALY = "anomaly"
    ASSOCIATOR = "associator"
    PREPROCESSOR = "preprocessor"
    TRANSFORMER = "transformer"
    FEATURE_SELECTOR = "feature_selector"
    FEATURE_EXTRACTOR = "feature_extractor"
    FEATURE_CONSTRUCTOR = "feature_constructor"
    METRIC = "metric"
    EVALUATOR = "evaluator"
    TIMESERIES = "timeseries"
    SURVIVAL = "survival"
    UPLIFT = "uplift"


class Registrable(ABC):
    """Base mixin for all registrable components.

    Any class that wants to be registered in the hub should inherit from this.
    """

    _component_type: ComponentType = None
    _component_name: Optional[str] = None

    @classmethod
    def get_component_info(cls) -> Dict[str, Any]:
        """Return component metadata for registration.

        Returns
        -------
        info : dict
            A dictionary containing component information.
        """
        return {
            "name": cls._component_name or cls.__name__,
            "type": cls._component_type.value if cls._component_type else "unknown",
            "description": cls.__doc__ or "No description available",
            "parameters": cls.get_parameter_schema(),
            "version": getattr(cls, "_version", "1.0.0"),
            "author": getattr(cls, "_author", None),
            "tags": getattr(cls, "_tags", []),
        }

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for component parameters.

        Override this method to define custom parameters.

        Returns
        -------
        schema : dict
            A dictionary mapping parameter names to their schemas.
        """
        return {}

#: Splits CamelCase boundaries so ``RandomForestClassifier`` tokenizes into
#: ``random forest classifier`` and multi-word queries can match it.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

#: Only the leading slice of a description is indexed. Full algorithm
#: docstrings run to thousands of words and otherwise swamp search with
#: incidental matches (e.g. "forest" hitting DecisionStumpClassifier).
_DESCRIPTION_INDEX_CHARS = 400

#: Field weights for ranking: a name hit beats a tag hit beats a prose hit.
_WEIGHT_NAME = 3
_WEIGHT_TAGS = 2
_WEIGHT_DESCRIPTION = 1


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase search tokens, breaking CamelCase.

    Parameters
    ----------
    text : str
        The text to tokenize.

    Returns
    -------
    tokens : List[str]
        Lowercase alphanumeric tokens.
    """
    return [
        token
        for token in re.split(r"[^a-z0-9]+", _CAMEL_BOUNDARY.sub(" ", text).lower())
        if token
    ]


def _squash(text: str) -> str:
    """Reduce text to bare lowercase alphanumerics.

    Lets a squashed query such as ``"kmeans"`` still match ``KMeansClusterer``,
    whose tokens are ``k`` and ``means``.

    Parameters
    ----------
    text : str
        The text to squash.

    Returns
    -------
    squashed : str
        Lowercase text with all non-alphanumeric characters removed.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


class Registry:
    """Local registry for TuiML components.

    Provides discovery, registration, and instantiation of local components.
    Supports plugins from external packages.

    Examples
    --------
    >>> from tuiml.registry import registry
    >>> # Register a custom algorithm
    >>> @registry.register("classifier")
    ... class MyClassifier:
    ...     pass
    >>> # List all classifiers
    >>> registry.list(ComponentType.CLASSIFIER)
    >>> # Get a component by name
    >>> cls = registry.get("MyClassifier")
    """

    _instance = None
    _components: Dict[str, Dict[str, Any]] = {}
    _type_index: Dict[ComponentType, List[str]] = {}
    _hooks: Dict[str, List[Callable]] = {
        "on_register": [],
        "on_unregister": [],
    }

    _populated = False

    #: State captured by :meth:`clear`, restored on the next read. The
    #: component modules register via import side effects, which do not repeat
    #: once the module is cached, so re-importing after a clear would register
    #: nothing; this is what makes clear() reversible within a process. None
    #: unless a clear is outstanding.
    _snapshot = None

    def _ensure_populated(self) -> None:
        """Import the component packages so a read sees the whole library.

        Components register themselves as a side effect of being imported, so
        a registry that nothing has imported into is simply empty. This used
        to be arranged by ``tuiml/__init__`` importing the world, which made
        every ``import tuiml`` -- including the one behind ``tuiml --version``
        -- pay for the entire library. Doing it on first read instead keeps
        that cost on the code paths that actually need components.

        Write methods deliberately do not call this: the imports below are
        what invoke :meth:`register`, so populating from there would recurse.
        """
        if Registry._populated:
            return
        # Set before importing, not after: the imports below register
        # components, and any read they perform must not re-enter here.
        Registry._populated = True

        # After clear(), re-importing would register nothing: the modules are
        # already in sys.modules, so their decorators do not run a second time.
        # Restore what clear() put aside instead.
        if Registry._snapshot is not None:
            components, type_index = Registry._snapshot
            Registry._snapshot = None
            self._components.update(components)
            for kind, names in type_index.items():
                self._type_index[kind] = list(names)
            return

        import importlib
        for module in (
            "tuiml.algorithms",
            "tuiml.training",
            "tuiml.benchmarking",
            "tuiml.serving",
            "tuiml.workflow",
            "tuiml.automl",
            "tuiml.agent",
            # Optional bridges: absent unless the extra is installed.
            "tuiml.sklearn",
            "tuiml.capymoa",
            "tuiml.weka",
            "tuiml.foundation",
        ):
            try:
                importlib.import_module(module)
            except Exception:
                # A missing optional bridge, or one broken component, must
                # never make the registry itself unusable.
                pass

    def __new__(cls):
        """Singleton pattern - only one hub instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._components = {}
            cls._instance._type_index = {t: [] for t in ComponentType}
            cls._instance._hooks = {"on_register": [], "on_unregister": []}
            cls._instance._suppress_overwrite_warnings = False
        return cls._instance

    @contextmanager
    def suppress_overwrite_warnings(self):
        """Temporarily silence the "already registered" overwrite warning.

        Used for intentional re-registration, e.g. reloading user algorithms
        at startup or after creating a new version, where overwriting the
        existing entry is the expected behavior, not an accidental name clash.

        Yields
        ------
        None
            The context within which overwrite warnings are suppressed.
        """
        prev = getattr(self, "_suppress_overwrite_warnings", False)
        self._suppress_overwrite_warnings = True
        try:
            yield
        finally:
            self._suppress_overwrite_warnings = prev

    def register(
        self,
        component_type: Union[ComponentType, str],
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: str = "1.0.0",
        author: Optional[str] = None,
    ) -> Callable[[Type], Type]:
        """Decorator to register a component in the hub.

        Parameters
        ----------
        component_type : ComponentType or str
            Type of component being registered.
        name : str, optional, default=None
            Custom name for the component. Defaults to the class name.
        tags : list of str, optional, default=None
            Tags for component discovery.
        version : str, default="1.0.0"
            Component version string.
        author : str, optional, default=None
            Name of the component author.

        Returns
        -------
        decorator : callable
            A class decorator that registers the component.

        Examples
        --------
        >>> from tuiml.registry import registry
        >>> @registry.register("classifier", tags=["ensemble"])
        ... class MyClassifier:
        ...     pass
        """
        # Convert string to ComponentType
        if isinstance(component_type, str):
            try:
                component_type = ComponentType(component_type.lower())
            except ValueError:
                # Raise rather than default to ALGORITHM. A mistyped type used
                # to bury the component in a bucket nothing lists: it would
                # register, report success, and then never appear in the
                # category the author meant -- indistinguishable from a
                # registration that did not happen.
                valid = ", ".join(sorted(t.value for t in ComponentType))
                raise ValueError(
                    f"Unknown component type {component_type!r}. Valid types: {valid}"
                ) from None

        def decorator(cls: Type) -> Type:
            # Set component metadata
            cls._component_type = component_type
            cls._component_name = name or cls.__name__
            cls._tags = tags or []
            cls._version = version
            cls._author = author

            # Get component info
            if hasattr(cls, "get_component_info"):
                info = cls.get_component_info()
            else:
                info = {
                    "name": cls._component_name,
                    "type": component_type.value,
                    "description": cls.__doc__ or "No description available",
                    "parameters": {},
                    "version": version,
                    "author": author,
                    "tags": tags or [],
                }

            component_name = info["name"]

            # Warn if already registered, unless the overwrite is intentional
            # (e.g. reloading a user algorithm). Route to stderr so the message
            # never corrupts an MCP stdio JSON-RPC stream on stdout.
            if (component_name in self._components
                    and not getattr(self, "_suppress_overwrite_warnings", False)):
                print(
                    f"Warning: Component '{component_name}' is already registered. Overwriting.",
                    file=sys.stderr,
                )

            # Store in registry
            self._components[component_name] = {
                "class": cls,
                "info": info,
                "type": component_type,
            }

            # Update type index
            if component_name not in self._type_index[component_type]:
                self._type_index[component_type].append(component_name)

            # Call hooks
            for hook in self._hooks["on_register"]:
                hook(cls, info)

            return cls

        return decorator

    def register_class(
        self,
        cls: Type,
        component_type: ComponentType,
        name: Optional[str] = None,
    ) -> None:
        """Register a class directly (non-decorator usage).

        Parameters
        ----------
        cls : Type
            The class to register.
        component_type : ComponentType
            The type of component.
        name : str, optional, default=None
            Custom name for the component.
        """
        decorator = self.register(component_type, name=name)
        decorator(cls)

    def unregister(self, name: str) -> bool:
        """Remove a component from the registry.

        Parameters
        ----------
        name : str
            The name of the component to remove.

        Returns
        -------
        success : bool
            True if the component was found and removed, False otherwise.
        """
        if name not in self._components:
            return False

        component = self._components[name]
        component_type = component["type"]

        # Call hooks
        for hook in self._hooks["on_unregister"]:
            hook(component["class"], component["info"])

        # Remove from type index
        if name in self._type_index[component_type]:
            self._type_index[component_type].remove(name)

        # Remove from registry
        del self._components[name]
        return True

    def get(self, name: str) -> Type:
        """Get a component class by name.

        Parameters
        ----------
        name : str
            The name of the component to retrieve.

        Returns
        -------
        cls : Type
            The registered component class.

        Raises
        ------
        KeyError
            If the component name is not found in the registry.
        """
        self._ensure_populated()
        if name not in self._components:
            available = ", ".join(self._components.keys())
            raise KeyError(
                f"Component '{name}' not found. Available: {available}"
            )
        return self._components[name]["class"]

    def get_info(self, name: str) -> Dict[str, Any]:
        """Get component metadata by name.

        Parameters
        ----------
        name : str
            The name of the component.

        Returns
        -------
        info : dict
            A dictionary containing component metadata.
        """
        self._ensure_populated()
        if name not in self._components:
            raise KeyError(f"Component '{name}' not found")
        return self._components[name]["info"]

    def create(self, name: str, **kwargs) -> Any:
        """Create an instance of a component.

        Parameters
        ----------
        name : str
            The name of the component to instantiate.
        **kwargs : Any
            Arguments passed to the component's constructor.

        Returns
        -------
        instance : Any
            An instance of the requested component.
        """
        self._ensure_populated()
        cls = self.get(name)
        return cls(**kwargs)

    def list(
        self,
        component_type: Optional[ComponentType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """List components, optionally filtered by type or tags.

        Parameters
        ----------
        component_type : ComponentType, optional, default=None
            Filter components by type.
        tags : list of str, optional, default=None
            Filter components by tags. Components must have ALL specified tags.

        Returns
        -------
        results : List[dict]
            A list of component metadata dictionaries.
        """
        self._ensure_populated()
        results = []

        for name, component in self._components.items():
            # Filter by type
            if component_type and component["type"] != component_type:
                continue

            # Filter by tags
            if tags:
                component_tags = component["info"].get("tags", [])
                if not all(tag in component_tags for tag in tags):
                    continue

            results.append(component["info"])

        return results

    def list_names(
        self,
        component_type: Optional[ComponentType] = None,
    ) -> List[str]:
        """List component names, optionally filtered by type.

        Parameters
        ----------
        component_type : ComponentType, optional, default=None
            Filter components by type.

        Returns
        -------
        names : List[str]
            A list of registered component names.
        """
        self._ensure_populated()
        if component_type:
            return list(self._type_index[component_type])
        return list(self._components.keys())

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search components by keyword, ranked by relevance.

        The query is tokenized and matched against the component name (with
        CamelCase split apart), its tags, and the leading portion of its
        description. A component matches only if *every* query token hits at
        least one of those fields, so multi-word queries such as
        ``"random forest"`` work against names like ``RandomForestClassifier``
        and namespaced keys like ``sklearn.RandomForestClassifier``.

        Parameters
        ----------
        query : str
            Search query string.
        limit : int, optional, default=None
            Maximum number of results to return. ``None`` returns all matches.

        Returns
        -------
        results : List[dict]
            Matching component metadata dictionaries, best match first.

        Examples
        --------
        >>> from tuiml.registry import registry
        >>> [c["name"] for c in registry.search("random forest", limit=3)]
        ['RandomForestClassifier', 'RandomForestRegressor', 'capymoa.AdaptiveRandomForest']
        """
        self._ensure_populated()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored = []

        for component in self._components.values():
            info = component["info"]
            name = info["name"]
            name_tokens = set(_tokenize(name))
            squashed_name = _squash(name)
            tag_tokens = set(_tokenize(" ".join(info.get("tags", []))))
            description_tokens = set(
                _tokenize((info.get("description") or "")[:_DESCRIPTION_INDEX_CHARS])
            )

            def hits_name(token: str) -> bool:
                return token in name_tokens or token in squashed_name

            if not all(
                hits_name(token)
                or token in tag_tokens
                or token in description_tokens
                for token in query_tokens
            ):
                continue

            score = (
                _WEIGHT_NAME * sum(hits_name(t) for t in query_tokens)
                + _WEIGHT_TAGS * sum(t in tag_tokens for t in query_tokens)
                + _WEIGHT_DESCRIPTION * sum(t in description_tokens for t in query_tokens)
            )
            scored.append((score, name, info))

        # Descending score, then name for a stable, deterministic ordering.
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        results = [info for _, _, info in scored]

        if limit is None:
            return results
        # Clamp so a negative limit returns nothing instead of silently
        # slicing off the tail (results[:-1] drops the last match).
        return results[: max(limit, 0)]

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a hook to be called on registry events.

        Parameters
        ----------
        event : str
            The event name ('on_register' or 'on_unregister').
        callback : callable
            The function to call. It will receive (class, info) as arguments.

        Raises
        ------
        ValueError
            If the event name is unknown.
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown event: {event}")
        self._hooks[event].append(callback)

    def clear(self) -> None:
        """Clear all registered components (mainly for testing).

        Resetting ``_populated`` is the point: every read goes through
        :meth:`_ensure_populated`, which short-circuits once that flag is set.
        Emptying the maps while leaving it True makes the registry permanently
        empty rather than merely cleared -- worst in the testing this exists
        for, where it silently drains every later lookup in the process.
        """
        # Snapshot first. Restoring on the next read is the only way back:
        # the component modules populate the registry as an import side effect,
        # and those do not run again once the modules are cached. Capturing
        # here rather than at first population means a clear also returns
        # anything registered since.
        # Only when no clear is already outstanding: a second clear before any
        # read would otherwise snapshot the emptied maps over the good one and
        # make the loss permanent, which is the failure this whole mechanism
        # exists to prevent.
        if Registry._snapshot is None:
            Registry._snapshot = (
                dict(self._components),
                {kind: list(names) for kind, names in self._type_index.items()},
            )
        self._components.clear()
        self._type_index = {t: [] for t in ComponentType}
        Registry._populated = False

    def __contains__(self, name: str) -> bool:
        """Check if a component is registered."""
        self._ensure_populated()
        return name in self._components

    def __len__(self) -> int:
        """Return number of registered components."""
        self._ensure_populated()
        return len(self._components)

# Singleton instance
registry = Registry()
