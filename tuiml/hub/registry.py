"""Local registry for TuiML components."""

from typing import Dict, List, Optional, Any, Type, Callable, Union

from tuiml.hub.types import ComponentType, Registrable


class Registry:
    """Local registry for TuiML components.

    Provides discovery, registration, and instantiation of local components.
    Supports plugins from external packages.

    Examples
    --------
    >>> from tuiml.hub import registry
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

    def __new__(cls):
        """Singleton pattern - only one hub instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._components = {}
            cls._instance._type_index = {t: [] for t in ComponentType}
            cls._instance._hooks = {"on_register": [], "on_unregister": []}
        return cls._instance

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
        >>> @hub.register("classifier", tags=["ensemble"])
        ... class MyClassifier:
        ...     pass
        """
        # Convert string to ComponentType
        if isinstance(component_type, str):
            try:
                component_type = ComponentType(component_type.lower())
            except ValueError:
                # Default to ALGORITHM if unknown string
                component_type = ComponentType.ALGORITHM

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

            # Warn if already registered
            if component_name in self._components:
                print(f"Warning: Component '{component_name}' is already registered. Overwriting.")

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
        if component_type:
            return list(self._type_index[component_type])
        return list(self._components.keys())

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search components by keyword.

        Parameters
        ----------
        query : str
            Search query string. Matches against component name, description, 
            and tags.

        Returns
        -------
        results : List[dict]
            A list of matching component metadata dictionaries.
        """
        query = query.lower()
        results = []

        for component in self._components.values():
            info = component["info"]
            searchable = (
                f"{info['name']} "
                f"{info['description']} "
                f"{' '.join(info.get('tags', []))}"
            ).lower()

            if query in searchable:
                results.append(info)

        return results

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
        """Clear all registered components (mainly for testing)."""
        self._components.clear()
        self._type_index = {t: [] for t in ComponentType}

    def __contains__(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._components

    def __len__(self) -> int:
        """Return number of registered components."""
        return len(self._components)

# Singleton instance
registry = Registry()
