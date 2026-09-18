"""Built-in primitive param types, registered into the default TypeRegistry.

These are the scalar param kinds (DESIGN.md §3). Each ``type_id`` equals its kind.
``bool`` is registered before ``int`` because ``bool`` is an ``int`` subclass and
``identify`` returns the first match.

Author: B.G.
"""

from __future__ import annotations

from .registries import register_type

# order matters for identify(): more specific first
register_type("bool", "bool", bool)
register_type("int", "int", int)
register_type("float", "float", float)
register_type("string", "string", str)
register_type("path", "path", lambda v: isinstance(v, str))
register_type("color", "color", lambda v: isinstance(v, (str, tuple, list)))
register_type("enum", "enum", lambda v: isinstance(v, (str, int)))
