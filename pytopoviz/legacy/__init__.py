"""Legacy pre-alpha figure code (map_object, fig2d/3d, processors, styles).

Throwaway alpha kept for reference only; being rethought from scratch as
``figmaker`` (DESIGN.md §9). Not part of the framework. Modules are imported
directly (e.g. ``from pytopoviz.legacy.fig2d import quickmap``) or via the lazy
attributes on the top-level package. Heavy optional deps (pyvista, matplotlib)
are pulled only when a specific legacy module is imported.
"""
