"""Quick PyVista 3D helpers for MapObject/GridObject.

Author: B.G.
"""

from typing import Dict, Iterable, Tuple, Union

import numpy as np
import pyvista as pv

from topotoolbox import GridObject

from .map_object import MapObject
from .processing import is_plottable, _processor_is_compatible

__all__ = ["quickmap3d", "Fig3DObject"]


def _ensure_map(obj: Union[MapObject, GridObject]) -> MapObject:
    """Convert GridObject to MapObject when needed."""
    if isinstance(obj, MapObject):
        return obj
    if isinstance(obj, GridObject):
        return MapObject(obj)
    raise TypeError("quickmap3d accepts MapObject or GridObject instances.")


def _structured_grid_from_map(
    mapper: MapObject,
    surface_map: MapObject,
    z_exaggeration: float | None = None,
) -> Tuple[pv.StructuredGrid, str]:
    """Build a StructuredGrid using surface values and mapper scalars."""
    scalars = mapper.value
    z = surface_map.value
    cellsize = getattr(surface_map.grid, "cellsize", 1.0)
    cellsize_value = float(np.asarray(cellsize).ravel()[0]) if cellsize is not None else 1.0
    if cellsize_value <= 0:
        cellsize_value = 1.0
    z_factor = float(surface_map.z_scale_factor)
    if z_exaggeration is None:
        xmin, xmax, ymin, ymax = surface_map.grid.extent
        span_units = max(xmax - xmin, ymax - ymin)
        span_cells = span_units / cellsize_value if cellsize_value > 0 else span_units
        finite = np.isfinite(z)
        z_range = float(np.nanmax(z) - np.nanmin(z)) if finite.any() else 0.0
        if z_range > 0:
            target_relief = 3.2 * span_cells
            z_scale = (target_relief / z_range) * z_factor
        else:
            z_scale = 0.0
    else:
        z_scale = (float(z_exaggeration) / cellsize_value) * z_factor
    ny, nx = z.shape
    xmin, xmax, ymin, ymax = surface_map.grid.extent

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)

    # Geometry uses filled values; scalar array preserves NaNs for transparency.
    grid = pv.StructuredGrid(xx, yy, np.nan_to_num(z, nan=0.0) * z_scale)
    scalars_name = "scalars"
    grid[scalars_name] = scalars.ravel(order="F")

    mask = np.isnan(scalars)
    if mask.any():
        grid = grid.extract_points(~mask.ravel(order="F"), adjacent_cells=True)

    return grid, scalars_name


class Fig3DObject:
    """Wrapper around a PyVista plotter to manage 3D map layers."""

    def __init__(
        self,
        *,
        background: str = "white",
        smooth_shading: bool = True,
        show_scalar_bar: bool = True,
        z_exaggeration: float | None = None,
    ):
        self.plotter = pv.Plotter()
        self.plotter.set_background(background)
        self.plotter.enable_eye_dome_lighting()

        self.smooth_shading = smooth_shading
        self.show_scalar_bar = show_scalar_bar
        self.z_exaggeration = None if z_exaggeration is None else float(z_exaggeration)

        self.meshes: Dict[MapObject, pv.Actor] = {}
        self._light: pv.Light | None = None
        self._light_added = False
        self._pending_camera_position = None
        self.base_maps: list[MapObject] = []

    def add_maps(self, *maps: Union[MapObject, GridObject], surface_map: Union[MapObject, GridObject, None] = None):
        """Add MapObjects to the plotter after applying processors in order."""
        if not maps:
            raise ValueError("Provide at least one MapObject or GridObject to add.")

        map_list_raw = tuple(_ensure_map(m) for m in maps)
        self.base_maps.extend(map_list_raw)
        plot_list: list[tuple[MapObject, MapObject]] = []
        surface_map_override = _ensure_map(surface_map) if surface_map is not None else None
        default_surface = surface_map_override if surface_map_override is not None else map_list_raw[0]

        def collect(current: MapObject, inherited_surface: MapObject):
            surface_map = inherited_surface if current.draped else current
            plot_list.append((current, surface_map))
            for proc in current.processors:
                if not _processor_is_compatible(proc, "3d"):
                    continue
                produced = proc(current)
                if produced is None:
                    continue
                produced_list: Iterable = produced if isinstance(produced, (list, tuple)) else (produced,)
                if surface_map_override is not None:
                    next_inherited = surface_map_override
                else:
                    next_inherited = current if not current.draped else inherited_surface
                for item in produced_list:
                    if not is_plottable(item):
                        continue
                    collect(item, next_inherited)

        for mapper in map_list_raw:
            collect(mapper, default_surface)

        for idx, (mapper, surface_map) in enumerate(plot_list):
            if mapper.eye_dome_lighting is not None:
                if mapper.eye_dome_lighting:
                    self.plotter.enable_eye_dome_lighting()
                else:
                    self.plotter.disable_eye_dome_lighting()

            if (
                mapper.light_azimuth is not None
                or mapper.light_elevation is not None
                or mapper.light_intensity is not None
            ):
                if self._light is None:
                    self._light = pv.Light()
                    self._light.set_scene_light()
                azim = mapper.light_azimuth if mapper.light_azimuth is not None else 315.0
                elev = mapper.light_elevation if mapper.light_elevation is not None else 45.0
                self._light.set_direction_angle(elev, azim)
                if mapper.light_intensity is not None:
                    self._light.intensity = mapper.light_intensity
                if not self._light_added:
                    self.plotter.add_light(self._light)
                    self._light_added = True

            mesh, scalars_name = _structured_grid_from_map(
                mapper,
                surface_map,
                z_exaggeration=self.z_exaggeration,
            )
            scalar_bar_args = {}
            if isinstance(mapper.cbar, str) and self.show_scalar_bar:
                scalar_bar_args["title"] = mapper.cbar
            else:
                scalar_bar_args["title"] = ""

            actor = self.plotter.add_mesh(
                mesh,
                scalars=scalars_name,
                cmap=mapper.cmap,
                clim=(mapper.vmin, mapper.vmax),
                opacity=mapper.alpha,
                nan_opacity=0.0,
                smooth_shading=self.smooth_shading
                if mapper.smooth_shading is None
                else mapper.smooth_shading,
                show_scalar_bar=bool(scalar_bar_args.get("title")),
                scalar_bar_args=scalar_bar_args,
                name=f"map_{idx}",
                ambient=mapper.ambient,
                diffuse=mapper.diffuse,
                specular=mapper.specular,
                specular_power=mapper.specular_power,
            )
            self.meshes[mapper] = actor
        if self._pending_camera_position is not None:
            self.plotter.camera_position = self._pending_camera_position
        return self.plotter

    def set_camera_position(self, camera_position):
        """Set the camera position tuple (position, focal point, view up)."""
        self._pending_camera_position = camera_position
        self.plotter.camera_position = camera_position

    def show(
        self,
        screenshot_path: str = "quickmap3d.png",
        auto_close: bool = False,
        print_camera: bool = True,
    ):
        """Render the scene, optionally save a screenshot, and return camera position."""
        reset_camera = True
        if self._pending_camera_position is not None:
            self.plotter.camera_position = self._pending_camera_position
            reset_camera = False
        elif reset_camera:
            self.plotter.reset_camera()
        try:
            cpos = self.plotter.show(auto_close=auto_close, reset_camera=reset_camera)
        except TypeError:
            cpos = self.plotter.show(auto_close=auto_close)
        if cpos is None:
            cpos = self.plotter.camera_position
        if print_camera:
            print(f"camera_position={cpos}")
        if screenshot_path:
            self.plotter.screenshot(screenshot_path)
        if auto_close:
            self.plotter.close()
        return cpos

    def save(self, screenshot_path: str = "quickmap3d.png", transparent_background: bool = False):
        """Render off-screen and save a screenshot without showing a window."""
        was_off_screen = self.plotter.off_screen
        self.plotter.off_screen = True
        if self._pending_camera_position is not None:
            self.plotter.camera_position = self._pending_camera_position
        self.plotter.render()
        self.plotter.screenshot(screenshot_path, transparent_background=transparent_background)
        self.plotter.off_screen = was_off_screen


def quickmap3d(
    *maps: Union[MapObject, GridObject],
    screenshot_path: str = "quickmap3d.png",
    background: str = "white",
    smooth_shading: bool = True,
    show_scalar_bar: bool = True,
    z_exaggeration: float | None = None,
    camera_position=None,
    print_camera: bool = True,
    surface_map: Union[MapObject, GridObject, None] = None,
):
    """Display one or more MapObjects in 3D with PyVista and save a screenshot."""
    if len(maps) == 0:
        raise ValueError("Provide at least one MapObject or GridObject.")

    fig = Fig3DObject(
        background=background,
        smooth_shading=smooth_shading,
        show_scalar_bar=show_scalar_bar,
        z_exaggeration=z_exaggeration,
    )
    if camera_position is not None:
        fig.set_camera_position(camera_position)
    fig.add_maps(*maps, surface_map=surface_map)
    return fig.show(screenshot_path=screenshot_path, auto_close=False, print_camera=print_camera)
