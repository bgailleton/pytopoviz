"""River network routines (composite processes).

Author: B.G.
"""

from __future__ import annotations

from ..core import Output, Param, Port, process
from ..core.registries import PROCESSES

_LINES_DOC = "The channel network as lines, with per-link and per-vertex values."


@process(
    id="pytopoviz.river_network_lsdtt3",
    label="River network",
    inputs=[
        Port("dem", "lsdtt3.Raster", doc="Elevation to trace channels on."),
        Port("filled_dem", "lsdtt3.Raster", optional=True,
             doc="Your own depression-free DEM; skips the filling step."),
        Port("flow_directions", "lsdtt3.FlowInfo", optional=True,
             doc="Flow directions already computed; skips filling and routing."),
        Port("sources", "geopoints", optional=True,
             doc="Channel heads to start from (e.g. drawn); replaces the head detection."),
        Port("outlets", "geopoints", optional=True, doc="Where channels stop."),
    ],
    params=[
        Param("boundary_conditions", "lsdtt3.boundary_conditions", default="oooo"),
        Param("min_slope", "float", default=1e-4, min=0.0,
              doc="Small slope given to filled areas so water keeps flowing."),
        Param("allow_pits_as_outlets", "bool", default=False,
              doc="Let leftover pits end the flow instead of failing."),
        Param("heads_method", "enum", default="threshold", choices=["threshold", "wiener"],
              choice_labels=["Drainage threshold", "Wiener curvature"],
              doc="Channels from every cell above the threshold, or from heads found by "
                  "Wiener-filtered curvature (projected metre DEM). Unused with sources."),
        Param("threshold_contributing_pixels", "int", default=1000, min=1,
              doc="Upstream cells needed to be a channel."),
        Param("pruning_drainage_area_m2", "float", default=1000.0, min=0.0,
              doc="Wiener heads: channels draining less than this are pruned (m²)."),
        Param("surface_fitting_radius_metres", "float", default=6.0, min=0.0,
              doc="Wiener heads: radius of the surface fitted for curvature (m, at least a cell)."),
        Param("connected_components_threshold_pixels", "int", default=100, min=0,
              doc="Wiener heads: channel patches smaller than this are dropped (cells)."),
        Param("snap_sources", "bool", default=True, doc="Move each source to the nearest channel cell."),
        Param("source_snap_max_radius_pixels", "int", default=25, min=0,
              doc="Farthest a source moves when snapped (cells)."),
        Param("outlet_snap_max_distance_metres", "float", default=0.0, min=0.0,
              doc="Farthest an outlet moves onto a channel (m)."),
        Param("chi", "bool", default=True, doc="Add chi to every vertex."),
        Param("m_over_n", "float", default=0.45, min=0.0, doc="Concavity m/n of chi."),
        Param("reference_area", "float", default=1.0, min=0.0,
              doc="Reference drainage area A0 of chi (m²)."),
        Param("ksn", "bool", default=False,
              doc="Add ksn to every vertex, from piecewise fits of the chi-elevation profiles."),
        Param("critical_divergence_metres", "float", default=5.0, min=0.0,
              doc="ksn: largest elevation misfit before a profile segment is split (m)."),
        Param("mainstem", "enum", default="max_flow_length", choices=["max_flow_length", "max_chi"],
              choice_labels=["Longest flow path", "Largest chi"],
              doc="ksn: how each basin's main stem is picked for the fit."),
    ],
    outputs=[
        Output("lines", "geolines", doc=_LINES_DOC),
        Output("network", "lsdtt3.ChannelNetwork", doc="Junction graph, reusable by later runs."),
        Output("flow", "lsdtt3.FlowInfo", doc="Flow directions, reusable by later runs."),
        Output("filled", "lsdtt3.Raster", optional=True,
               doc="The depression-free DEM used (not produced when flow directions are given)."),
        Output("heads", "geopoints", optional=True, doc="Wiener heads: the heads found."),
        Output("segments", "datatable", optional=True,
               doc="ksn: the fitted chi-elevation segments, with their ksn."),
    ],
    impl="composite",
)
def river_network_lsdtt3(
    dem, filled_dem=None, flow_directions=None, sources=None, outlets=None,
    boundary_conditions="oooo", min_slope=1e-4, allow_pits_as_outlets=False,
    heads_method="threshold", threshold_contributing_pixels=1000,
    pruning_drainage_area_m2=1000.0, surface_fitting_radius_metres=6.0,
    connected_components_threshold_pixels=100, snap_sources=True,
    source_snap_max_radius_pixels=25, outlet_snap_max_distance_metres=0.0,
    chi=True, m_over_n=0.45, reference_area=1.0, ksn=False,
    critical_divergence_metres=5.0, mainstem="max_flow_length",
):
    """The channel network of a DEM, as lines.

    Chains lsdtt3.fill_pits_boundary_aware -> lsdtt3.flow_info ->
    [lsdtt3.channel_heads] -> lsdtt3.channel_network -> lsdtt3.network_lines.
    ``filled_dem`` skips the fill, ``flow_directions`` skips fill and routing;
    ``sources`` replaces the head detection. The ksn fit runs on the filled DEM
    (``filled_dem``, else ``dem`` when flow directions are given).
    """
    flow = flow_directions
    filled = filled_dem
    if flow is None:
        if filled is None:
            filled = PROCESSES.get("lsdtt3.fill_pits_boundary_aware")(
                raster=dem, boundary_conditions=boundary_conditions, min_slope=min_slope
            )["filled"]
        flow = PROCESSES.get("lsdtt3.flow_info")(
            dem=filled, boundary_conditions=boundary_conditions,
            allow_pits_as_outlets=allow_pits_as_outlets,
        )["flow"]
    surface = filled if filled is not None else dem

    out = {}
    if sources is None and heads_method == "wiener":
        sources = PROCESSES.get("lsdtt3.channel_heads")(
            dem=surface, flow=flow, method="wiener",
            threshold_contributing_pixels=threshold_contributing_pixels,
            pruning_drainage_area_m2=pruning_drainage_area_m2,
            surface_fitting_radius_metres=surface_fitting_radius_metres,
            connected_components_threshold_pixels=connected_components_threshold_pixels,
        )["heads"]
        out["heads"] = sources
        if sources.n_features == 0:
            raise ValueError("the Wiener method found no channel head")
    network = PROCESSES.get("lsdtt3.channel_network")(
        flow=flow, sources=sources, outlets=outlets,
        threshold_contributing_pixels=threshold_contributing_pixels,
        snap_sources=snap_sources, source_snap_max_radius_pixels=source_snap_max_radius_pixels,
        outlet_snap_max_distance_metres=outlet_snap_max_distance_metres,
    )["network"]
    lines = PROCESSES.get("lsdtt3.network_lines")(
        network=network, dem=surface if ksn else None, chi=chi, m_over_n=m_over_n,
        reference_area=reference_area, ksn=ksn,
        critical_divergence_metres=critical_divergence_metres, mainstem=mainstem,
    )
    out.update({"lines": lines["lines"], "network": network, "flow": flow, "filled": filled})
    if "segments" in lines:
        out["segments"] = lines["segments"]
    return out


@process(
    id="pytopoviz.river_network_topotoolbox",
    label="River network",
    inputs=[
        Port("dem", "topotoolbox.GridObject", doc="Elevation to trace streams on."),
        Port("flow_directions", "topotoolbox.FlowObject", optional=True,
             doc="Flow directions already computed; skips routing."),
        Port("channelheads", "geopoints", optional=True,
             doc="Channel heads: keeps the streams downstream of them; replaces the threshold."),
        Port("outlets", "geopoints", optional=True,
             doc="Keeps the streams upstream of these points (on stream cells)."),
    ],
    params=[
        Param("sink_resolution", "enum", default="carve", choices=["carve", "lcat"],
              choice_labels=["Carve", "Least-cost (lcat)"],
              doc="How flow leaves filled sinks and flats."),
        Param("units", "enum", default="pixels", choices=["pixels", "mapunits", "m2", "km2"],
              choice_labels=["Cells", "Map units²", "m²", "km²"], doc="Units of the threshold."),
        Param("threshold", "float", default=0.0, min=0.0,
              doc="Upstream area needed to be a stream; 0 = automatic."),
        Param("k_largest", "int", default=0, min=0,
              doc="Keep the k largest networks only; 0 = all."),
        Param("trunk", "bool", default=False, doc="Keep the trunk stream of each network only."),
        Param("stream_order", "enum", default="strahler", choices=["none", "strahler", "shreve"],
              choice_labels=["None", "Strahler", "Shreve"], doc="Stream order of every vertex."),
        Param("drainage_area", "bool", default=True, doc="Add the drainage area (m²)."),
        Param("upstream_distance", "bool", default=True,
              doc="Add the distance from the farthest channel head (map units)."),
        Param("downstream_distance", "bool", default=False,
              doc="Add the distance to the outlet (map units)."),
        Param("gradient", "bool", default=False, doc="Add the channel gradient."),
        Param("ksn", "bool", default=False, doc="Add the normalised steepness index."),
        Param("chi", "bool", default=False, doc="Add chi."),
        Param("impose", "bool", default=False,
              doc="Gradient and ksn: impose downstream minima first (no negative slopes)."),
        Param("theta", "float", default=0.45, min=0.0, doc="ksn: reference concavity."),
        Param("mn", "float", default=0.45, min=0.0, doc="chi: m/n ratio."),
        Param("a0", "float", default=1e6, min=0.0, doc="chi: reference area (map units²)."),
    ],
    outputs=[
        Output("lines", "geolines", doc=_LINES_DOC),
        Output("stream", "topotoolbox.StreamObject", doc="Stream network, reusable by later runs."),
        Output("flow", "topotoolbox.FlowObject", doc="Flow directions, reusable by later runs."),
    ],
    impl="composite",
)
def river_network_topotoolbox(
    dem, flow_directions=None, channelheads=None, outlets=None, sink_resolution="carve",
    units="pixels", threshold=0.0, k_largest=0, trunk=False, stream_order="strahler",
    drainage_area=True, upstream_distance=True, downstream_distance=False, gradient=False,
    ksn=False, chi=False, impose=False, theta=0.45, mn=0.45, a0=1e6,
):
    """The stream network of a DEM, as lines.

    Chains topotoolbox.flow_object -> topotoolbox.stream_object ->
    [topotoolbox.upstreamto] -> [topotoolbox.klargestconncomps] ->
    [topotoolbox.trunk] -> topotoolbox.stream_lines. ``flow_directions``
    skips routing.
    """
    flow = flow_directions
    if flow is None:
        flow = PROCESSES.get("topotoolbox.flow_object")(
            grid=dem, sink_resolution=sink_resolution)["flow"]
    stream = PROCESSES.get("topotoolbox.stream_object")(
        flow=flow, channelheads=channelheads, units=units, threshold=threshold)["stream"]
    if outlets is not None:
        stream = PROCESSES.get("topotoolbox.upstreamto")(stream=stream, points=outlets)["upstream"]
    if k_largest > 0:
        stream = PROCESSES.get("topotoolbox.klargestconncomps")(stream=stream, k=k_largest)["largest"]
    if trunk:
        stream = PROCESSES.get("topotoolbox.trunk")(stream=stream)["trunk"]
    lines = PROCESSES.get("topotoolbox.stream_lines")(
        stream=stream, dem=dem, flow=flow, stream_order=stream_order,
        drainage_area=drainage_area, upstream_distance=upstream_distance,
        downstream_distance=downstream_distance, gradient=gradient, ksn=ksn, chi=chi,
        impose=impose, theta=theta, mn=mn, a0=a0,
    )["lines"]
    return {"lines": lines, "stream": stream, "flow": flow}
