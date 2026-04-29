import pytopoviz as tpz
from cmcrameri import cm
import topotoolbox as ttb

# Load the sample DEM
dem = ttb.load_dem("bigtujunga")
# dem.cellsize = 500.

elevation = tpz.MapObject(dem, cmap=cm.batlowW, cbar="Elevation (m)")
nan_proc = tpz.processor.nan_below()
nan_proc.threshold = 500
elevation.processors.append(nan_proc)

smooth = tpz.processor.gaussian_smooth()
smooth.sigma = 1
elevation.processors.append(smooth)

shade_proc = tpz.processor.multishade()
shade_proc.azimuths = (280.0, 250.0)
shade_proc.alpha = 0.2
elevation.processors.append(shade_proc)

elevation.processors.append(tpz.processor.double_scale())
# elevation.processors.append(tpz.processor.double_scale())

fig = tpz.Fig3DObject(
    background="black",
    smooth_shading=True,
    show_scalar_bar=True,
    # z_exaggeration=1.,
)
fig.add_maps(elevation)
fig.show(screenshot_path="scene.png", auto_close=True)
