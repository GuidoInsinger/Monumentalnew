import numpy as np
import rerun as rr
import rerun.blueprint as rrb

blueprint = rrb.Spatial2DView(name="Map", origin="/map", background=[32, 0, 16])

rr.init("genrono", spawn=True)
rr.send_blueprint(blueprint=blueprint)


def gerono(t: np.floating) -> tuple[float, float]:
    if t < 20:
        k = np.pi * t / 10 - np.pi / 2
    else:
        k = 3 * np.pi / 2

    x = -2 * np.sin(k) * np.cos(k)
    y = 2 * (np.sin(k) + 1)

    return x, y


allpoints: list[list[float]] = []
for t in np.arange(0, 20, 0.01):
    x, y = gerono(t)
    allpoints.append([x, y])

rr.log("map/gerono", rr.LineStrips2D(allpoints))
