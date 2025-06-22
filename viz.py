import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from utils import StateVector


def init_rr():
    rr.init("test", spawn=True)
    blueprint = rrb.Horizontal(
        rrb.Spatial2DView(name="Map", origin="/map"),
        rrb.Vertical(
            rrb.TimeSeriesView(name="x EKF", origin="/x_ekf"),
            rrb.TimeSeriesView(name="y EKF", origin="/y_ekf"),
            rrb.TimeSeriesView(name="theta EKF", origin="/theta_ekf"),
            rrb.TimeSeriesView(name="vl EKF", origin="/vl_ekf"),
            rrb.TimeSeriesView(name="vr EKF", origin="/vr_ekf"),
        ),
    )
    rr.send_blueprint(blueprint=blueprint)


def update_rr(state_hist: list[StateVector], t: float):
    rr.set_time("time", duration=t)

    rr.log(
        "map/box",
        rr.Boxes2D(centers=[state_hist[-1].x, state_hist[-1].y], half_sizes=[0.5, 1]),
    )

    rr.log(
        "map/path",
        rr.LineStrips2D(
            np.array(
                [[state.x for state in state_hist], [state.y for state in state_hist]]
            ).T,
            radii=0.2,
            colors=[0, 255, 255, 255],
        ),
    )
    rr.log("x_ekf", rr.Scalars(state_hist[-1].x))
    rr.log("y_ekf", rr.Scalars(state_hist[-1].y))
    rr.log("theta_ekf", rr.Scalars(state_hist[-1].theta))
    rr.log("vl_ekf", rr.Scalars(state_hist[-1].vl))
    rr.log("vr_ekf", rr.Scalars(state_hist[-1].vr))
