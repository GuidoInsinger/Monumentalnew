from typing import cast

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .utils import GPSMeasurement, StateVector, gerono


def init_rr() -> None:
    rr.init("test", spawn=True)
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="Map", origin="/map"),
        rrb.Vertical(
            rrb.TimeSeriesView(name="x EKF", origin="/x_ekf"),
            rrb.TimeSeriesView(name="y EKF", origin="/y_ekf"),
            rrb.TimeSeriesView(name="theta EKF", origin="/theta_ekf"),
            rrb.TimeSeriesView(name="v EKF", origin="/v_ekf"),
        ),
    )
    r_wheel = 0.1
    w_wheel = 0.05

    d_body = 0.125
    w_body = 0.25
    h_body = 0.05

    rr.send_blueprint(blueprint=blueprint)
    rr.set_time("time", duration=0.0)
    rr.log(
        "map/box",
        rr.Boxes3D(
            centers=[0, 0, r_wheel],
            half_sizes=[d_body, w_body, h_body],
            fill_mode=3,
        ),
    )
    rr.log(
        "map/box/wheel",
        rr.Ellipsoids3D(
            half_sizes=np.tile(np.array([r_wheel, w_wheel, r_wheel]), 2),
            centers=np.array(
                [[0.0, w_body + w_wheel, r_wheel], [0.0, -(w_body + w_wheel), r_wheel]]
            ),
            fill_mode=3,
        ),
    )

    goal_path = [
        np.hstack((gerono(t=cast(float, t)), np.zeros(1)))
        for t in np.arange(0, 20, 0.01)
    ]
    rr.log("map/goal", rr.LineStrips3D(goal_path))


def update_rr(
    state_hist: list[StateVector], gps_hist: list[GPSMeasurement], t: float
) -> None:
    rr.set_time("time", duration=t)
    rr.log(
        "map/goalpoint",
        rr.Points3D(positions=np.hstack((gerono(t=t), np.zeros(1))), radii=0.05),
    )
    rr.log(
        "map/path",
        rr.LineStrips3D(
            np.array(
                [
                    [state.x for state in state_hist],
                    [state.y for state in state_hist],
                    [0.0 for _ in range(len(state_hist))],
                ]
            ).T,
        ),
    )
    rr.log(
        "map/GPS",
        rr.Points3D(
            positions=[np.hstack((gps.vec, np.zeros(1))) for gps in gps_hist],
            colors=[1.0, 0.65, 0.0, 1.0],
            radii=0.01,
        ),
    )
    rr.log(
        "map/box",
        rr.Transform3D(
            translation=np.array([state_hist[-1].x, state_hist[-1].y, 0.025]),
            rotation_axis_angle=rr.RotationAxisAngle(
                axis=np.array([0, 0, 1]), radians=cast(float, state_hist[-1].theta)
            ),
        ),
    )

    rr.log("x_ekf", rr.Scalars(state_hist[-1].x))
    rr.log("y_ekf", rr.Scalars(state_hist[-1].y))
    rr.log("theta_ekf", rr.Scalars(state_hist[-1].theta))
    rr.log("v_ekf", rr.Scalars(state_hist[-1].v))
