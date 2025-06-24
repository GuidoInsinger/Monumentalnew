from typing import cast

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .robot import Robot
from .utils import gerono


def init_rr(robot: Robot) -> None:
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

    rr.send_blueprint(blueprint=blueprint)
    rr.set_time("time", duration=0.0)
    rr.log(
        "map/box",
        rr.Boxes3D(
            centers=robot.dimensions.body_center,
            half_sizes=robot.dimensions.body_half_sizes,
            fill_mode=3,
        ),
    )
    rr.log(
        "map/box/wheel",
        rr.Ellipsoids3D(
            half_sizes=robot.dimensions.wheel_half_sizes,
            centers=robot.dimensions.wheel_centers,
            fill_mode=3,
        ),
    )

    goal_path = [
        np.hstack((gerono(t=cast(float, t)), np.zeros(1)))
        for t in np.arange(0, 20, 0.01)
    ]
    rr.log("map/goal", rr.LineStrips3D(goal_path))


def update_rr(robot: Robot, t: float) -> None:
    last_state = robot.state_hist[-1]
    rr.set_time("time", duration=t)
    rr.log(
        "map/goalpoint",
        rr.Arrows3D(
            vectors=np.array([0, 0, -1]),
            origins=np.hstack((gerono(t=t), np.ones(1))),
        ),
    )
    rr.log(
        "map/path",
        rr.LineStrips3D(
            np.array(
                [
                    [state.x for state in robot.state_hist],
                    [state.y for state in robot.state_hist],
                    [0.0 for _ in range(len(robot.state_hist))],
                ]
            ).T,
        ),
    )
    rr.log(
        "map/GPS",
        rr.Points3D(
            positions=[np.hstack((gps.vec, np.zeros(1))) for gps in robot.gps_hist],
            colors=[1.0, 0.65, 0.0, 1.0],
            radii=0.01,
        ),
    )
    rr.log(
        "map/accelerometer",
        rr.Points3D(
            positions=[np.hstack((gps.vec, np.zeros(1))) for gps in robot.gps_hist],
            colors=[1.0, 0.65, 0.0, 1.0],
            radii=0.01,
        ),
    )
    rr.log(
        "map/box",
        rr.Transform3D(
            translation=np.array([last_state.x, last_state.y, 0.025]),
            rotation_axis_angle=rr.RotationAxisAngle(
                axis=np.array([0, 0, 1]),
                radians=cast(float, last_state.theta),
            ),
        ),
    )

    rr.log("x_ekf", rr.Scalars(last_state.x))
    rr.log("y_ekf", rr.Scalars(last_state.y))
    rr.log("theta_ekf", rr.Scalars(last_state.theta))
    rr.log("v_ekf", rr.Scalars(last_state.v))
