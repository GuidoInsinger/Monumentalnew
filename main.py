import numpy as np

from src.controller import Controller
from src.EKF import EKF
from src.robot import Robot
from src.utils.datatypes import RobotDimensions, StateVector
from src.websocket_wrapper import WebsocketWrapper

if __name__ == "__main__":
    # ekf parameters
    state0 = StateVector(*np.array([0.0, 0.0, 0.0, 0.0]))
    cov0 = np.diag([1e-4, 1e-4, 1e-5, 1e-4])

    Q = 0.005 * np.diag([3e-3, 3e-3, 4e-3, 4e-3])
    R = np.diag([0.1, 0.1])

    # control parameters
    k1 = 2.5
    k2 = -8.0
    k3 = -12.0

    # these are just dimensions for visualization
    r_wheel = 0.1
    w_wheel = 0.05

    d_body = 0.125
    w_body = 0.25
    h_body = 0.05

    # setup robot
    ekf = EKF(cov0=cov0, Q=Q, R=R)  # setup ekf
    controller = Controller(k1=k1, k2=k2, k3=k3)  # setup controller
    dimensions = RobotDimensions(
        d_body=d_body, w_body=w_body, h_body=h_body, r_wheel=r_wheel, w_wheel=w_wheel
    )  # setup robot dimensions
    robot = Robot(dimensions=dimensions, state0=state0, ekf=ekf, controller=controller)

    # run websocket
    viz_ekf = WebsocketWrapper(robot=robot)  # websocket
    viz_ekf.run_forever()  # type: ignore
