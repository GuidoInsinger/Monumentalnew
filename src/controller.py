from dataclasses import dataclass

import numpy as np

from src.utils import Controls, StateVector, angle_between_vectors, gerono


@dataclass
class Controller:
    k1: float
    k2: float
    k3: float

    t_epsilon: float = 1e-4

    def compute_controls(self, state: StateVector, t: float) -> Controls:
        path_dir = gerono(t=t + self.t_epsilon) - gerono(t=t)
        forward_dir = np.array([np.cos(state.theta), np.sin(state.theta)])

        path_velocity = np.linalg.norm(path_dir / self.t_epsilon)

        path_omega = (
            angle_between_vectors(
                gerono(t=t) - gerono(t=t - self.t_epsilon),
                gerono(t=t + self.t_epsilon) - gerono(t=t),
            )
            / self.t_epsilon
        )

        theta_e = angle_between_vectors(
            forward_dir,
            gerono(t=t + self.t_epsilon) - gerono(t=t),
        )
        # print(path_angle)
        pos_error = gerono(t=t) - state.vec[:2]
        x_e, y_e = (
            np.array(
                [
                    [
                        np.cos(state.theta),
                        np.sin(state.theta),
                    ],
                    [
                        -np.sin(state.theta),
                        np.cos(state.theta),
                    ],
                ]
            )
            @ pos_error
        )

        v_des = path_velocity + self.k1 * x_e
        omega_des = -path_omega + self.k2 * y_e + self.k3 * np.sin(theta_e)

        print(path_omega)
        controls = Controls(
            *np.array([v_des + omega_des * 0.25, v_des - omega_des * 0.25])
        )
        return controls
