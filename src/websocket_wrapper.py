import json
import time
from typing import cast

import pandas as pd
import websocket

from .robot import Robot
from .utils.datatypes import SensorMessage
from .visualization import init_rr, update_rr


class WebsocketWrapper(websocket.WebSocketApp):
    def __init__(self, robot: Robot):
        super().__init__(  # type:ignore
            "ws://91.99.103.188:8765",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        self.robot = robot
        init_rr(self.robot)  # initialize visualizer

    def on_message(self, ws: websocket.WebSocket, message: str) -> None:
        t = time.time() - self.start_time
        msg = json.loads(message)  # convert to dict
        if msg["message_type"] == "score":
            print(msg)
            self.close()  # type:ignore

        elif msg["message_type"] == "sensors":
            sensormessages = cast(list[SensorMessage], msg["sensors"])

            accelerometer_message = sensormessages[0]
            gyro_message = sensormessages[1]
            gps_message = sensormessages[2]

            # unpack data
            accelerometer_data = accelerometer_message["data"]
            gyro_data = gyro_message["data"]
            gps_data = gps_message["data"]

            # unpack timestamps
            inertial_timestamp = pd.Timestamp(
                accelerometer_message["timestamp"]
            )  # assumes gyro_timestamp is equal to accelerometer timestamp
            gps_timestamp = pd.Timestamp(gps_message["timestamp"])

            # First process GPS
            # This will ensure that the last prior is used
            if len(self.robot.gps_hist) > 0:
                if self.robot.gps_hist[-1].timestamp != gps_timestamp:
                    self.robot.process_gps(
                        gps_data=gps_data, gps_timestamp=gps_timestamp
                    )
            else:
                self.robot.process_gps(gps_data=gps_data, gps_timestamp=gps_timestamp)

            # Then compute controls
            # This will ensure either the last prior is used
            # (which predicts the current timestamp)
            # or when gps is available the updated state
            controls = self.robot.controller.compute_controls(
                state=self.robot.state_hist[-1], t=t
            )

            inputs = {
                "v_left": controls.v_l_desired,
                "v_right": controls.v_r_desired,
            }

            ws.send(json.dumps(inputs))

            if len(self.robot.inertial_hist) > 0:
                update_rr(self.robot, t=t)

            # process inertial measurements last such that the last state is now
            # the prior for the next timestep
            self.robot.process_inertial(
                accelerometer_data=accelerometer_data,
                gyro_data=gyro_data,
                inertial_timestamp=inertial_timestamp,
            )

    def on_error(self, ws: websocket.WebSocket, error: Exception) -> None:
        print(error)

    def on_close(
        self, ws: websocket.WebSocket, close_status_code: int, close_msg: str
    ) -> None:
        print("### closed ###")

    def on_open(self, ws: websocket.WebSocket) -> None:
        self.start_time = time.time()
        inputs = {  # start by sending control to go forward
            "v_left": 2.0,
            "v_right": 2.0,
        }
        ws.send(json.dumps(inputs))
        print("Opened connection")
