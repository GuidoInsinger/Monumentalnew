# import rerun.blueprint as rrb
import json
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import websocket

start = time.time()


rr.init("rerun_gyro", spawn=True)

blueprint = rrb.Horizontal(
    rrb.Spatial2DView(name="Map", origin="/map"),
    rrb.Vertical(
        rrb.TimeSeriesView(name="GPS x", origin="/xgps"),
        rrb.TimeSeriesView(name="GPS y", origin="/ygps"),
        rrb.TimeSeriesView(name="Accelerometer x", origin="/xacc"),
        rrb.TimeSeriesView(name="Accelerometer y", origin="/yacc"),
        rrb.TimeSeriesView(name="Gyro", origin="/gyro"),
    ),
)
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

sensorhistory: list[list[float]] = []


def on_message(ws: websocket.WebSocket, message: str):
    t = time.time() - start
    msg = json.loads(message)
    if msg["message_type"] == "score":
        print(msg)

    else:
        sensors = msg["sensors"]

        accdata = sensors[0]["data"]
        gyrodata = sensors[1]["data"]
        gpsdata = sensors[2]["data"]

        if len(sensorhistory) == 0:
            sensorhistory.append([*accdata, *gyrodata, *gpsdata])

        elif gpsdata != sensorhistory[-1][3:]:
            print("new!")
            sensorhistory.append([*accdata, *gyrodata, *gpsdata])

        rr.set_time("time", duration=t)

        rr.log("map/box", rr.Boxes2D(centers=gpsdata, half_sizes=[0.5, 1]))
        rr.log(
            "map/path",
            rr.LineStrips2D(
                [s[3:] for s in sensorhistory], radii=0.2, colors=[0, 255, 255, 255]
            ),
        )

        rr.log("xgps", rr.Scalars(gpsdata[0]))
        rr.log("ygps", rr.Scalars(gpsdata[1]))
        rr.log("xacc", rr.Scalars(accdata[0]))
        rr.log("yacc", rr.Scalars(accdata[1]))
        rr.log("gyro", rr.Scalars(gyrodata[0]))

        inputs = {
            "v_left": np.cos(t),
            "v_right": np.sin(t),
        }
        ws.send(json.dumps(inputs))


def on_error(ws: websocket.WebSocket, error: Exception):
    print(error)


def on_close(ws: websocket.WebSocket, close_status_code: int, close_msg: str):
    print("### closed ###")


def on_open(ws: websocket.WebSocket):
    print("Opened connection")


if __name__ == "__main__":
    # websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        "ws://91.99.103.188:8765",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever()  # type: ignore
