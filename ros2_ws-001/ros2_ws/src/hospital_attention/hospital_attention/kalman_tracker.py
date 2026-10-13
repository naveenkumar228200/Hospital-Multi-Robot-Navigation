import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

from filterpy.kalman import KalmanFilter

import numpy as np
import time
import math


class KalmanPredictor(Node):

    def __init__(self):

        super().__init__("kalman_predictor")

        self.create_subscription(
            Float32MultiArray,
            "/people_tracks",
            self.track_callback,
            10
        )

        self.pred_pub = self.create_publisher(
            Float32MultiArray,
            "/person_predictions",
            10
        )

        self.filters = {}

        self.last_time = {}

        self.get_logger().info(
            "Kalman Predictor Started"
        )

    # -----------------------------------
    # Create Kalman Filter
    # -----------------------------------

    def create_filter(self, x, y):

        kf = KalmanFilter(
            dim_x=4,
            dim_z=2
        )

        # State:
        # x y vx vy

        kf.x = np.array([
            x,
            y,
            0.0,
            0.0
        ])

        kf.F = np.array([
            [1,0,1,0],
            [0,1,0,1],
            [0,0,1,0],
            [0,0,0,1]
        ])

        kf.H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])

        kf.P *= 100.0

        kf.R *= 0.1

        kf.Q *= 0.01

        return kf

    # -----------------------------------
    # Callback
    # -----------------------------------

    def track_callback(self, msg):

        data = msg.data

        output = []

        now = time.time()

        for i in range(0, len(data), 4):

            track_id = int(data[i])

            x = float(data[i+1])

            y = float(data[i+2])

            # ------------------------
            # First observation
            # ------------------------

            if track_id not in self.filters:

                self.filters[track_id] = \
                    self.create_filter(x,y)

                self.last_time[track_id] = now

            kf = self.filters[track_id]

            dt = now - self.last_time[track_id]

            self.last_time[track_id] = now

            if dt < 0.01:
                dt = 0.01

            # ------------------------
            # Update transition matrix
            # ------------------------

            kf.F = np.array([
                [1,0,dt,0],
                [0,1,0,dt],
                [0,0,1,0],
                [0,0,0,1]
            ])

            # ------------------------
            # Prediction step
            # ------------------------

            kf.predict()

            # ------------------------
            # Measurement update
            # ------------------------

            kf.update(
                np.array([x,y])
            )

            px = float(kf.x[0])

            py = float(kf.x[1])

            vx = float(kf.x[2])

            vy = float(kf.x[3])

            speed = math.sqrt(
                vx*vx + vy*vy
            )

            if speed > 2.0:

                vx = 0.0
                vy = 0.0

            horizon = 10.0

            future_x = px + vx*horizon

            future_y = py + vy*horizon

            output.extend([
                float(track_id),

                px,
                py,

                vx,
                vy,

                future_x,
                future_y
            ])

            self.get_logger().info(
                f"ID={track_id} "
                f"Pos=({px:.2f},{py:.2f}) "
                f"Vel=({vx:.2f},{vy:.2f}) "
                f"Future=({future_x:.2f},{future_y:.2f})"
            )

        out_msg = Float32MultiArray()

        out_msg.data = output

        self.pred_pub.publish(out_msg)


def main(args=None):

    rclpy.init(args=args)

    node = KalmanPredictor()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
