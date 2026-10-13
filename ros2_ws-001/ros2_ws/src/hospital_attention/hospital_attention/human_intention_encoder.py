import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

import math


class HumanEncoder(Node):

    def __init__(self):

        super().__init__("human_encoder")

        self.intention_data = None
        self.prediction_data = None

        self.create_subscription(
            Float32MultiArray,
            "/people_intentions",
            self.intention_callback,
            10
        )

        self.create_subscription(
            Float32MultiArray,
            "/person_predictions",
            self.prediction_callback,
            10
        )

        self.pub = self.create_publisher(
            Float32MultiArray,
            "/human_feature",
            10
        )

        self.max_humans = 3

        self.get_logger().info(
            "Human Encoder Started"
        )

    # ======================================
    # One Hot Encoding
    # ======================================

    def one_hot(self, intention):

        if intention == 0:
            return [1.0, 0.0, 0.0, 0.0]

        elif intention == 1:
            return [0.0, 1.0, 0.0, 0.0]

        elif intention == 2:
            return [0.0, 0.0, 1.0, 0.0]

        elif intention == 3:
            return [0.0, 0.0, 0.0, 1.0]

        return [0.0, 0.0, 0.0, 0.0]

    # ======================================
    # Callbacks
    # ======================================

    def intention_callback(self, msg):

        self.intention_data = msg.data

        self.try_encode()

    def prediction_callback(self, msg):

        self.prediction_data = msg.data

        self.try_encode()

    # ======================================
    # Main Encoding Logic
    # ======================================

    def try_encode(self):

        if (
            self.intention_data is None or
            self.prediction_data is None
        ):
            return

        intentions = {}
        predictions = {}

        # ----------------------------------
        # Read Intention Data
        # Format:
        # ID x y speed heading intent
        # ----------------------------------

        data = self.intention_data

        for i in range(0, len(data), 6):

            track_id = int(data[i])

            intentions[track_id] = {
                "x": float(data[i + 1]),
                "y": float(data[i + 2]),
                "speed": float(data[i + 3]),
                "heading": float(data[i + 4]),
                "intent": int(data[i + 5])
            }

        # ----------------------------------
        # Read Prediction Data
        # Format:
        # ID px py vx vy fx fy
        # ----------------------------------

        data = self.prediction_data

        for i in range(0, len(data), 7):

            track_id = int(data[i])

            predictions[track_id] = {
                "future_x": float(data[i + 5]),
                "future_y": float(data[i + 6])
            }

        humans = []

        # ----------------------------------
        # Merge
        # ----------------------------------

        for track_id in intentions:

            if track_id not in predictions:
                continue

            person = intentions[track_id]

            x = person["x"]
            y = person["y"]

            future_x = predictions[track_id]["future_x"]
            future_y = predictions[track_id]["future_y"]

            dx_future = future_x - x
            dy_future = future_y - y

            distance = math.sqrt(
                x * x +
                y * y
            )

            heading_norm = (
                person["heading"] / 180.0
            )

            humans.append(
                (
                    distance,
                    x,
                    y,
                    person["speed"],
                    heading_norm,
                    dx_future,
                    dy_future,
                    person["intent"]
                )
            )

        # ----------------------------------
        # Keep Nearest 3 Humans
        # ----------------------------------

        humans.sort(
            key=lambda h: h[0]
        )

        humans = humans[:self.max_humans]

        encoded = []

        for human in humans:

            (
                _,
                x,
                y,
                speed,
                heading_norm,
                dx_future,
                dy_future,
                intent
            ) = human

            encoded.extend([
                x,
                y,
                speed,
                heading_norm,
                dx_future,
                dy_future
            ])

            encoded.extend(
                self.one_hot(intent)
            )

        # ----------------------------------
        # Padding
        # 3 humans × 10 features = 30
        # ----------------------------------

        while len(encoded) < 30:

            encoded.extend([
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ])

        encoded = encoded[:30]

        out_msg = Float32MultiArray()

        out_msg.data = encoded

        self.pub.publish(out_msg)

        self.get_logger().info(
            f"Published Human Feature Shape = ({len(encoded)},)"
        )


def main(args=None):

    rclpy.init(args=args)

    node = HumanEncoder()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
