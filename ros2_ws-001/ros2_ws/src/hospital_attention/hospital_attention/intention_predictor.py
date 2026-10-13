import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

import math


class IntentionPredictor(Node):

    def __init__(self):

        super().__init__("intention_predictor")

        self.create_subscription(
            Float32MultiArray,
            "/person_predictions",
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            Float32MultiArray,
            "/people_intentions",
            10
        )

        self.prev_heading = {}

        self.get_logger().info(
            "Intention Predictor Started"
        )

    def normalize_angle(self, angle):

        while angle > 180:
            angle -= 360

        while angle < -180:
            angle += 360

        return angle

    def callback(self, msg):

        data = msg.data

        output = []

        labels = {
            0: "STOP",
            1: "STRAIGHT",
            2: "LEFT",
            3: "RIGHT"
        }

        for i in range(0, len(data), 7):

            track_id = int(data[i])

            x = float(data[i + 1])
            y = float(data[i + 2])

            vx = float(data[i + 3])
            vy = float(data[i + 4])

            future_x = float(data[i + 5])
            future_y = float(data[i + 6])

            speed = math.sqrt(
                vx * vx +
                vy * vy
            )

            # --------------------------------
            # Compute heading from trajectory
            # --------------------------------

            dx = future_x - x
            dy = future_y - y

            heading = math.degrees(
                math.atan2(dy, dx)
            )

            # --------------------------------
            # Default
            # --------------------------------

            intention = 1

            # --------------------------------
            # STOP
            # --------------------------------

            if speed < 0.05:

                intention = 0

            else:

                if track_id in self.prev_heading:

                    delta_heading = (
                        heading -
                        self.prev_heading[track_id]
                    )

                    delta_heading = self.normalize_angle(
                        delta_heading
                    )

                    # -------------------------
                    # LEFT
                    # -------------------------

                    if delta_heading > 15:

                        intention = 2

                    # -------------------------
                    # RIGHT
                    # -------------------------

                    elif delta_heading < -15:

                        intention = 3

                    # -------------------------
                    # STRAIGHT
                    # -------------------------

                    else:

                        intention = 1

            self.prev_heading[track_id] = heading

            output.extend([
                float(track_id),
                float(x),
                float(y),
                float(speed),
                float(heading),
                float(intention)
            ])

            self.get_logger().info(
                f"ID={track_id} "
                f"Speed={speed:.2f} "
                f"Heading={heading:.1f} "
                f"Intent={labels[intention]}"
            )

        out_msg = Float32MultiArray()

        out_msg.data = output

        self.pub.publish(out_msg)


def main(args=None):

    rclpy.init(args=args)

    node = IntentionPredictor()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
