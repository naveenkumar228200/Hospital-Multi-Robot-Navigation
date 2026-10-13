import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

import math
import time


class HumanTracker(Node):

    def __init__(self):

        super().__init__("human_tracker")

        self.create_subscription(
            Float32MultiArray,
            "/people_world_position",
            self.people_callback,
            10
        )

        self.pub = self.create_publisher(
            Float32MultiArray,
            "/people_tracks",
            10
        )

        # track_id -> {x,y,last_seen}
        self.tracks = {}

        self.next_id = 1

        self.max_distance = 1.0

        self.track_timeout = 2.0

        self.get_logger().info(
            "Human Tracker Started"
        )

    def distance(self, x1, y1, x2, y2):

        return math.sqrt(
            (x1 - x2) ** 2 +
            (y1 - y2) ** 2
        )

    def people_callback(self, msg):

        detections = []

        current_time = time.time()

        data = msg.data

        for i in range(0, len(data), 3):

            x = float(data[i])
            y = float(data[i + 1])
            conf = float(data[i + 2])

            detections.append(
                (x, y, conf)
            )

        # -------------------------
        # Remove old tracks
        # -------------------------

        alive_tracks = {}

        for track_id, track in self.tracks.items():

            if current_time - track["last_seen"] < self.track_timeout:

                alive_tracks[track_id] = track

        self.tracks = alive_tracks

        used_tracks = set()

        output = []

        # -------------------------
        # Associate detections
        # -------------------------

        for x, y, conf in detections:

            best_id = None

            best_distance = 9999.0

            for track_id, track in self.tracks.items():

                if track_id in used_tracks:
                    continue

                d = self.distance(
                    x,
                    y,
                    track["x"],
                    track["y"]
                )

                if d < best_distance:

                    best_distance = d

                    best_id = track_id

            # -------------------------
            # Existing track
            # -------------------------

            if (
                best_id is not None
                and
                best_distance < self.max_distance
            ):

                track_id = best_id

                used_tracks.add(track_id)

            else:

                track_id = self.next_id

                self.next_id += 1

            # -------------------------
            # Update track
            # -------------------------

            self.tracks[track_id] = {
                "x": x,
                "y": y,
                "last_seen": current_time
            }

            output.extend([
                float(track_id),
                float(x),
                float(y),
                float(conf)
            ])

            self.get_logger().info(
                f"ID={track_id} "
                f"x={x:.2f} "
                f"y={y:.2f}"
            )

        out_msg = Float32MultiArray()

        out_msg.data = output

        self.pub.publish(out_msg)


def main(args=None):

    rclpy.init(args=args)

    node = HumanTracker()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
