import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point

class TrajectoryVisualizer(Node):

    def __init__(self):
        super().__init__("trajectory_visualizer")

        self.create_subscription(
            Float32MultiArray,
            "/person_predictions",
            self.callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/trajectory_markers",
            10
        )

        self.get_logger().info(
            "Trajectory Visualizer Started"
        )

    def callback(self, msg):
        marker_array = MarkerArray()
        data = msg.data
        marker_id = 0

        for i in range(0, len(data), 7):
            track_id = int(data[i])

            current_x = float(data[i + 1])
            current_y = float(data[i + 2])

            vx = float(data[i + 3])
            vy = float(data[i + 4])

            future_x = float(data[i + 5])
            future_y = float(data[i + 6])

            stamp = self.get_clock().now().to_msg()

            # -------------------------------------------------
            # Current Position Marker
            # -------------------------------------------------
            current_marker = Marker()
            current_marker.header.frame_id = "odom"
            current_marker.header.stamp = stamp
            current_marker.ns = "current_position"
            current_marker.id = marker_id
            marker_id += 1

            current_marker.type = Marker.SPHERE
            current_marker.action = Marker.ADD

            current_marker.pose.position.x = current_x
            current_marker.pose.position.y = current_y
            current_marker.pose.position.z = 0.25

            current_marker.scale.x = 0.25
            current_marker.scale.y = 0.25
            current_marker.scale.z = 0.25

            current_marker.color.r = 0.0
            current_marker.color.g = 1.0
            current_marker.color.b = 0.0
            current_marker.color.a = 1.0

            marker_array.markers.append(current_marker)

            # -------------------------------------------------
            # Future Position Marker
            # -------------------------------------------------
            future_marker = Marker()
            future_marker.header.frame_id = "odom"
            future_marker.header.stamp = stamp
            future_marker.ns = "future_position"
            future_marker.id = marker_id
            marker_id += 1

            future_marker.type = Marker.SPHERE
            future_marker.action = Marker.ADD

            future_marker.pose.position.x = future_x
            future_marker.pose.position.y = future_y
            future_marker.pose.position.z = 0.25

            future_marker.scale.x = 0.25
            future_marker.scale.y = 0.25
            future_marker.scale.z = 0.25

            future_marker.color.r = 1.0
            future_marker.color.g = 0.0
            future_marker.color.b = 0.0
            future_marker.color.a = 1.0

            marker_array.markers.append(future_marker)

            # -------------------------------------------------
            # Trajectory Arrow
            # -------------------------------------------------
            arrow = Marker()
            arrow.header.frame_id = "odom"
            arrow.header.stamp = stamp
            arrow.ns = "trajectory"
            arrow.id = marker_id
            marker_id += 1

            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            arrow.scale.x = 0.08
            arrow.scale.y = 0.15
            arrow.scale.z = 0.15

            arrow.color.r = 0.0
            arrow.color.g = 0.0
            arrow.color.b = 1.0
            arrow.color.a = 1.0

            p1 = Point()
            p1.x = current_x
            p1.y = current_y
            p1.z = 0.25

            p2 = Point()
            p2.x = future_x
            p2.y = future_y
            p2.z = 0.25

            arrow.points.append(p1)
            arrow.points.append(p2)

            marker_array.markers.append(arrow)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
