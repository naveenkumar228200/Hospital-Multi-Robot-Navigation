import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge

from ultralytics import YOLO

import cv2


class HumanDetector(Node):

    def __init__(self):

        super().__init__("human_detector")

        self.bridge = CvBridge()

        self.model = YOLO("yolov8n.pt")

        self.subscription = self.create_subscription(
            Image,
            "/rgb_image",
            self.image_callback,
            10
        )

        self.people_pub = self.create_publisher(
            Float32MultiArray,
            "/people_detection",
            10
        )

        self.get_logger().info(
            "Human Detector Started"
        )

    def image_callback(self, msg):

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="rgb8"
            )

            results = self.model(
                frame,
                verbose=False
            )

            detections = []

            for result in results:

                boxes = result.boxes

                for box in boxes:

                    cls_id = int(box.cls[0])

                    # COCO class 0 = person
                    if cls_id != 0:
                        continue

                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    u = (x1 + x2) / 2.0
                    v = (y1 + y2) / 2.0

                    detections.extend([
                        u,
                        v,
                        conf
                    ])

                    self.get_logger().info(
                        f"Person: u={u:.1f}, "
                        f"v={v:.1f}, "
                        f"conf={conf:.2f}"
                    )

                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        frame,
                        (int(u), int(v)),
                        5,
                        (255, 0, 0),
                        -1
                    )

            msg_out = Float32MultiArray()
            msg_out.data = detections

            self.people_pub.publish(msg_out)

            cv2.imshow(
                "Human Detection",
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )
            )

            cv2.waitKey(1)

        except Exception as e:

            self.get_logger().error(
                f"Detection failed: {e}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = HumanDetector()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
