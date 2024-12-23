API = "IZiSYgNQKAQpV47qj08K"
from roboflow import Roboflow
rf = Roboflow(api_key=API)
project = rf.workspace("mohamed-traore-2ekkp").project("gtsdb---german-traffic-sign-detection-benchmark")
version = project.version(3)
dataset = version.download("yolov11")

