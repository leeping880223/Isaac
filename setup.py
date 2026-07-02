from setuptools import setup, find_packages

setup(
    name="isaac-surgical-tool-detection",
    version="0.1.0",
    description="Isaac Sim 手術器械合成資料生成與 YOLO 辨識",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "ultralytics==8.4.60",
        "opencv-python==4.13.0.92",
        "pyyaml",
        "numpy>=2.0",
        "torch>=2.0",
        "torchvision",
    ],
)
