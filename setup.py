from setuptools import find_packages, setup

package_name = 'oak_yolo_wall'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='elena.algorri@th-koeln.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_seg_detections = oak_yolo_wall.yolo_seg_detections:main',
            'yolo_seg_detections_optimized = oak_yolo_wall.yolo_seg_detections_optimized:main',
            'tape_wall_from_polygons = oak_yolo_wall.tape_wall_from_polygons:main',
            'tape_wall_polygons_homography = oak_yolo_wall.tape_wall_polygons_homography:main',
            'yolo_overlay_combiner = oak_yolo_wall.yolo_overlay_combiner:main',
        ],
    },
)
