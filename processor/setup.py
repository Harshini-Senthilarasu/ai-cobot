from setuptools import find_packages, setup

package_name = 'processor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    py_modules=[
        'processor.llm_handler',
        'processor.vision_handler'
    ],
    install_requires=[
        'setuptools', 
        'flask',
        'opencv-python',
        'numpy',
        'pyrealsense2',
        'ultralytics',
        'python-dotenv',
        'google-generativeai'
        ],
    zip_safe=True,
    maintainer='harshini',
    maintainer_email='2102216@sit.singaporetech.edu.sg',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "processor = processor.processor:main"
        ],
    },
)
