import sys
sys.path.append('/home/harshini/capstone_venv/lib/python3.10/site-packages')
from pyRobotiqGripper import RobotiqGripper

gripper = RobotiqGripper()
gripper.resetActivate()
gripper.goTo(0)
gripper.goTo(255)
gripper.printInfo()