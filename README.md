# ai-cobot

## Project Overview
This repository is organised into key modules:
- **`processor/`** - Handles sensor data processing from camera, AI model inference, and decision-making logic
- **`tm_commander/`** - Sends commands to TM5-700 Omron Cobot and gets feedback from ROS2 topics
- **`user_interface/`** - Enables interaction between user and system
- **`test_files/`** - Includes scripts and data for testing individual components of system

--

## Repository Structure
<pre><code>ai-cobot/
|-- processor/processor # AI and data processing module
    |-- llm_handler.py
    |-- processor.py
    |-- vision_handler.py
|-- tm_commander/tm_commander # TM5-700 cobot command 
    |-- commander.py
|-- user_interface/user_interface # User interface layer
    |-- user_interface.py
    |-- templates/index.html
|-- test_files/ # Test and validation scripts</code></pre>

