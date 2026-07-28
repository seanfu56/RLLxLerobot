# piper-lerobot

Piper 機械手臂 + LeRobot framework 的 imitation learning 專案。

包含資料收集/訓練/評估腳本、相關工具，以及兩個 lab 專用 LeRobot plugin（ROBOTIS leader、keyboard teleoperator）。
Piper follower、OMY leader、OMY→Piper retarget 三個 plugin 已獨立發佈，安裝時 clone 到 `plugins/` 底下（見 Quick Start）。

## list

```
plugins/          LeRobot plugins：2 個 lab 專用（內附）+ 3 個發佈版（clone 進來，見 Quick Start）
scripts/          操作腳本（teleop、record、train、eval）
tools/            維護與 debug 工具
config/           udev rules 等硬體設定
doc/              完整文件（架構、各 phase 記錄、踩坑筆記）
waypoints/        Waypoint 軌跡檔（JSON）
```

## Quick Start

### Hardware Requirements

- **Follower**: AgileX Piper 機械手臂（CAN bus 連接）
- **Leader**: ROBOTIS OMY-L100 leader arm（USB Serial, Dynamixel Protocol 2.0）
- **Camera**: USB camera（例如 Logitech C270 overhead + ARC wrist cam）
- **USB-CAN adapter**: 連接 Piper 用

### 1. Environment Setup

```bash
conda create -n piper python=3.10
conda activate piper

git clone https://github.com/seanfu56/RLLxLerobot.git
cd RLLxLerobot

git clone https://github.com/huggingface/lerobot.git
cd lerobot && git checkout v0.4.4

pip install -e ".[dev]"

cd ..

# 安裝 Piper SDK
pip install piper_sdk

# 已發佈的 plugin：clone 到 plugins/ 底下（各自獨立維護的 repo，本 repo 不 vendor）
cd plugins
git clone https://github.com/charlie8612/lerobot_robot_piper.git
git clone https://github.com/charlie8612/lerobot_teleoperator_omy.git
git clone https://github.com/charlie8612/lerobot_omy_piper_retarget.git
cd ..

# 安裝全部 plugins（3 個 clone 進來的 + 2 個本 repo 內附的 lab 專用 plugin）
pip install -e plugins/lerobot_robot_piper          # Piper follower（piper_follower / bi_piper_follower）
pip install -e plugins/lerobot_teleoperator_omy     # OMY leader（omy_leader，純讀角度）
pip install -e plugins/lerobot_omy_piper_retarget   # OMY→Piper 映射 processor
pip install -e plugins/lerobot-teleoperator-robotis # lab 專用：ROBOTIS leader（內建映射，目前 teleop 用這個）
pip install -e plugins/lerobot-teleoperator-keypad  # lab 專用：鍵盤 teleop
```

### 2. Hardware Setup 

For the sudoer

```bash
sudo cp config/99-usb-camera.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# 把使用者加入 video group（存取 camera 用）
sudo usermod -aG video $USER
# 加完後需重新登入
```

For the user

```bash
啟動 CAN bus（每次開機後都要跑一次）
bash scripts/0_can_up.sh
```

### 3. Teleoperate

用 leader arm 控制 Piper：

```bash
# 純關節控制（無 camera）
bash scripts/2_teleoperate_leader.sh

# 加一顆 camera + GUI 預覽
bash scripts/3_teleoperate_single_cam.sh
```

按 `Ctrl+C` 結束。結束時手臂會自動移到 rest position 再斷電。

### 4. 錄製資料集

```bash
# 錄製 50 個 episodes（可在腳本內調整）
bash scripts/2_record_leader_cam.sh
```

錄製時快捷鍵：
| 按鍵 | 功能 |
|------|------|
| → (右箭頭) | 提前結束當前 episode |
| ← (左箭頭) | 重錄當前 episode |
| Esc | 停止錄製，正常退出 |

> 不要用 Ctrl+C 結束錄製，會導致 camera 資源無法釋放。

### 5. 訓練模型

```bash
# Diffusion Policy
bash scripts/5_train_diffusion.sh

# SmolVLA（輕量 VLA，推薦）
bash scripts/7_train_smolvla.sh
```

### 6. 評估模型

```bash
# SmolVLA eval
bash scripts/7_eval_smolvla_piper.sh
```

### 7. 工具

```bash
# 讀取 Piper 當前關節角度
python tools/read_piper_pose.py
python tools/read_piper_pose.py --rest-dict    # 輸出 REST_STATE_DEG 格式
python tools/read_piper_pose.py --waypoint     # 輸出 waypoint JSON 格式

# 錄製 camera 影片
python tools/record_cam.py
python tools/record_cam.py --device /dev/cam_c270 -o /tmp2/charlie/video.mp4

# Waypoint 錄製與播放（固定流程自動執行，不需訓練）
python tools/waypoint.py record -o waypoints/my_task.json
python tools/waypoint.py execute waypoints/my_task.json --speed 45 --loop 3 --loop-delay 5

# 清除殘留的 lerobot process
bash tools/kill_lerobot.sh
```

## 詳細文件

詳見 [doc/00-infra-overview.md](doc/00-infra-overview.md)。
