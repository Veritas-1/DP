## 🎥 Demos

### Piper Single-Arm Red-on-Green Block Stacking

**Policy:** Diffusion Policy (DP)  
**Robot:** Piper Single Arm  
**Dataset:** 60 Episodes  
**Observation:** Head Camera + Left Wrist Camera  
**Action Space:** 7D (6 Joints + 1 Gripper)

### ✅ Successful Cases

| Demo | Description | Success | Demo | Description | Success |
|:---:|:---|:---:|:---:|:---|:---:|
| [![Success 1](GIF_SUCCESS_1_URL)](https://github.com/user-attachments/assets/d4fd5c6a-b191-4970-a578-39d22432c717) | Successful red-on-green stacking | True | [![Success 2](GIF_SUCCESS_2_URL)](https://github.com/user-attachments/assets/b7683d45-61b9-4941-9747-3b98837854d2) | Successful red-on-green stacking | True |

### ❌ Failure Cases

| Demo | Description | Success | Demo | Description | Success |
|:---:|:---|:---:|:---:|:---|:---:|
| [![Failure 1](GIF_FAILURE_1_URL)](https://github.com/user-attachments/assets/66a6ea00-8fc0-4c67-a09f-8cc1a7195922) | 未能将红色方块置于绿色方块之上，之后寻找不到红色方块| False | [![Failure 2](GIF_FAILURE_2_URL)](https://github.com/user-attachments/assets/e9641125-7b2e-4265-a5a0-8686a08ec2c0) | 未能寻找到红色方块| False |

### Failure Case 3

| Demo | Description | Success |
|:---:|:---|:---:|
| [![Failure 3](GIF_FAILURE_3_URL)](https://github.com/user-attachments/assets/a7150198-76ad-4b98-98d9-6d72919e5669) | 触碰到绿色方块之后下一步推理始终不变（日志：[PIPER7] step=500 target=[-0.1037  1.9095 -1.2354  0.0267  0.7709 -0.1186] gripper=0.647
[PIPER7] step=600 target=[-0.0835  1.895  -1.1746  0.0236  0.7361 -0.099 ] gripper=0.606 | False |

---


