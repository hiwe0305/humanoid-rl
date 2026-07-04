# Thuật toán Balance (Học Đứng Cân Bằng)

Duy trì trạng thái cân bằng cho một humanoid robot có base tự do (floating-base) là bài toán nền tảng. Thay vì dùng một lực nhân tạo (virtual force) tác động vào trọng tâm, thuật toán Reinforcement Learning (RL) học cách điều khiển các khớp để tự cân bằng.

## 1. Công thức Toán và Cấu trúc Hàm Thưởng (Reward Function)

Bản chất của Balance là việc giữ cho **Trọng tâm (Center of Mass - CoM)** nằm bên trong **Đa giác chân đế (Base of Support)** và giữ cho hướng của cơ thể luôn thẳng đứng.

### 1.1. Center of Mass Over Feet (CoM qua chân)
Yếu tố quan trọng nhất. Robot sẽ nhận được điểm cao nhất khi trọng tâm chiếu vuông góc xuống mặt đất nằm ngay giữa hai chân.

Công thức:
$$ R_{com\_over\_feet} = W_{com} \cdot \exp\left(-k_{com} \cdot (dx^2 + dy^2)\right) $$

Trong đó:
- $W_{com} = 5.0$ (Trọng số thưởng mạnh nhất)
- $k_{com} = 100.0$ (Hệ số nhạy, sai lệch nhỏ sẽ bị giảm thưởng mạnh)
- $dx, dy$ là khoảng cách theo trục X và Y từ hình chiếu của CoM đến trung điểm của hai mắt cá chân (Ankle support center).

### 1.2. Orientation (Định hướng cơ thể)
Thưởng cho việc giữ cơ thể thẳng, tức là roll (nghiêng trái/phải) và pitch (cúi/ngửa) tiến về 0.

Công thức:
$$ R_{orientation} = W_{ori} \cdot \exp\left(-k_{ori} \cdot (\text{roll}^2 + \text{pitch}^2)\right) $$

Trong đó:
- $W_{ori} = 3.0$
- $k_{ori} = 10.0$
- Roll, Pitch được tính từ quaternion của thân robot (torso).

### 1.3. Upright (Đứng thẳng) và Alive Bonus (Sống sót)
Bổ sung thưởng để robot không bị sập.

Công thức:
$$ R_{upright} = W_{upright} \cdot \cos(\theta_{tilt}) $$
$$ R_{alive} = W_{alive} \quad \text{nếu} \quad \text{height}_{torso} > \text{height}_{min} $$

Với $\theta_{tilt}$ là góc lệch so với trục Z của thế giới. $W_{upright} = 3.0, W_{alive} = 2.0$.

### 1.4. Phạt Vận tốc (Velocity Penalties)
Khi đứng thăng bằng, robot phải hạn chế tối đa việc di chuyển CoM hoặc lắc lư.

Công thức:
$$ P_{com\_vel} = - W_{cv} \cdot (v_{x\_com}^2 + v_{y\_com}^2) $$
$$ P_{ang\_vel} = - W_{av} \cdot (\text{roll\_rate}^2 + \text{pitch\_rate}^2) $$

Với $W_{cv} = 2.0$ (Phạt tốc độ CoM ngang), $W_{av} = 1.0$ (Phạt tốc độ xoay thân).

## 2. Cách triển khai trong Code

- **Observation Space (Đầu vào mạng RL):** 82 chiều, trong đó bổ sung quan trọng là thông tin CoM so với điểm tựa (`com_rel`), tốc độ CoM (`com_vel_xy`), và định hướng thân (`root_orient`, `root_ang_vel`). Việc cấp cho policy các giá trị đo đạc này giúp nó biết mình đang nghiêng đi đâu.
- **Action Space:** 19 bậc tự do (DOFs) cho các khớp (hip, knee, ankle, shoulder, elbow). Tín hiệu hành động nằm trong khoảng `[-1, 1]`, được scale và cộng vào tư thế mặc định `H1_DEFAULT_QPOS` rồi được chuyển thành torque thông qua PD control.

```python
# Tính CoM trong h1_env.py
com = sum(data.xipos[i] * mass[i] for i in bodies) / total_mass
sup = (pos_left_ankle + pos_right_ankle) / 2.0
com_rel = com[:2] - sup[:2] # dx, dy
```
