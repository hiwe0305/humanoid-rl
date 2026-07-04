# Thuật toán Các Locomotion Tasks

Sau khi có base là "Balance" policy (kỹ năng đứng cân bằng), chúng ta sử dụng kiến thức này làm nền tảng (warm-start) để train 4 task locomotion phức tạp hơn: Walk, Run, Stair, và Jump. Các task này đều dựa chung vào một cơ chế hàm thưởng (reward mechanism) cốt lõi là **LocomotionRewards**.

## 1. Công thức Toán cho Locomotion

Các task di chuyển đều yêu cầu theo dõi một mục tiêu vận tốc.

### 1.1. Forward Velocity (Bám sát vận tốc đích)
Khuyến khích robot đạt được vận tốc dọc theo trục X (`vx`) gần với `target_speed`.

Công thức:
$$ R_{vel\_fwd} = W_{fwd} \cdot \exp\left(-(v_x - v_{target})^2\right) $$

Sử dụng hàm mũ lõm (Gaussian kernel) giúp gradient trả về mạnh mẽ hơn khi robot đạt gần tới `target_speed`, thay vì hàm tuyến tính thông thường.

### 1.2. Backward & Lateral Penalties (Phạt hướng di chuyển sai)
Để đảm bảo robot không đi lùi hoặc dạt sang hai bên, hai hình phạt bình phương được áp dụng:

Công thức phạt đi lùi (chỉ kích hoạt nếu $v_x < 0$):
$$ P_{back} = - W_{back} \cdot v_x^2 $$

Công thức phạt trượt ngang:
$$ P_{lat} = - W_{lat} \cdot v_y^2 $$

Mức độ phạt đi lùi cực kỳ mạnh (thường $W_{back} = 3.0$ trong bài Walk) nhằm rèn robot phải duy trì mô-men tiến lên phía trước.

## 2. Chi tiết từng Task

### 2.1. Walk Task (Đi bộ)
- **Mục tiêu:** Di chuyển tới trước với $v_{target} = 1.0$ m/s, cố gắng giữ cơ thể đứng thẳng (upright).
- **Trọng số:**
  - $W_{fwd} = 1.0$, $W_{back} = 3.0$
  - Upright được giữ nguyên quan trọng ($W_{upright} = 1.5$)

### 2.2. Run Task (Chạy)
- **Mục tiêu:** Tốc độ cao $v_{target} = 3.0$ m/s trên địa hình mấp mô (rough terrain).
- **Khác biệt:** 
  - $W_{fwd} = 3.0$ (Tăng cường để khuyến khích vận tốc cao nhất có thể).
  - Trọng số phạt năng lượng ($torque\_penalty$) có thể nới lỏng để ưu tiên lực đẩy lớn.

### 2.3. Stair Task (Leo cầu thang)
- **Mục tiêu:** Di chuyển tới phía trước trên cầu thang (chênh lệch độ cao).
- **Phần thưởng mở rộng:**
  - **Height Gain:** Thưởng cho việc nâng cao trọng tâm qua từng bước thời gian.
    $$ R_{h\_gain} = W_{hg} \cdot \max(0, \Delta \text{height}_{torso}) $$
  - **Vertical Velocity:** Thưởng khi có lực hướng lên (đi lên thang).
    $$ R_{vel\_z} = W_{vz} \cdot \max(0, v_z) $$
  - Các trọng số này ($W_{hg}=3.0$, $W_{vz}=0.5$) tạo động lực cho robot vượt qua bậc.

### 2.4. Jump Task (Nhảy rào)
- **Mục tiêu:** Phóng qua các rào chắn cao 25cm.
- **Khác biệt:**
  - Bật nhảy yêu cầu torque bộc phát. $W_{vz} = 2.0$ (Thưởng mạnh cho vận tốc trục Z).
  - Hình phạt năng lượng giảm đi rất nhiều ($0.0002$) vì nhảy yêu cầu tiêu tốn năng lượng cực đại.

## 3. Cách triển khai (Curriculum Learning)
Các task không được dạy từ zero. Chúng tuân theo **Curriculum Learning (Học theo lộ trình)**:
1. **Balance:** Học đứng cân bằng không ngã.
2. **Walk Flat:** Lấy Checkpoint của Balance làm "Não ban đầu", chạy tiếp với mục tiêu $1.0$ m/s trên sàn phẳng.
3. **Walk Rough:** Đưa địa hình mấp mô vào dần.
4. **Run / Stair / Jump:** Base từ quá trình đi bộ ổn định để học các task cần động lượng (momentum) cao hơn.
