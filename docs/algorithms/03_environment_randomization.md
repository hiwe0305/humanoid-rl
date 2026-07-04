# Xây Dựng Môi Trường với Tính Chất Vật Lý và Ngoại Lực (Domain Randomization)

Để robot học được sự cân bằng thực sự và có khả năng chống chịu với sự sai lệch trong môi trường thực tế (Sim-to-Real), hệ thống triển khai hai khái niệm: **Domain Randomization (Ngẫu nhiên hóa Vật lý)** và **Push Perturbation (Ngoại lực đẩy ngẫu nhiên)**.

## 1. Domain Randomization (Ngẫu nhiên hóa Thông số Vật lý)

Thuật toán thay đổi các thông số vật lý của robot (mass, friction, damping, gear) một cách ngẫu nhiên trước mỗi Episode. Điều này ép Policy không được phép "ghi nhớ" một cấu hình vật lý cố định, mà phải học ra quy luật phản hồi chung (robust control).

### Các thông số được điều chỉnh:
- **Mass (Khối lượng):** Thay đổi khối lượng của các bộ phận cơ thể.
  $$ M_{new} = M_{base} \times (1.0 + \text{uniform}(-0.15, 0.15)) $$
  (Dao động $\pm 15\%$ khối lượng)
  
- **Friction (Ma sát bề mặt):** Sự trơn trượt của mặt sàn và bàn chân.
  $$ F_{new} = F_{base} \times (1.0 + \text{uniform}(-0.25, 0.25)) $$
  (Dao động $\pm 25\%$ độ ma sát)

- **Joint Damping (Lực cản/Ma sát khớp):** Mô phỏng sự sai khác về ma sát của hệ thống bánh răng/motor bên trong.
  $$ D_{new} = D_{base} \times (1.0 + \text{uniform}(-0.15, 0.15)) $$

- **Actuator Gear (Lực đầu ra của Motor):**
  $$ G_{new} = G_{base} \times (1.0 + \text{uniform}(-0.10, 0.10)) $$

**Cách Triển Khai (trong `domain_rand.py`):**
Cơ chế này sử dụng API của MuJoCo: `model.body_mass`, `model.geom_friction`, `model.dof_damping`. Các giá trị bị can thiệp trực tiếp trước mỗi chu kỳ Reset môi trường (trong hàm `env.reset()`).

## 2. Push Perturbation (Tạo Lực Đẩy Ngoại Cảnh)

Môi trường sẽ ngẫu nhiên tác động các lực vô hình (như có người xô đẩy) vào thân của Robot (`torso`). Mục tiêu là mô phỏng hiện tượng va chạm bất ngờ và xem robot có khôi phục thăng bằng được không.

### Cơ Chế Toán và Vật Lý:
Mỗi chu kỳ nhỏ (step), sẽ có một xác suất $P_{push} = 0.005$ (0.5%) để xuất hiện ngoại lực. Một ngoại lực sẽ kéo dài trong vòng vài mô phỏng nhỏ (`duration = 5`).

- **Độ lớn ngoại lực (Force Magnitude):** Chọn ngẫu nhiên trong khoảng $[0.5 \times F_{max}, F_{max}]$ (với $F_{max} = 80N$).
- **Góc đẩy (Force Angle):** Tạo một góc ngẫu nhiên $\theta \in [0, 2\pi]$ radian.

Thành phần lực đẩy ($F_x, F_y$):
$$ F_x = \text{Magnitude} \times \cos(\theta) $$
$$ F_y = \text{Magnitude} \times \sin(\theta) $$

Lực này được đưa trực tiếp vào `data.xfrc_applied` của `torso_link`:
```python
# Cấp ngoại lực vào ma trận xfrc_applied (lực 6 trục: [torque(3), force(3)])
data.xfrc_applied[torso_id, 3] = F_x
data.xfrc_applied[torso_id, 4] = F_y
```

Điều này tương đương với phương trình định luật 2 Newton $F = m \cdot a$. Do lực đẩy áp dụng vào phần thân trên (cao), nó sẽ sinh ra một mô-men lật (torque) lớn đối với cổ chân (vị trí bản lề), tạo thách thức lớn cho việc bám đất của robot.

---
**Tổng kết:** Khi RL học trong môi trường có đủ Domain Randomization và Push Perturbation, nó không chỉ thuộc lòng cách điều khiển bước đi trong một phòng thí nghiệm chân không hoàn hảo, mà tạo ra một bộ điều khiển "cứng cỏi" sẵn sàng cho thế giới thực.
