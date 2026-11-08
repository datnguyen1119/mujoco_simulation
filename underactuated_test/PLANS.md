---
title: "Kế hoạch 2 tuần – Cart-pole, Acrobot, mở đầu đi bộ"
course: "MIT 6.832 - Underactuated Robotics"
source: "https://underactuated.csail.mit.edu"
tags:
  - underactuated-robotics
  - ke-hoach
  - LQR
  - trajectory-optimization
  - lyapunov
created: 2026-11-03
---

# Kế hoạch 2 tuần

> [!info] Phạm vi
> Lộ trình đầy đủ có 6 bước (cart-pole → acrobot → compass gait → biped 2D → humanoid 3D → robot thật), cộng lại mất **vài tháng**.
> Trong **2 tuần**, mục tiêu là **làm chắc bước 1 và bước 2**, rồi **mở đầu bước 3**.

> [!note] Giả định
> - Khoảng **2–3 giờ mỗi ngày**.
> - Công cụ chính: **Drake** (`DirectCollocation`, `LinearQuadraticRegulator`, `FiniteHorizonLinearQuadraticRegulator`, notebook Colab của sách).
> - **MuJoCo** là tùy chọn, để kiểm tra lại kết quả trên bộ mô phỏng khác.

## Lộ trình tổng thể (để định hướng)

| Bước | Dự án | Kỹ năng mới | Trong 2 tuần? |
|---|---|---|---|
| 1 | Cart-pole: swing-up bằng `DirectCollocation`, bám bằng TVLQR, giữ đỉnh bằng LQR | Ghép đủ chuỗi ch. 8 + 9 + 10 | ✅ Tuần 1 |
| 2 | Acrobot: cùng quy trình | Hệ thiếu chấp hành khó hơn | ✅ Tuần 2 |
| 3 | Compass gait / rimless wheel (ch. 4) | Chu kỳ giới hạn, Poincaré map, va chạm | 🔸 Mở đầu |
| 4 | Biped phẳng 2D (5–7 khớp), ZMP hoặc centroidal MPC | Bước chân, chuyển pha tiếp xúc (ch. 5) | Sau |
| 5 | Humanoid 3D trong mô phỏng | Whole-body control | Sau |
| 6 | Robot thật (nhỏ, rẻ) | Ước lượng trạng thái, sim-to-real | Sau |

---

## Tuần 1 – Cart-pole: đủ chuỗi chương 8 → 9 → 10

> [!goal] Mục tiêu cuối tuần
> Cart-pole tự swing-up từ vị trí buông thõng, có **giới hạn ray** và **giới hạn lực**, rồi giữ thăng bằng ở đỉnh, **kể cả khi mô hình bị sai**.

### Ngày 1 – Cài đặt và mô hình
- [ ] Cài Drake (hoặc mở notebook Colab của sách), cài MuJoCo (tùy chọn)
- [ ] Mở notebook cart-pole của sách, đọc code
- [ ] Tự viết phương trình chuyển động cart-pole (ch. 3)
- [ ] So với `MultibodyPlant`: mô phỏng thả rơi tự do

**Kiến thức:** Phụ lục B, ch. 3
**Kiểm tra:** mô phỏng khớp với phương trình tự viết

### Ngày 2 – LQR giữ đỉnh
- [ ] `LinearQuadraticRegulator(plant, context, Q, R)` quanh vị trí thẳng đứng
- [ ] Đẩy thử (thêm lực ngoài), xem có giữ được không
- [ ] Đổi Q, R; ghi lại hành vi thay đổi thế nào

**Kiến thức:** 8.1
**Kiểm tra:** cart-pole đứng vững; ghi chú ảnh hưởng của Q, R

### Ngày 3 – Vùng hút bằng thực nghiệm
- [ ] Thả cart-pole ở lưới góc lệch (và vận tốc) quanh đỉnh
- [ ] Ghi lại điểm nào về được, điểm nào ngã
- [ ] Vẽ đường mức $\bar x^TS\bar x$ chồng lên
- [ ] Chọn ρ lớn nhất sao cho tập mức $\{\bar x^TS\bar x \le \rho\}$ nằm trong vùng về được

**Kiến thức:** 9.1.4
**Kiểm tra:** hình "vùng hút thật" so với tập mức ellipse; giá trị ρ

### Ngày 4 – Swing-up năng lượng + chuyển sang LQR
- [ ] Cài luật $u = k\,\dot\theta\cos\theta\,\tilde E$ (qua PFL điều khiển gia tốc cart)
- [ ] Chuyển sang LQR khi $\bar x^TS\bar x \le \rho$ (ρ của ngày 3)
- [ ] Thử từ vị trí buông thõng (nhớ đẩy nhẹ ban đầu)

**Kiến thức:** 9.1.2 (LaSalle), ch. 3
**Kiểm tra:** swing-up thành công, chuyển mượt sang LQR

### Ngày 5 – Swing-up tối ưu bằng collocation
- [ ] `DirectCollocation` với giới hạn lực và **giới hạn ray**
- [ ] Khởi tạo trạng thái bằng đường thẳng từ buông thõng tới đỉnh
- [ ] Thử thêm / bỏ cost thời gian
- [ ] So với swing-up năng lượng: thời gian, lực, độ dài ray dùng

**Kiến thức:** 10.3.2
**Kiểm tra:** quỹ đạo tối ưu + bảng so sánh

### Ngày 6 – TVLQR bám quỹ đạo
- [ ] `FiniteHorizonLinearQuadraticRegulator` dọc quỹ đạo ngày 5, đặt $S(t_f) = S_\infty$
- [ ] Chạy **vòng hở** (chỉ $u^*(t)$) và **vòng kín** (TVLQR)
- [ ] Làm sai mô hình: khối lượng ±20 %, thêm nhiễu cảm biến

**Kiến thức:** 8.2, 10.5
**Kiểm tra:** đồ thị vòng hở trôi, vòng kín bám được

### Ngày 7 – Ghép và tổng kết (ngày đệm)
- [ ] Ghép: collocation → TVLQR → LQR thành một demo
- [ ] (Tùy chọn) MPC tuyến tính quanh đỉnh có giới hạn lực
- [ ] Viết ghi chú: cái gì chạy, cái gì hỏng, vì sao
- [ ] Bắt kịp phần còn thiếu nếu chậm tiến độ

**Kiểm tra:** demo hoàn chỉnh + ghi chú tuần 1

---

## Tuần 2 – Acrobot, rồi mở đầu đi bộ

> [!goal] Mục tiêu cuối tuần
> Lặp lại quy trình trên một hệ **khó hơn**, và hiểu cách phân tích ổn định cho **chuyển động lặp** (Poincaré map).

### Ngày 8 – Acrobot: LQR ở đỉnh
- [ ] Mô hình acrobot, tuyến tính hóa ở đỉnh
- [ ] Kiểm tra tính điều khiển được
- [ ] LQR; so kích thước vùng hút với cart-pole

**Kiến thức:** 8.1, ch. 3
**Kiểm tra:** acrobot đứng vững; nhận xét vì sao vùng hút nhỏ hơn

### Ngày 9 – Swing-up bằng PFL + năng lượng
- [ ] Partial feedback linearization (collocated hoặc non-collocated)
- [ ] Luật năng lượng của Spong
- [ ] Chuyển sang LQR ở đỉnh

**Kiến thức:** ch. 3
**Kiểm tra:** swing-up thành công, hoặc ghi rõ chỗ thất bại

### Ngày 10 – Swing-up bằng collocation
- [ ] `DirectCollocation` cho acrobot, giới hạn mô-men
- [ ] Thử **có** và **không có** khởi tạo đường thẳng

**Kiến thức:** 10.3.2
**Kiểm tra:** thấy rõ vai trò của khởi tạo

### Ngày 11 – TVLQR + LQR cho acrobot
- [ ] TVLQR dọc quỹ đạo ngày 10, chuyển sang LQR ở đỉnh
- [ ] Thử độ bền: sai tham số, nhiễu

**Kiến thức:** 8.2, 9
**Kiểm tra:** demo acrobot hoàn chỉnh; so độ bền với cart-pole

### Ngày 12 – Rimless wheel
- [ ] Mô phỏng lăn bị động xuống dốc, có va chạm
- [ ] Tính **Poincaré map** bằng số (ánh xạ bước-sang-bước)
- [ ] Tìm điểm bất động

**Kiến thức:** ch. 4
**Kiểm tra:** đồ thị ánh xạ bước-sang-bước và điểm bất động

### Ngày 13 – Compass gait
- [ ] Mô phỏng compass gait đi bị động xuống dốc
- [ ] Tìm **chu kỳ giới hạn** = điểm bất động của Poincaré map
- [ ] Xét ổn định: trị riêng của $\partial P/\partial x$ có độ lớn < 1?

**Kiến thức:** ch. 4
**Kiểm tra:** dáng đi bị động ổn định; giá trị các trị riêng

### Ngày 14 – Ôn tập và chuẩn bị (ngày đệm)
- [ ] Viết ghi chú tổng kết 2 tuần
- [ ] Liệt kê điểm còn yếu
- [ ] Lên kế hoạch cho bước 3 đầy đủ (compass gait có động cơ, ch. 4 và 16) và bước 4 (biped 2D, ch. 5)

**Kiểm tra:** bản tổng kết + kế hoạch tiếp theo

---

## Nguyên tắc làm việc

> [!tip] Để 2 tuần hiệu quả
> - **Mỗi ngày phải ra một hình hoặc một con số.** Không ra được thì ghi lại vì sao; đó cũng là kết quả.
> - **Luôn làm sai mô hình một chút** (khối lượng, ma sát, nhiễu, độ trễ). Bộ điều khiển chỉ chạy trên mô hình hoàn hảo là chưa xong.
> - **Đọc code notebook của sách trước khi tự viết.**
> - **Kẹt quá 1 ngày ở một việc thì đi tiếp**, ghi lại để quay về vào ngày 7 hoặc 14.
> - **Ngày 7 và 14 là ngày đệm** để bắt kịp tiến độ.

## Nhật ký tiến độ

| Ngày | Hoàn thành | Kết quả chính | Vấn đề gặp phải |
|---|---|---|---|
| 1 | ☐ | | |
| 2 | ☐ | | |
| 3 | ☐ | | |
| 4 | ☐ | | |
| 5 | ☐ | | |
| 6 | ☐ | | |
| 7 | ☐ | | |
| 8 | ☐ | | |
| 9 | ☐ | | |
| 10 | ☐ | | |
| 11 | ☐ | | |
| 12 | ☐ | | |
| 13 | ☐ | | |
| 14 | ☐ | | |

## Liên kết
- [[Chuong 8 - LQR (8.2 tro di)]]
- [[Chương 9 - Lyapunov Analysis]]
- [[Chương 10 - Trajectory Optimization]]
- Sách: <https://underactuated.csail.mit.edu>