import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="SA-4 Setpoint Ramp & Rate Limit", page_icon="🏃")

st.title("🏃 SA-4 – Setpoint Ramp & Rate Limit Lab (Oda Isıtma)")
st.write(
    """
Bu laboratuvarda PID kontrollü oda ısıtma sisteminde,
**setpoint'i (hedef sıcaklığı) ne kadar hızlı değiştirdiğinin** sistem tepkisini
nasıl etkilediğini inceleyeceksin.

- Senaryo 1: Setpoint ani adım (step) şeklinde değişiyor  
- Senaryo 2: Setpoint, saniyede en fazla R °C hızla değişebiliyor (rate limit)
"""
)

st.markdown("---")


# -----------------------------
# Sistem parametreleri
# -----------------------------
st.subheader("1️⃣ Sistem Parametreleri")

col_sys1, col_sys2, col_sys3 = st.columns(3)

with col_sys1:
    T_ambient = st.slider(
        "Ortam sıcaklığı (°C)",
        0.0,
        30.0,
        20.0,
        1.0,
    )
with col_sys2:
    T_start = st.slider(
        "Başlangıç hedefi (°C) – setpoint_start",
        15.0,
        30.0,
        20.0,
        0.5,
    )
with col_sys3:
    T_target = st.slider(
        "Yeni hedef (°C) – setpoint_target",
        15.0,
        30.0,
        24.0,
        0.5,
    )

tau = st.slider(
    "Sistemin zaman sabiti τ (s)",
    10.0,
    200.0,
    60.0,
    10.0,
)
k_heat = st.slider(
    "Isıtıcı kazancı k_heat",
    0.1,
    2.0,
    0.5,
    0.1,
)

st.write(
    f"Sistem: ortam **{T_ambient:.1f}°C**, başlangıç hedefi **{T_start:.1f}°C**, "
    f"yeni hedef **{T_target:.1f}°C**, τ = **{tau:.0f} s**, k_heat = **{k_heat:.2f}**"
)


# -----------------------------
# PID parametreleri
# -----------------------------
st.subheader("2️⃣ PID Parametreleri")

col_pid1, col_pid2, col_pid3 = st.columns(3)
with col_pid1:
    Kp = st.slider("Kp", 0.0, 10.0, 3.0, 0.1)
with col_pid2:
    Ki = st.slider("Ki", 0.0, 1.0, 0.2, 0.01)
with col_pid3:
    Kd = st.slider("Kd", 0.0, 2.0, 0.0, 0.1)

st.write(f"PID: **Kp = {Kp:.2f}**, **Ki = {Ki:.2f}**, **Kd = {Kd:.2f}**")


# -----------------------------
# Simülasyon ayarları
# -----------------------------
st.subheader("3️⃣ Simülasyon Ayarları & Rate Limit")

col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    T_initial = st.slider(
        "Başlangıç sıcaklığı T₀ (°C)",
        0.0,
        30.0,
        20.0,
        0.5,
    )
with col_sim2:
    t_max = st.slider(
        "Toplam süre (s)",
        60.0,
        600.0,
        300.0,
        30.0,
    )

dt = st.slider(
    "Zaman adımı Δt (s)",
    0.1,
    5.0,
    1.0,
    0.1,
)

max_rate = st.slider(
    "Setpoint değişim hızı sınırı R (°C/s)",
    0.1,
    1.0,
    0.2,
    0.1,
    help="Rate limit aktifken setpoint saniyede en fazla R °C değişir.",
)

n_steps = int(t_max / dt) + 1
st.write(
    f"Simülasyon: {t_max:.0f} s, Δt = {dt:.1f} s, adım ≈ {n_steps}. "
    f"Rate limit: |d(setpoint)/dt| ≤ {max_rate:.2f} °C/s"
)


# -----------------------------
# Yardımcı fonksiyonlar
# -----------------------------
def simulate_room_with_sp_profile(
    T_ambient,
    tau,
    k_heat,
    T_initial,
    Kp,
    Ki,
    Kd,
    dt,
    n_steps,
    sp_profile,
):
    """Verilen setpoint profili ile PID'li oda simülasyonu."""
    t = np.zeros(n_steps)
    T = np.zeros(n_steps)
    u = np.zeros(n_steps)
    e = np.zeros(n_steps)

    T[0] = T_initial
    integral = 0.0
    prev_error = sp_profile[0] - T[0]

    for k in range(n_steps - 1):
        setpoint = sp_profile[k]
        error = setpoint - T[k]
        e[k] = error

        integral += error * dt
        derivative = (error - prev_error) / dt

        u_raw = Kp * error + Ki * integral + Kd * derivative
        u[k] = np.clip(u_raw, 0.0, 100.0)

        dTdt = -(T[k] - T_ambient) / tau + k_heat * (u[k] / 100.0)
        T[k + 1] = T[k] + dTdt * dt
        t[k + 1] = t[k] + dt

        prev_error = error

    e[-1] = sp_profile[-1] - T[-1]
    u[-1] = u[-2]

    return t, T, u, e


def create_step_profile(T_start, T_target, t, t_change=0.0):
    """t_change anında ani adım setpoint profili."""
    sp = np.full_like(t, T_start)
    sp[t >= t_change] = T_target
    return sp


def create_rate_limited_profile(T_start, T_target, t, max_rate):
    """
    Setpoint'i, saniyede en fazla max_rate °C değişecek şekilde
    T_start'tan T_target'a doğru yürüt.
    """
    sp = np.zeros_like(t)
    sp[0] = T_start
    for k in range(len(t) - 1):
        dt_local = t[k + 1] - t[k]
        diff = T_target - sp[k]
        max_change = max_rate * dt_local
        change = np.clip(diff, -max_change, max_change)
        sp[k + 1] = sp[k] + change
    return sp


# -----------------------------
# Profilleri ve simülasyonları oluştur
# -----------------------------
t_arr = np.linspace(0.0, t_max, n_steps)

sp_step = create_step_profile(T_start, T_target, t_arr, t_change=10.0)
sp_rate = create_rate_limited_profile(T_start, T_target, t_arr, max_rate)

t_step, T_step, u_step, e_step = simulate_room_with_sp_profile(
    T_ambient,
    tau,
    k_heat,
    T_initial,
    Kp,
    Ki,
    Kd,
    dt,
    n_steps,
    sp_step,
)

t_rate, T_rate, u_rate, e_rate = simulate_room_with_sp_profile(
    T_ambient,
    tau,
    k_heat,
    T_initial,
    Kp,
    Ki,
    Kd,
    dt,
    n_steps,
    sp_rate,
)


# -----------------------------
# Grafikleri çiz
# -----------------------------
st.markdown("---")
st.subheader("4️⃣ Sıcaklık ve Setpoint Karşılaştırması")

fig1, ax1 = plt.subplots(figsize=(7, 4))

ax1.plot(t_step, T_step, label="T(t) – Step SP")
ax1.plot(t_rate, T_rate, label="T(t) – Rate-limited SP")
ax1.plot(t_arr, sp_step, linestyle="--", label="Step SP")
ax1.plot(t_arr, sp_rate, linestyle=":", label="Rate-limited SP")

ax1.set_xlabel("t (s)")
ax1.set_ylabel("Sıcaklık (°C)")
ax1.set_title("Setpoint Step vs Rate-Limited – Sistem Tepkisi")
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax1.legend()

st.pyplot(fig1)

st.subheader("Kontrol Sinyalleri – u(t)")

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(t_step, u_step, label="u(t) – Step SP")
ax2.plot(t_rate, u_rate, label="u(t) – Rate-limited SP")
ax2.set_xlabel("t (s)")
ax2.set_ylabel("u(t) (%)")
ax2.set_title("PID Çıkışı – Ani vs Yumuşak Setpoint")
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax2.legend()

st.pyplot(fig2)


# -----------------------------
# İlk adımlar tablosu (rate-limited senaryo)
# -----------------------------
st.subheader("5️⃣ Rate-Limited Senaryonun İlk Adımları")

max_rows = min(20, n_steps)
df = pd.DataFrame(
    {
        "t (s)": t_rate[:max_rows],
        "Setpoint (rate)": sp_rate[:max_rows],
        "T(t)": T_rate[:max_rows],
        "u(t) (%)": u_rate[:max_rows],
    }
)

st.dataframe(
    df.style.format(
        {"t (s)": "{:.1f}", "Setpoint (rate)": "{:.2f}", "T(t)": "{:.2f}",
         "u(t) (%)": "{:.2f}"}
    )
)


# -----------------------------
# Öğretmen kutusu
# -----------------------------
st.markdown("---")
st.info(
    "Setpoint'i bir anda zıplatmak yerine, sınırlı hızla (rate limit) değiştirmek "
    "overshoot'u azaltabilir ve sistemi daha yumuşak hale getirebilir. "
    "Bu, özellikle proses endüstrisinde hassas ürünler için kritik olabilir."
)

with st.expander("👩‍🏫 Öğretmen Kutusu – Önerilen Sorular (SA-4)"):
    st.write(
        """
1. Aynı PID ayarlarıyla:

   - Step setpoint için maksimum overshoot (hedefin kaç °C üstüne çıktı?)  
   - Rate-limited setpoint için maksimum overshoot ne kadar?

2. Rate limit R'yi küçültürsen:

   - Setpoint daha yavaş mı yükseliyor?  
   - Overshoot nasıl değişiyor?  
   - Hedefe ulaşma süresi hakkında ne söyleyebilirsin?

3. Gerçek dünyada:

   - Hangi süreçlerde "yumuşak hedef değişimi" (ramp veya rate limit) daha iyi olabilir?  
     (Fırınlar, kimya reaktörleri, motor hız kontrolü vb.)
"""
    )

st.caption("SA-4: Setpoint ramp / rate limit ile PID davranışını sezgisel olarak inceleme.")
