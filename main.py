import time
import math
import numpy as np

VARIANT = 21
N_FFT = 2 ** VARIANT       # N = 2^21 = 2,097,152
m = int(math.log2(N_FFT))  # m = 21


def compute_dft(x):
    N = len(x)
    C_k = np.zeros(N, dtype=complex)

    mult_count = 0
    add_count = 0

    start_time = time.perf_counter()
    inv_N = 1.0 / N

    for k in range(N):
        sum_real = 0.0
        sum_imag = 0.0

        for i in range(N):
            angle = -(2.0 * math.pi / N) * k * i
            wr = math.cos(angle)
            wi = math.sin(angle)
            
            xr_i = float(x[i].real)
            xi_i = float(x[i].imag)
            sum_real += xr_i * wr - xi_i * wi
            sum_imag += xr_i * wi + xi_i * wr
            mult_count += 1

            if i > 0:
                add_count += 1

        C_k[k] = complex(sum_real * inv_N, sum_imag * inv_N)

    execution_time = time.perf_counter() - start_time
    return C_k, execution_time, mult_count, add_count


def compute_fft_sato(x):
    n = len(x)
    stages = int(math.log2(n)) if n > 0 else 0
    if n <= 0 or (2 ** stages) != n:
        raise ValueError("Розмірність вибірки N має бути степенем двійки (2^m).")

    # Розділення вхідного сигналу на дійсну та уявну частини
    xr = [float(val.real) for val in x]
    xi = [float(val.imag) for val in x]

    mult_count = 0
    add_count = 0

    start_time = time.perf_counter()

    j = 0
    for i in range(n - 1):
        if i < j:
            xr[i], xr[j] = xr[j], xr[i]
            xi[i], xi[j] = xi[j], xi[i]
        k = n // 2
        while k <= j:
            j -= k
            k //= 2
        j += k

    for s in range(1, stages + 1):
        M = 2 ** s          # розмір поточного блоку
        L = M // 2          # півкрок

        for r in range(L):
            angle = -2.0 * math.pi * r / M
            wr = math.cos(angle)
            wi = math.sin(angle)

            for i in range(r, n, M):
                idx1 = i
                idx2 = i + L

                tr = wr * xr[idx2] - wi * xi[idx2]
                ti = wr * xi[idx2] + wi * xr[idx2]
                mult_count += 1
    
                xr[idx2] = xr[idx1] - tr
                xi[idx2] = xi[idx1] - ti
                xr[idx1] = xr[idx1] + tr
                xi[idx1] = xi[idx1] + ti
                add_count += 2

    inv_n = 1.0 / n
    for i in range(n):
        xr[i] *= inv_n
        xi[i] *= inv_n

    C_k = np.array([complex(xr[i], xi[i]) for i in range(n)])
    execution_time = time.perf_counter() - start_time
    return C_k, execution_time, mult_count, add_count


def main():
    np.random.seed(42)
    t = np.linspace(0, 1, N_FFT, endpoint=False)
    x_signal = (
        3.0 * np.cos(2 * np.pi * 50 * t)
        + 1.5 * np.sin(2 * np.pi * 120 * t)
        + 0.5 * np.random.randn(N_FFT)
    )

    N_sub = 1024
    x_sub = x_signal[:N_sub]

    c_dft_sub, t_dft_sub, m_dft_sub, a_dft_sub = compute_dft(x_sub)
    c_fft_sub, t_fft_sub, m_fft_sub, a_fft_sub = compute_fft_sato(x_sub)
    diff_sub = np.max(np.abs(c_dft_sub - c_fft_sub))

    print(f"1. Запуск обох методів на N = {N_sub} = 2^10:")
    print(f"   ДПФ: час = {t_dft_sub*1000:.2f} мс | множень = {m_dft_sub:,} | додавань = {a_dft_sub:,}")
    print(f"   ШПФ: час = {t_fft_sub*1000:.2f} мс | множень = {m_fft_sub:,} | додавань = {a_fft_sub:,}")
    print(f"   Максимальна різниця спектрів |C_ДПФ - C_ШПФ|: {diff_sub:.2e}\n")

    print(f"2. Повний розрахунок Сато на N = 2^{VARIANT} = {N_FFT:,}...")
    c_fft_21, t_fft_21, m_fft_21, a_fft_21 = compute_fft_sato(x_signal)

    c_ref_21 = np.fft.fft(x_signal) / N_FFT
    eps_max = np.max(np.abs(c_fft_21 - c_ref_21))

    m_dft_21 = N_FFT ** 2
    a_dft_21 = N_FFT * (N_FFT - 1)
    t_dft_21_est = t_dft_sub * ((N_FFT / N_sub) ** 2)

    speedup_mult = m_dft_21 / m_fft_21
    speedup_add = a_dft_21 / a_fft_21
    speedup_total = (m_dft_21 + a_dft_21) / (m_fft_21 + a_fft_21)
    speedup_time = t_dft_21_est / t_fft_21

    print("-" * 82)
    print(f"{'Показник (комплексні операції)':<32} | {'ДПФ':^22} | {'ШПФ':^22}")
    print("-" * 82)
    print(f"{'Кількість множень':<32} | {f'{m_dft_21:.3e}':^22} | {f'{m_fft_21:,}':^22}")
    print(f"{'Кількість додавань':<32} | {f'{a_dft_21:.3e}':^22} | {f'{a_fft_21:,}':^22}")
    print(f"{'Усього операцій':<32} | {f'{m_dft_21 + a_dft_21:.3e}':^22} | {f'{m_fft_21 + a_fft_21:,}':^22}")
    print(f"{'Час обчислення':<32} | {f'~{t_dft_21_est/3600:.1f} год (оцінка)':^22} | {f'{t_fft_21:.2f} с (реальний)':^22}")
    print("-" * 82)
    print(f"Виграш за кількістю множень:  у {speedup_mult:,.0f} разів")
    print(f"Виграш за кількістю додавань: у {speedup_add:,.0f} разів")
    print(f"Загальний виграш за діями:    у {speedup_total:,.0f} разів")
    print(f"Виграш за часом обчислення:   у {speedup_time:,.0f} разів")
    print(f"Максимальна похибка відносно np.fft: {eps_max:.2e}")


if __name__ == "__main__":
    main()