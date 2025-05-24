#Выполнил Шакула Дмитрий Андреевич 090301-ПОВа-о24

import numpy as np
import time
from scipy.linalg.blas import zgemm

N_SMALL = 200
N_LARGE = 2048

# Инициализация матриц
np.random.seed(42)
A_large = np.random.rand(N_LARGE, N_LARGE).astype(np.complex128) + 1j * np.random.rand(N_LARGE, N_LARGE).astype(np.complex128)
B_large = np.random.rand(N_LARGE, N_LARGE).astype(np.complex128) + 1j * np.random.rand(N_LARGE, N_LARGE).astype(np.complex128)
A_small = np.random.rand(N_SMALL, N_SMALL).astype(np.complex128) + 1j * np.random.rand(N_SMALL, N_SMALL).astype(np.complex128)
B_small = np.random.rand(N_SMALL, N_SMALL).astype(np.complex128) + 1j * np.random.rand(N_SMALL, N_SMALL).astype(np.complex128)

# Вычисление сложности
complexity_small = 2 * N_SMALL**3
complexity_large = 2 * N_LARGE**3

# Функция для измерения производительности
def measure_performance(func, *args, iterations=3):
    elapsed_times = []
    result = None
    for _ in range(iterations):
        start_time = time.time()
        result = func(*args)
        elapsed_times.append(time.time() - start_time)
    elapsed_time = min(elapsed_times)  # Берём лучшее время
    mflops = complexity_small / (elapsed_time * 1e6) if args[0].shape[0] == N_SMALL else complexity_large / (elapsed_time * 1e6)
    return result, elapsed_time, mflops

# Стандартное перемножение (IJK)
def matrix_multiply_formula(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# BLAS перемножение (cgemm)
def matrix_multiply_blas(A, B):
    return zgemm(alpha=1.0, a=A, b=B)

# Оптимизированное блочное перемножение
def matrix_multiply_optimized(A, B, block_size=512, use_zgemm=True):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.complex128)
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            for k in range(0, n, block_size):
                k_end = min(k + block_size, n)
                if use_zgemm:
                    C[i:i_end, j:j_end] += zgemm(
                        alpha=1.0,
                        a=A[i:i_end, k:k_end],
                        b=B[k:k_end, j:j_end]
                    )
                else:
                    C[i:i_end, j:j_end] += np.dot(
                        A[i:i_end, k:k_end],
                        B[k:k_end, j:j_end]
                    )
    return C

def main():
    print("Работу выполнил: Шакула Дмитрий Андреевич 090301-ПОВа-o24")
    print("\n1-Й ВАРИАНТ: УМНОЖЕНИЕ ПО ФОРМУЛЕ ИЗ ЛИНЕЙНОЙ АЛГЕБРЫ")
    print(f"Размер матрицы: {N_SMALL}x{N_SMALL}")
    C_formula, time_formula, mflops_formula = measure_performance(matrix_multiply_formula, A_small, B_small)
    print(f"Время выполнения: {time_formula:.2f} секунд")
    print(f"Производительность: {mflops_formula:.2f} MFLOPS")

    print("\n\n2-Й ВАРИАНТ: ИСПОЛЬЗОВАНИЕ CBALS_CGEMM ИЗ БИБЛИОТЕКИ BLAS")
    print(f"Размер матрицы: {N_LARGE}x{N_LARGE}")
    C_blas, time_blas, mflops_blas = measure_performance(matrix_multiply_blas, A_large, B_large)
    print(f"Время выполнения: {time_blas:.2f} секунд")
    print(f"Производительность: {mflops_blas:.2f} MFLOPS")


    print("\n\n3-Й ВАРИАНТ: ОПТИМИЗИРОВАННЫЙ АЛГОРИТМ")
    print(f"Размер матрицы: {N_LARGE}x{N_LARGE}")
    C_optimized, time_optimized, mflops_optimized = measure_performance(
        lambda x, y: matrix_multiply_optimized(x, y, block_size=512, use_zgemm=True), A_large, B_large
    )
    print(f"Время выполнения: {time_optimized:.2f} секунд")
    print(f"Производительность: {mflops_optimized:.2f} MFLOPS")


    print("\n\nСРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ:")
    print(f"1-й вариант (размер {N_SMALL}x{N_SMALL}): {mflops_formula:.2f} MFLOPS")
    print(f"2-й вариант (размер {N_LARGE}x{N_LARGE}): {mflops_blas:.2f} MFLOPS")
    print(f"3-й вариант (размер {N_LARGE}x{N_LARGE}): {mflops_optimized:.2f} MFLOPS")
    performance_ratio = mflops_optimized / mflops_blas
    print(f"Отношение производительности (3-й / 2-й): {performance_ratio:.2f}")
    print(f"Условие выполнено (Оптимизированное >= 30% от BLAS): {'Да' if performance_ratio >= 0.3 else 'Нет'}")

    # Проверка корректности
    norm12 = np.sqrt(np.sum(np.abs(C_formula - matrix_multiply_blas(A_small, B_small))**2))
    norm23 = np.sqrt(np.sum(np.abs(C_blas - C_optimized)**2))
    print("\nПроверка корректности (нормы разностей):")
    print(f"Norm(C1 - C2): {norm12:.2e}")
    print(f"Norm(C2 - C3): {norm23:.2e}")

if __name__ == "__main__":
    main()
