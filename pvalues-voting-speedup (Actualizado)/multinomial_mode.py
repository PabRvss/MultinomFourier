import numpy as np
import math
import heapq


# funciona
def multinomial_mode(n, p):
    """
    Calcula la moda de una distribución multinomial.
    
    Args:
    - n (int): Número total de ensayos.
    - p (numpy array): Array de probabilidades de las categorías.
    
    Returns:
    - numpy array: Vector de frecuencias x_i que maximiza la probabilidad.
    """
    k = len(p)  # Número de categorías
    x = np.zeros(k, dtype=int)  # Inicializar x_i en 0
    remaining = n  # Ensayos por asignar

    # Ordenar índices de p en orden descendente
    sorted_indices = np.argsort(-p)  # Índices ordenados por probabilidad descendente

    # Asignar ensayos según las probabilidades ordenadas
    for i in sorted_indices:
        x[i] += min(remaining, round(n * p[i]))  # Aproximación inicial
        remaining = n - np.sum(x)  # Recalcular los ensayos restantes
        if remaining <= 0:
            break

    # Distribuir ensayos restantes (si no fueron distribuidos exactamente)
    i = 0
    while remaining > 0:
        x[sorted_indices[i]] += 1
        remaining -= 1
        i = (i + 1) % k  # Ciclar por los índices

    return x


# no da el resultado correcto
def greedy_mode_find(n, p):
    """
    Implementación del algoritmo GreedyModeFind para encontrar la moda de una distribución multinomial.

    Args:
    - n (int): Número total de ensayos.
    - p (numpy array): Array de probabilidades de las categorías.

    Returns:
    - numpy array: Vector de frecuencias x_i que maximiza la probabilidad.
    """
    r = len(p)  # Número de categorías
    x = np.zeros(r, dtype=int)  # Inicializar x_i en 0
    remaining = n  # Ensayos por asignar

    # Ordenar índices de p en orden descendente
    sorted_indices = np.argsort(-p)  # Índices ordenados por probabilidad descendente

    # Asignar ensayos según las probabilidades ordenadas
    for i in sorted_indices:
        x[i] = int(np.floor((p[i] * (n + r - 1)) - 0.5))
        remaining -= x[i]

    # Ajustar para asegurar que la suma de x sea n
    while remaining != 0:
        for i in sorted_indices:
            if remaining == 0:
                break
            if remaining > 0:
                x[i] += 1
                remaining -= 1
            elif remaining < 0 and x[i] > 0:
                x[i] -= 1
                remaining += 1

    return x


# no funciona, incompleto
def greedy_mode_find_v2(n, p):
    """
    Implementación del algoritmo GreedyModeFind para encontrar la moda de una distribución multinomial.

    Args:
    - n (int): Número total de ensayos.
    - p (numpy array): Array de probabilidades de las categorías.

    Returns:
    - numpy array: Vector de frecuencias x_i que maximiza la probabilidad.
    """
    m = 0
    # get p sorted and the respective indexes
    # sorted_p = np.sort(p)
    sorted_indices = np.argsort(p)

    r = len(p)  # Número de categorías
    x = np.zeros(r, dtype=int)  # Inicializar x_i en 0
    # for i in range(r):
    for i in sorted_indices:
        x[i] = int(np.floor(p[i] * (n + r / 2)))
        m += x[i]
    if m != n:
        for i in sorted_indices:
            if m == n:
                break
            if m < n:
                x[i] += 1
                m += 1
            elif m > n and x[i] > 0:
                x[i] -= 1
                m -= 1
    # x = np.zeros(r, dtype=int)  # Inicializar x_i en 0
    # remaining = n  # Ensayos por asignar

    # Ordenar índices de p en orden descendente
    sorted_indices = np.argsort(-p)  # Índices ordenados por probabilidad descendente

    # Asignar ensayos según las probabilidades ordenadas
    for i in sorted_indices:
        x[i] = int(np.floor((p[i] * (n + r - 1)) - 0.5))
        remaining -= x[i]

    # Ajustar para asegurar que la suma de x sea n
    while remaining != 0:
        for i in sorted_indices:
            if remaining == 0:
                break
            if remaining > 0:
                x[i] += 1
                remaining -= 1
            elif remaining < 0 and x[i] > 0:
                x[i] -= 1
                remaining += 1

    return x


# funciona
def greedy_mode_find1(n, p):
    """
    Implementación del algoritmo GreedyModeFind1 para encontrar la moda de una distribución multinomial.
    
    Args:
    - n (int): Número total de ensayos.
    - p (numpy array): Array de probabilidades de las categorías.
    
    Returns:
    - numpy array: Vector de frecuencias x que maximiza la probabilidad.
    """
    r = len(p)  # Número de categorías
    x = np.zeros(r, dtype=int)  # Inicializar el vector x
    m = 0  # Contador para la suma total de elementos en x

    # Paso 1: Asignación inicial
    for i in range(r):
        x[i] = int(np.floor(p[i] * (n + r / 2)))  # Asignación inicial
        m += x[i]

    # Si la suma inicial coincide con n, listo
    if m == n:
        return x
        
    # Paso 2: Construcción del heap
    h = []  # Heap vacío
    for i in range(r):
        if m < n:  # Caso de déficit
            score = p[i] / (x[i] + 1)
        else:  # Caso de exceso
            score = x[i] / p[i] if x[i] > 0 else -np.inf
        heapq.heappush(h, (-score, i))  # Heap con valores negativos para max-heap

    # Paso 3: Ajuste iterativo
    while m != n:
        best_score, best_elem = heapq.heappop(h)  # Extraer el mejor elemento
        best_elem = int(best_elem)

        if m < n:  # Incrementar
            x[best_elem] += 1
            m += 1
            new_score = p[best_elem] / (x[best_elem] + 1)
        else:  # Decrementar
            x[best_elem] -= 1
            m -= 1
            new_score = x[best_elem] / p[best_elem] if x[best_elem] > 0 else -np.inf

        # Insertar el elemento ajustado nuevamente en el heap
        heapq.heappush(h, (-new_score, best_elem))

    return x


# pmf of the multinomial distribution
def multinomial_pmf(x, n, p):
    """
    Calcula la probabilidad de una distribución multinomial.

    Args:
    - x (numpy array): Vector de frecuencias x_i.
    - n (int): Número total de ensayos.
    - p (numpy array): Array de probabilidades de las categorías.

    Returns:
    - float: Probabilidad de la distribución multinomial.
    """
    return math.factorial(n) / np.prod([math.factorial(x_i) for x_i in x]) * np.prod([p_i ** x_i for x_i, p_i in zip(x, p)])


if __name__ == '__main__':
    # Ejemplo de uso
    n = 10
    p = np.array([0.4, 0.3, 0.2, 0.1])
    mode = greedy_mode_find(n, p)
    print("Moda:", mode)
    print("pmf = ", multinomial_pmf(mode, n, p))

    # Ejemplo de uso
    n = 10
    p = np.array([0.4, 0.3, 0.2, 0.1])
    mode = multinomial_mode(n, p)
    print("pmf = ", multinomial_pmf(mode, n, p))

    # Ejemplo de uso
    n = 10
    p = np.array([0.4, 0.1, 0.2, 0.2, 0.1])
    mode = greedy_mode_find1(n, p)
    print("Moda:", mode)
    print("pmf = ", multinomial_pmf(mode, n, p))
