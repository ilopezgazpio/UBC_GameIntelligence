def tabu_search_with_advanced_memory(initial_solution, max_iterations, tabu_tenure_short,long_term_memory_threshold, intensification_factor):
    """
    Implementación de Búsqueda Tabú con memoria a corto, medio y largo plazo,
    y criterio de aspiración.

    Args:
        initial_solution: La solución de partida para el algoritmo.
        max_iterations: Número máximo de iteraciones a ejecutar.
        tabu_tenure_short: Duración de los movimientos en la lista tabú a corto plazo.
        long_term_memory_threshold: Frecuencia a partir de la cual se penalizan los atributos
                                    para diversificación.
        intensification_factor: Factor para enfocar la búsqueda en soluciones élite.

    Returns:
        La mejor solución encontrada durante la búsqueda.
    """

    # --- MEMORIA A CORTO PLAZO (Tabu List) ---
    # Almacena movimientos (atributos de movimientos) que no deben repetirse por un tiempo.
    tabu_list = [] # Lista de tuplas: (atributo_movimiento, iteracion_expiracion)

    # --- MEMORIA A MEDIO PLAZO (Intensificación) ---
    # Almacena los atributos de las mejores soluciones encontradas.
    # Se usa para guiar la búsqueda hacia regiones prometedoras.
    elite_solutions_attributes = []

    # --- MEMORIA A LARGO PLAZO (Diversificación) ---
    # Almacena la frecuencia de uso de diferentes atributos o regiones del espacio de búsqueda.
    # Se usa para forzar la exploración de nuevas áreas.
    attribute_frequency = {} # Diccionario: {atributo: contador_de_frecuencia}


    # 1. INICIALIZACIÓN
    current_solution = initial_solution
    best_solution_found = current_solution
    best_cost_found = calculate_cost(current_solution)
    current_iteration = 0

    print(f"Inicio de Búsqueda Tabú. Coste inicial: {best_cost_found}")

    # 2. BUCLE PRINCIPAL
    while current_iteration < max_iterations:
        neighborhood = generate_neighbors(current_solution)
        best_neighbor_this_iteration = None
        best_neighbor_cost_this_iteration = float('inf')

        # Para aplicar el criterio de aspiración, necesitamos recordar el mejor movimiento tabú
        best_tabu_move_candidate = None
        best_tabu_move_cost_candidate = float('inf')

        # 3. EXPLORAR VECINDARIO
        for neighbor in neighborhood:
            neighbor_cost = calculate_cost(neighbor)
            move_to_neighbor = identify_move(current_solution, neighbor) # Define el "movimiento" (ej. intercambio de elementos)

            # Actualizar memoria a largo plazo (diversificación)
            update_long_term_memory(attribute_frequency, move_to_neighbor)

            # Es el movimiento hacia este vecino un movimiento TABÚ?
            is_tabu = check_if_tabu(move_to_neighbor, tabu_list, current_iteration)

            if not is_tabu:
                # Si no es tabú, lo consideramos para el mejor vecino de esta iteración
                if neighbor_cost < best_neighbor_cost_this_iteration:
                    best_neighbor_cost_this_iteration = neighbor_cost
                    best_neighbor_this_iteration = neighbor
            else:
                # 4. CRITERIO DE ASPIRACIÓN: ¿Podemos ignorar el estado tabú?
                # Sí, si el movimiento tabú lleva a una solución MEJOR que la MEJOR global encontrada hasta ahora.
                if neighbor_cost < best_cost_found:
                    if neighbor_cost < best_tabu_move_cost_candidate:
                        best_tabu_move_cost_candidate = neighbor_cost
                        best_tabu_move_candidate = neighbor
                    # Consideramos este movimiento tabú aspiracional como el mejor candidato si supera al no-tabú
                    if neighbor_cost < best_neighbor_cost_this_iteration:
                        best_neighbor_cost_this_iteration = neighbor_cost
                        best_neighbor_this_iteration = neighbor


        # 5. ELEGIR EL MEJOR MOVIMIENTO PARA LA ITERACIÓN
        # Si un movimiento aspiracional fue el mejor candidato, lo elegimos.
        # De lo contrario, elegimos el mejor movimiento no tabú encontrado.
        if best_neighbor_this_iteration is None:
            # Esto puede ocurrir si todos los movimientos son tabú y ninguno cumple la aspiración
            # En problemas complejos, se podría implementar una estrategia de "escape" o reiniciar.
            print("Advertencia: No se pudo encontrar un movimiento válido. Terminando o reiniciando.")
            break # O implementa una estrategia de reinicio/diversificación forzada

        # Actualizar la solución actual
        old_solution = current_solution
        current_solution = best_neighbor_this_iteration
        current_cost = best_neighbor_cost_this_iteration

        # 6. ACTUALIZAR LISTA TABÚ (Memoria a Corto Plazo)
        move_made = identify_move(old_solution, current_solution)
        add_to_tabu_list(tabu_list, move_made, current_iteration + tabu_tenure_short)
        remove_expired_tabu_moves(tabu_list, current_iteration)

        # 7. ACTUALIZAR LA MEJOR SOLUCIÓN ENCONTRADA Y MEMORIA A MEDIO PLAZO (Intensificación)
        if current_cost < best_cost_found:
            best_cost_found = current_cost
            best_solution_found = current_solution
            # Añadir atributos de esta nueva solución élite a la memoria a medio plazo
            add_to_elite_solutions(elite_solutions_attributes, current_solution)
            print(f"Nueva mejor solución encontrada en iteración {current_iteration}: Coste = {best_cost_found}")


        # 8. ESTRATEGIAS DE DIVERSIFICACIÓN E INTENSIFICACIÓN (Basadas en Memoria a Largo y Medio Plazo)
        # Esto es donde la "inteligencia" avanzada entra en juego.
        if current_iteration % 100 == 0 and current_iteration > 0: # Ejemplo: cada 100 iteraciones
            if should_diversify(attribute_frequency, long_term_memory_threshold):
                # Aplicar una estrategia de diversificación (ej. saltar a una solución generada aleatoriamente
                # que evite atributos muy frecuentes, o relajar restricciones tabú temporalmente).
                print(f"Diversificando en iteración {current_iteration} debido a la memoria a largo plazo.")
                current_solution = generate_diverse_solution(attribute_frequency)
                current_cost = calculate_cost(current_solution)
                if current_cost < best_cost_found: # La nueva solución diversa podría ser mejor
                    best_cost_found = current_cost
                    best_solution_found = current_solution

            elif should_intensify(elite_solutions_attributes, intensification_factor):
                # Aplicar una estrategia de intensificación (ej. hacer una búsqueda más exhaustiva
                # en el vecindario de una de las soluciones élite).
                print(f"Intensificando en iteración {current_iteration} alrededor de soluciones élite.")
                current_solution = intensify_search_around_elite(elite_solutions_attributes, current_solution)
                current_cost = calculate_cost(current_solution)
                if current_cost < best_cost_found:
                    best_cost_found = current_cost
                    best_solution_found = current_solution

        current_iteration += 1

    print(f"\nBúsqueda Tabú finalizada. Mejor coste encontrado: {best_cost_found}")
    return best_solution_found


# --- FUNCIONES AUXILIARES (Estas serían implementaciones específicas para tu problema) ---

def calculate_cost(solution):
    """Calcula el coste de una solución. (Ej. distancia total, tiempo, etc.)"""
    # Implementación específica del problema
    return len(solution) # Ejemplo trivial

def generate_neighbors(solution):
    """Genera el conjunto de soluciones vecinas a una solución dada."""
    # Implementación específica del problema (ej. intercambiar dos elementos, cambiar un valor)
    return [solution] # Ejemplo trivial

def identify_move(old_solution, new_solution):
    """Identifica el 'movimiento' que transformó old_solution en new_solution.
       Esto es lo que se guarda en la lista tabú."""
    # Implementación específica del problema (ej. "intercambio_de_posiciones_x_y")
    return "some_move_attribute" # Ejemplo trivial

def check_if_tabu(move_attribute, tabu_list, current_iteration):
    """Verifica si un movimiento está actualmente en la lista tabú."""
    for forbidden_move, expiration_iteration in tabu_list:
        if forbidden_move == move_attribute and current_iteration < expiration_iteration:
            return True
    return False

def add_to_tabu_list(tabu_list, move_attribute, expiration_iteration):
    """Añade un movimiento a la lista tabú."""
    tabu_list.append((move_attribute, expiration_iteration))

def remove_expired_tabu_moves(tabu_list, current_iteration):
    """Elimina movimientos de la lista tabú que ya han expirado."""
    # Se recomienda crear una nueva lista en lugar de modificar in-place para evitar problemas de iteración
    tabu_list[:] = [(move, exp) for move, exp in tabu_list if exp > current_iteration]

def update_long_term_memory(attribute_frequency, move_attribute):
    """Actualiza la frecuencia de uso de un atributo para la memoria a largo plazo."""
    attribute_frequency[move_attribute] = attribute_frequency.get(move_attribute, 0) + 1

def add_to_elite_solutions(elite_solutions_attributes, solution):
    """Añade los atributos de una solución élite. Podrías querer limitar el tamaño."""
    attributes = identify_key_attributes(solution) # Función para extraer atributos clave
    if attributes not in elite_solutions_attributes:
        elite_solutions_attributes.append(attributes)
        # Mantener un tamaño razonable, quizás eliminando los menos prometedores
        if len(elite_solutions_attributes) > 10: # Ejemplo
            elite_solutions_attributes.pop(0) # Eliminar el más antiguo

def identify_key_attributes(solution):
    """Extrae atributos representativos de una solución para la memoria a medio plazo."""
    # Implementación específica del problema
    return "solution_attributes" # Ejemplo trivial

def should_diversify(attribute_frequency, threshold):
    """Decide si es necesario diversificar basándose en la memoria a largo plazo."""
    # Si muchos atributos se usan repetidamente, o el algoritmo está ciclando en una región
    # Se podría verificar la varianza de los contadores, o si el mejor coste no mejora en X iteraciones.
    if any(count > threshold for count in attribute_frequency.values()):
        return True
    return False

def generate_diverse_solution(attribute_frequency):
    """Genera una nueva solución que tiende a evitar los atributos más frecuentes."""
    # Podría ser una solución aleatoria con una penalización para atributos comunes
    # o una perturbación grande de la solución actual.
    print("  -> Generando solución diversa...")
    return ["new", "diverse", "solution"] # Ejemplo trivial

def should_intensify(elite_solutions_attributes, factor):
    """Decide si es necesario intensificar basándose en la memoria a medio plazo."""
    # Si se han encontrado buenas soluciones y se cree que hay más cerca.
    # Podría basarse en si el mejor coste ha mejorado recientemente, o en la calidad de las élite.
    if len(elite_solutions_attributes) > 2 and factor > 0.5: # Ejemplo
        return True
    return False

def intensify_search_around_elite(elite_solutions_attributes, current_solution):
    """Realiza una búsqueda más enfocada alrededor de una solución élite."""
    # Podría ser una búsqueda local más profunda, o reiniciar la búsqueda desde una solución élite
    # con un vecindario más restringido o una tenure tabú diferente.
    print("  -> Intensificando la búsqueda...")
    # Por simplicidad, aquí podríamos devolver una de las soluciones élite o una pequeña variación.
    return current_solution # Ejemplo trivial para el pseudocódigo
