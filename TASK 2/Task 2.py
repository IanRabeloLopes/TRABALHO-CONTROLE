import numpy as np

import matplotlib.pyplot as plt

import control as ctrl



# --- 1. Definir os parâmetros do sistema ---

# Estes parâmetros são consistentes com a Tarefa 1

# e a da Tarefa 2.

v = 10.0# [m/s] velocidade longitudinal do automóvel (constante)

b = 3.0# [m] entre-eixos

a = 1.5# [m] distância do centro de massa ao eixo traseiro (usado para k_alpha)

k_alpha = a / b # Razão geométrica a/b = 1.5 / 3.0 = 0.5



print("--- Parâmetros do Sistema ---")

print(f"Velocidade longitudinal (v): {v} m/s")

print(f"Distância entre-eixos (b): {b} m (Corrigido)")

print(f"Razão geométrica (k_alpha = a/b): {k_alpha:.3f}")

print("-" * 30)



# --- 2. Construir as matrizes do sistema linearizado (A, B, C_x2, C_theta, D) ---

# Matriz A :

A = np.array([[0, v],

[0, 0]])



# Matriz B, conforme a Tarefa 2 e o modelo da Tarefa 1.

# Esta é a principal alteração em relação ao código original do Apêndice B.

# B = np.array([[0], [v / b]])

B = np.array([[v * k_alpha],# Componente para x2 dot

[v / b]]) # Componente para theta dot



# Matrizes de saída para cada variável de interesse (assumindo C=I, D=0)

# Podemos criar sistemas separados para cada saída ou usar um C de 2x2 e depois extrair as FTs

C_x2 = np.array([[1, 0]])# Para a saída x2 (posição lateral)

C_theta = np.array([[0, 1]]) # Para a saída theta (orientação)

D = np.array([[0]])# Matriz D para uma única saída (será usada para cada FT)



print("\n--- Matrizes do Sistema Linearizado ---")

print("Matriz A:\n", A)

print("Matriz B :\n", B)

print("Matriz C_x2:\n", C_x2)

print("Matriz C_theta:\n", C_theta)

print("Matriz D:\n", D)

print("-" * 30)



# --- 3. Criar os modelos de espaço de estados para cada saída ---

# Criamos um modelo para cada saída, usando as matrizes C específicas

sys_x2 = ctrl.ss(A, B, C_x2, D)

sys_theta = ctrl.ss(A, B, C_theta, D)



# --- 4. Converter os modelos de espaço de estados para funções de transferência ---

tf_x2 = ctrl.ss2tf(sys_x2)

tf_theta = ctrl.ss2tf(sys_theta)



print("\n--- Funções de Transferência Obtidas ---")

print(f"G_x2(s) (Posição Lateral) = {tf_x2}")

print(f"G_theta(s) (Orientação) = {tf_theta}")

# As funções de transferência esperadas são:

# G_x2(s) = (5s + 33.333) / s^2

# G_theta(s) = 3.333 / s

# O controle.py arredonda para exibir, mas os valores internos são de ponto flutuante

print("-" * 30)



# --- 5. Definir a amplitude da entrada em degrau e o vetor de tempo ---

t = np.linspace(0, 5, 500) # Intervalo de tempo de 0 a 5 segundos, com 500 pontos

delta_step = 0.05# Amplitude do degrau em radianos (conforme revisão)



# --- 6. Simular a resposta ao degrau para ambas as funções de transferência ---

# Multiplicar a FT pela amplitude do degrau para obter a resposta com a escala correta

t_x2_resp, y_x2_resp = ctrl.step_response(delta_step * tf_x2, T=t)

t_theta_resp, y_theta_resp = ctrl.step_response(delta_step * tf_theta, T=t)



# --- 7. Gerar e exibir o gráfico da resposta ao degrau da posição lateral x2(t) ---

plt.figure(figsize=(10, 6))

plt.plot(t_x2_resp, y_x2_resp, 'b', label='Resposta de $x_2(t)$')

plt.title('Resposta ao Degrau: Posição Lateral $x_2(t)$')

plt.xlabel('Tempo [s]')

plt.ylabel('Posição Lateral $x_2$ [m]')

plt.grid(True)

plt.legend()

plt.show()



# --- 8. Gerar e exibir o gráfico da resposta ao degrau da orientação theta(t) ---

plt.figure(figsize=(10, 6))

plt.plot(t_theta_resp, y_theta_resp, 'orange', label='Resposta de $\\theta(t)$')

plt.title('Resposta ao Degrau: Orientação $\\theta(t)$')

plt.xlabel('Tempo [s]')

plt.ylabel('Orientação $\\theta$ [rad]')

plt.grid(True)

plt.legend()

plt.show()



# --- 9. Gerar e exibir o mapa de polos e zeros para a função de transferência Gx2(s) ---

plt.figure(figsize=(8, 8))

# Adicionar um ponto para o zero em -6.66 para visualização clara, se o pzmap não o mostrar automaticamente

# (o pzmap do python-control já deve identificar o zero da TF)

ctrl.pzmap(tf_x2, plot=True, title="Mapa de Polos e Zeros $G_{x_2}(s)$", grid=True)

plt.show()



# --- 10. Gerar e exibir o mapa de polos e zeros para a função de transferência Gtheta(s) ---

plt.figure(figsize=(8, 8))

ctrl.pzmap(tf_theta, plot=True, title="Mapa de Polos e Zeros $G_{\\theta}(s)$", grid=True)

plt.show()