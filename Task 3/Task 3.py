import numpy as np

import matplotlib.pyplot as plt

import control as ctrl



# --- 1. Parâmetros do Sistema ---

v = 10.0# [m/s] velocidade longitudinal do automóvel

b = 3.0# [m] distância entre eixos

a = 1.5# [m] distância do centro de massa até o eixo traseiro

k_alpha = a / b# razão geométrica a/b



print("--- Parâmetros do Sistema ---")

print(f"Velocidade longitudinal (v): {v} m/s")

print(f"Distância entre eixos (b): {b} m")

print(f"Razão geométrica (k_alpha = a/b): {k_alpha:.3f}")

print("-" * 30)



# --- 2. Função de Transferência da Planta G(s) ---

# G(s) = (v * k_alpha * s + v² / b) / s²

num_G = [v * k_alpha, v * v / b]

den_G = [1, 0, 0]# s²

G = ctrl.tf(num_G, den_G)



print("\n--- Função de Transferência da Planta G(s) ---")

print(f"G(s) = {G}")

print("-" * 30)



# --- 3. Controlador PI: C(s) = (Kp*s + Ki) / s ---

Kp = 0.8

Ki = 1.2

num_C = [Kp, Ki]

den_C = [1, 0]# s

C = ctrl.tf(num_C, den_C)



print("\n--- Função de Transferência do Controlador PI C(s) ---")

print(f"C(s) = {C}")

print("-" * 30)



# --- 4. Malha Aberta: L(s) = C(s) * G(s) ---

L = C * G



print("\n--- Função de Transferência de Malha Aberta L(s) ---")

print(f"L(s) = {L}")

print("-" * 30)



# --- 5. Malha Fechada: T(s) = L(s) / (1 + L(s)) ---

T = ctrl.feedback(L, 1)



print("\n--- Função de Transferência de Malha Fechada T(s) ---")

print(f"T(s) = {T}")

print("-" * 30)



# --- 6. Parâmetros de Simulação ---

t_sim = np.linspace(0, 5, 1000)# tempo de 0 a 5 segundos

delta_step_amplitude = 0.05# degrau de entrada em radianos



# --- 7. Simulação da Resposta ao Degrau ---

# Sem controle

t_sem_controle, y_sem_controle = ctrl.step_response(delta_step_amplitude * G, T=t_sim)



# Com controle PI

t_com_controle, y_com_controle = ctrl.step_response(delta_step_amplitude * T, T=t_sim)



# --- 8. Gráficos das Respostas ---



# FIGURA 14.1: Sem Controle

plt.figure(figsize=(10, 6))

plt.plot(t_sem_controle, y_sem_controle, 'r', label='Sem Controle')

plt.title('Resposta ao Degrau da Posição Lateral $x_2(t)$ - Sem Controle')

plt.xlabel('Tempo [s]')

plt.ylabel('Posição Lateral $x_2$ [m]')

plt.grid(True)

plt.legend()

plt.show()



# FIGURA 14.2: Com Controle PI

plt.figure(figsize=(10, 6))

plt.plot(t_com_controle, y_com_controle, 'g', label='Com Controle PI')

plt.title('Resposta ao Degrau da Posição Lateral $x_2(t)$ - Com Controle PI')

plt.xlabel('Tempo [s]')

plt.ylabel('Posição Lateral $x_2$ [m]')

plt.grid(True)

plt.legend()

plt.show()



# FIGURA 14.3: Comparação das Respostas (com zoom no eixo Y)

plt.figure(figsize=(10, 6))

plt.plot(t_sem_controle, y_sem_controle, 'r--', linewidth=2, label='Sem Controle')

plt.plot(t_com_controle, y_com_controle, 'g-', linewidth=2.5, label='Com Controle PI')



plt.title('Comparação das Respostas de $x_2(t)$')

plt.xlabel('Tempo [s]')

plt.ylabel('Posição Lateral $x_2$ [m]')

plt.grid(True)

plt.legend()



# Zoom no eixo Y (ajuste os valores conforme sua simulação)

plt.ylim(0, 0.15)# mostra apenas de 0 a 15 cm, por exemplo



plt.show()





# --- 9. Resultados Numéricos (para tabela ou relatório) ---

x2_max_sem_controle = y_sem_controle[-1] * 100# em cm

x2_max_com_controle = np.max(y_com_controle) * 100# em cm



# Informações da resposta controlada

info_com_controle = ctrl.step_info(T * delta_step_amplitude, T=t_sim)

tempo_pico = info_com_controle['PeakTime']

tempo_acomodacao = info_com_controle['SettlingTime']



# Impressão dos resultados

print("\n--- Resultados Numéricos---")

print(f"x2 máximo (Sem controle): {x2_max_sem_controle:.2f} cm")

print(f"x2 máximo (Com controle PI): {x2_max_com_controle:.2f} cm")

print(f"Tempo de pico (Com controle PI): {tempo_pico:.2f} s")

print(f"Tempo de acomodação 2% (Com controle PI): {tempo_acomodacao:.2f} s")

print(f"Erro permanente (Sem controle): Divergente")

print(f"Erro permanente (Com controle PI): 0 cm (devido ao integrador)")

print("-" * 30)