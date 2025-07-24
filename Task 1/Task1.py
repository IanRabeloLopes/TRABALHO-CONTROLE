"""
Trabalho de Sistemas de Controle – Modelo Bicicleta para AUTOMÓVEL (Python)
"""

import numpy as np
from scipy.integrate import odeint
import control as ctrl
import control.matlab as matlab # Importando o submódulo matlab para clareza
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. PARÂMETROS DO AUTOMÓVEL (4 RODAS)
# ------------------------------------------------------------------
print("=" * 60)
print("MODELO BICICLETA PARA AUTOMÓVEL DE 4 RODAS (VERSÃO REVISADA)")
print("=" * 60)

a = 1.5  # [m] distância CM → eixo traseiro
b = 3.0  # [m] entre-eixos
v = 10.0 # [m/s] velocidade longitudinal do automóvel (constante)

k_alpha = a / b

print(f"Parâmetros do Veículo:")
print(f"  a (dist. CM-eixo traseiro) = {a} m")
print(f"  b (entre-eixos) = {b} m")
print(f"  v (velocidade) = {v} m/s")
print(f"  k_alpha (a/b) = {k_alpha:.3f}")
print("=" * 60)


# ------------------------------------------------------------------
# 2. MODELO NÃO LINEAR COMPLETO
# ------------------------------------------------------------------
def automovel_reduzido_completo(x, t, delta, v, a, b):
    x2, theta = x
    alpha = np.arctan((a / b) * np.tan(delta))
    dx2 = v * np.sin(theta + alpha)
    dth = (v / b) * np.tan(delta)
    return [dx2, dth]


# ------------------------------------------------------------------
# 3. PONTO DE EQUILÍBRIO – AUTOMÓVEL EM LINHA RETA
# ------------------------------------------------------------------
print("\n3. ANÁLISE DE EQUILÍBRIO:")
print("Para movimento retilíneo do automóvel (θ = 0):")
print("- Ângulo de direção: δ = 0 rad (rodas alinhadas)")
print("- Estado de equilíbrio: (x₂, θ) = (0, 0)")

x_eq = np.array([0.0, 0.0])
delta_eq = 0.0

# ------------------------------------------------------------------
# 4. LINEARIZAÇÃO DO MODELO DO AUTOMÓVEL
# ------------------------------------------------------------------
print("\n4. LINEARIZAÇÃO:")
print("Jacobianas do modelo completo calculadas no ponto de equilíbrio:")

A = np.array([[0.0, v],
              [0.0, 0.0]])

B = np.array([[v * k_alpha],
              [v / b]])

C = np.eye(2)
D = np.zeros((2, 1))

print(f"Matriz A:\n{A}")
print(f"Matriz B (Revisada):\n{B}")

sys_automovel_revisado = ctrl.ss(A, B, C, D)

# ------------------------------------------------------------------
# 5. PROPRIEDADES DO SISTEMA LINEARIZADO
# ------------------------------------------------------------------
Co = ctrl.ctrb(A, B)
Ob = ctrl.obsv(A, C)
print(f"\nCONTROLABILIDADE: rank = {np.linalg.matrix_rank(Co)} (controlável)")
print(f"OBSERVABILIDADE: rank = {np.linalg.matrix_rank(Ob)} (observável)")
print(f"AUTOVALORES: {np.linalg.eigvals(A)} (sistema marginalmente estável)")

# ------------------------------------------------------------------
# 6. SIMULAÇÃO DO COMPORTAMENTO DO AUTOMÓVEL
# ------------------------------------------------------------------
print("\n6. SIMULAÇÃO (Comparando modelos equivalentes):")
print("Entrada: degrau no ângulo de direção δ = 0 → 0.05 rad (~2.9°)")

t = np.linspace(0, 2, 1000)
A_step = 0.05

# Simulação do automóvel não-linear COMPLETO
def nl_rhs(x, t):
    return automovel_reduzido_completo(x, t, delta=A_step, v=v, a=a, b=b)
x_nl = odeint(nl_rhs, x_eq, t)

# Simulação do modelo linearizado
# A função retorna (saídas, tempo, estados)
y_lin, t_lin, x_lin = matlab.lsim(sys_automovel_revisado, U=A_step, T=t, X0=x_eq)


# ------------------------------------------------------------------
# 7. RESULTADOS DA SIMULAÇÃO
# ------------------------------------------------------------------
df = pd.DataFrame({
    "tempo[s]": t,
    "x2_naolinear[m]": x_nl[:, 0],
    "x2_linearizado[m]": x_lin[:, 0],
    "theta_naolinear[rad]": x_nl[:, 1],
    "theta_linearizado[rad]": x_lin[:, 1]
})

df["erro_posicao[m]"] = np.abs(df["x2_naolinear[m]"] - df["x2_linearizado[m]"])
df["erro_orientacao[rad]"] = np.abs(df["theta_naolinear[rad]"] - df["theta_linearizado[rad]"])

df.to_csv("simulacao_automovel_revisada.csv", index=False)

print("\n7. RESULTADOS FINAIS (t=2s):")
print("=" * 50)
print(f"Posição lateral final:")
print(f"  - Modelo Não-linear: {df['x2_naolinear[m]'].iloc[-1]:.4f} m")
print(f"  - MOdelo Linearizado: {df['x2_linearizado[m]'].iloc[-1]:.4f} m")
print(f"  - Diferença: {df['erro_posicao[m]'].iloc[-1]:.4f} m")

print(f"\nOrientação final:")
print(f"  - Modelo Não-linear: {df['theta_naolinear[rad]'].iloc[-1]:.4f} rad")
print(f"  - Modelo Linearizado: {df['theta_linearizado[rad]'].iloc[-1]:.4f} rad")
print(f"  - Erro máximo: {df['erro_orientacao[rad]'].max() * 180 / np.pi:.4f}°")
print(f"\nArquivo salvo: simulacao_automovel.csv")


# ------------------------------------------------------------------
# 8. PLOTAGEM DOS RESULTADOS
# ------------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, df["x2_naolinear[m]"], label="Modelo Não linear ")
plt.plot(t, df["x2_linearizado[m]"], "--", label="Modelo Linearizado")
plt.xlabel("Tempo [s]")
plt.ylabel("Posição lateral x₂ [m]")
plt.title("Comparação de Posição Lateral x₂(t)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, df["theta_naolinear[rad]"], label="Modelo Não linear ")
plt.plot(t, df["theta_linearizado[rad]"], "--", label="Modelo Linearizado ")
plt.xlabel("Tempo [s]")
plt.ylabel("Orientação θ [rad]")
plt.title("Comparação de Orientação θ(t)")
plt.legend()
plt.grid(True)

plt.suptitle("Comparação dos Modelos (t=2s)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.figure(figsize=(6,4))
plt.plot(t, df["erro_posicao[m]"], label="|x₂ₙₗ − x₂ₗᵢₙ|")
plt.plot(t, df["erro_orientacao[rad]"], label="|θₙₗ − θₗᵢₙ|")
plt.xlabel("Tempo [s]")
plt.ylabel("Erro absoluto")
plt.title("Erro entre Modelos (t=2s)")
plt.legend()
plt.grid(True)
plt.show()
