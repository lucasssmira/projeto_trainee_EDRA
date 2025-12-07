from tello_sim_EDRA import Simulator
import numpy as np
import itertools
import math

# ----------------------------
# Criação do simulador e decolagem
# ----------------------------

drone = Simulator()
drone.takeoff()

# ----------------------------
# Funções de apoio (cálculos geométricos)
# ----------------------------

def calcular_distancia(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calcular_custo_rota(rota):
    custo = 0
    for i in range(len(rota) - 1):
        custo += calcular_distancia(rota[i], rota[i + 1])
    return custo

def calcular_angulo(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

# ----------------------------
# PLANEJAMENTO DA ROTA (TSP por força bruta)
# ----------------------------

tesouros = drone.treasures.copy() # Copia a lista de tesouros do simulador
tesouros.insert(0, (0,0))  # Adiciona a posicao inicial do drone

n = len(tesouros)
origem = 0
indices = list(range(n))
outras = [i for i in indices if i != origem] # índices dos tesouros (sem a origem)

# Variáveis auxiliares para guardar a melhor solução encontrada até agora
melhor_custo = float('inf') 
melhor_permutacao = None 

# Gera todas as permutações possíveis da ordem de visita aos tesouros
for perm in itertools.permutations(outras):
    rota_indices = [origem] + list(perm) + [origem]
    rota_pontos = [tesouros[i] for i in rota_indices]
    custo = calcular_custo_rota(rota_pontos)

    if custo < melhor_custo:
        melhor_custo = custo
        melhor_permutacao = rota_indices
        melhor_permutacao

melhor_rota_pontos = [tesouros[i] for i in melhor_permutacao]

# ----------------------------
# EXECUÇÃO DA ROTA NO SIMULADOR
# ----------------------------

angulo_atual = 90  # começa apontando para o norte

for destino in melhor_rota_pontos[1:]:  # pula o primeiro, que é a posição inicial
    angulo_destino = calcular_angulo(drone.cur_loc, destino)  # usa atan2
    angulo_rotacao = angulo_destino - angulo_atual

    # normalizar para (-180, 180]
    if angulo_rotacao > 180:
        angulo_rotacao -= 360
    elif angulo_rotacao <= -180:
        angulo_rotacao += 360

    # girar para o menor lado
    if angulo_rotacao > 0:
        drone.ccw(angulo_rotacao)   # gira anti-horário
    elif angulo_rotacao < 0:
        drone.cw(-angulo_rotacao)   # gira horário

    # agora o drone está apontando para o destino
    angulo_atual = angulo_destino   # ATUALIZA o ângulo atual

    distancia = calcular_distancia(drone.cur_loc, destino)
    drone.forward(distancia)

# ----------------------------
# FINALIZAÇÃO DA MISSÃO
# ----------------------------

drone.land()