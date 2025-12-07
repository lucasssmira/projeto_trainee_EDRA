import cv2 # Biblioteca do OpenCV
import numpy as np # Estritamente necessário para as operações morfológicas

KERNEL_SIZE = 5  # tamanho do kernel para filtro morfológico (limpeza de imagem)
AREA_MIN = 600   # área mínima de contorno para evitar ruído

# Ranges HSV (Hue, Saturation, Value) pré-definidos (pode calibrar com trackbars)
COLOR_RANGES = {
    "vermelho":   [(0, 120, 70), (10, 255, 255)],
    "Vermelho2": [(170, 100, 50), (179, 255, 255)], # Faixa extra para vermelho
    "azul":  [(90, 80, 50), (130, 255, 255)],
    "laranja": [(5, 100, 100), (20, 255, 255)],
    "magenta": [(140, 100, 100), (170, 255, 255)],
    "marrom": [(0, 50, 0), (179, 255, 60)],
    "verde": [(40, 50, 50), (80, 255, 255)]
    
}


# FUNÇÕES AUXILIARES

def create_trackbars():
    """Cria trackbars para calibração em tempo real."""
    cv2.namedWindow("Trackbars")
    
    # Controla os mínimos e máximos de Hue
    cv2.createTrackbar("Hmin", "Trackbars", 0, 179, lambda x: None)
    cv2.createTrackbar("Hmax", "Trackbars", 140, 179, lambda x: None)
    
    # Controla os mínimos e máximo de Saturation
    cv2.createTrackbar("Smin", "Trackbars", 0, 255, lambda x: None)
    cv2.createTrackbar("Smax", "Trackbars", 255, 255, lambda x: None)
    
    # Controla os mínimos e máximo de Value
    cv2.createTrackbar("Vmin", "Trackbars", 0, 255, lambda x: None)
    cv2.createTrackbar("Vmax", "Trackbars", 255, 255, lambda x: None)


def get_hsv_from_trackbars():
    """Lê os valores HSV -sistema de cores- das trackbars."""
    h_min = cv2.getTrackbarPos("Hmin", "Trackbars")
    h_max = cv2.getTrackbarPos("Hmax", "Trackbars")
    s_min = cv2.getTrackbarPos("Smin", "Trackbars")
    s_max = cv2.getTrackbarPos("Smax", "Trackbars")
    v_min = cv2.getTrackbarPos("Vmin", "Trackbars")
    v_max = cv2.getTrackbarPos("Vmax", "Trackbars")
    
    # Retorna limites mínimos e máximos inseridos
    return (h_min, s_min, v_min), (h_max, s_max, v_max)


def detect_shape(contour):
    """Classifica o formato geométrico baseado na aproximação poligonal."""
    peri = cv2.arcLength(contour, True)
    
    # Simplifica a forma utilizando Ramer-Douglas_Peucker
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    
    # Calcula os vértices encontrados no frame
    sides = len(approx)

    if sides == 3:
        return "Triângulo"
    elif sides == 4:
        # Checar se é quadrado (razão de aspecto ~1)
        x, y, w, h = cv2.boundingRect(approx)
        
        # Divide o comprimento pela largura
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:
            return "Quadrado"
        else:
            return "Retângulo"
    elif sides == 5:
        return "Pentagono"
    elif sides == 6:
        return "Hexagono"
    elif sides == 10:
        return "Estrela"
    elif sides == 12:
        return "Cruz"
    elif sides > 12:
        return "Circulo"
    else:
        return f"Poligono({sides})" # Mostra o n° de lados se não souber o nome


def detect_color(hsv_pixel):
    """Retorna a cor provável do objeto."""
    h, s, v = hsv_pixel

    for color, (lower, upper) in COLOR_RANGES.items():
        lh, ls, lv = lower
        uh, us, uv = upper
        
        # Requisitos de cor legível
        if lh <= h <= uh and ls <= s <= us and lv <= v <= uv:
            return color.capitalize()

    return "Desconhecida"


# LOOP PRINCIPAL DO SISTEMA DE VISÃO

def main():
    create_trackbars()

    cap = cv2.VideoCapture("video_desafio_1.mp4")

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8) # Uso de Numpy

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = get_hsv_from_trackbars()
        
        """Percorre cada pixel de hsv, verifica se estão
        dentro dos padrões de lower e upper. Se sim, 1 (branco)
        se não, 0 (preto)"""
        
        mask = cv2.inRange(hsv, lower, upper)

        # Remove ruídos da area 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Preenche buracos da area 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # INVERSÃO: Inverte a máscara para que os pixels pretos (formas) fiquem brancos
        mask = cv2.bitwise_not(mask)
        
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Ignora ruídos
            if area < AREA_MIN:
                continue

            shape = detect_shape(cnt)

            # pegar ponto médio para identificar a cor
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                color = detect_color(hsv[cy, cx])
            else:
                color = "?"

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = f"{shape} - {color}"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Processado", frame)
        
        mask_resized = cv2.resize(mask, (400, 360))
        cv2.imshow("Mascara", mask_resized)

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)
        if cv2.waitKey(delay) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
