import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.feature import corner_harris, corner_peaks, corner_shi_tomasi

# Rutas
IMAGES_DIR = "images"
TEMPLATE_PATH = os.path.join("template", "pattern.png")
RESULTS_DIR = "resultados"

# Parámetros
THRESHOLD = 0.75 

# Lista de imágenes
imagenes = [
    "COCA-COLA-LOGO.jpg",
    "coca_logo_1.png",
    "coca_logo_2.png",
    "coca_multi.png",
    "coca_retro_1.png",
    "coca_retro_2.png",
    "logo_1.png"
]

def mostrar_resultados(img_color, detecciones, nombre_img, esquinas_h=None, esquinas_st=None, lineas=None, circulos=None, overlays=False):
    """
    Visualiza y guarda la imagen con bounding boxes y niveles de confianza.
    Si overlays=True, dibuja esquinas y líneas/círculos.
    """
    img_viz = img_color.copy()
    # Dibuja bounding boxes
    if len(detecciones) == 0:
        cv2.putText(img_viz, "Sin detecciones", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    else:
        for (x, y, w, h, score) in detecciones:
            cv2.rectangle(img_viz, (x, y), (x+w, y+h), (0,255,0), 4)
            cv2.putText(img_viz, f"Score: {score}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    # Overlays solo si se piden
    if overlays:
        if esquinas_h is not None:
            for corner in esquinas_h:
                cv2.circle(img_viz, (corner[1], corner[0]), 3, (255, 0, 255), -1)
        if esquinas_st is not None:
            for corner in esquinas_st:
                cv2.circle(img_viz, (corner[1], corner[0]), 3, (0, 255, 255), -1)
        if lineas is not None:
            for x1, y1, x2, y2 in lineas:
                cv2.line(img_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if circulos is not None:
            for x, y, r in circulos:
                cv2.circle(img_viz, (x, y), r, (0, 0, 255), 2)
    plt.figure(figsize=(10,8))
    plt.imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
    plt.title(nombre_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"resultado_{nombre_img}.png"))
    plt.close()


def deteccion_simple(img_gray, template_gray):
    """
    Detección única: devuelve la mejor coincidencia.
    """
    if img_gray.shape[0] < template_gray.shape[0] or img_gray.shape[1] < template_gray.shape[1]:
        return []
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    h, w = template_gray.shape
    return [(max_loc[0], max_loc[1], w, h, max_val)]


def deteccion_multiple(img_gray, template_gray, threshold=THRESHOLD):
    """
    Detección múltiple: devuelve todas las coincidencias por encima del umbral.
    """
    if img_gray.shape[0] < template_gray.shape[0] or img_gray.shape[1] < template_gray.shape[1]:
        return []
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    h, w = template_gray.shape
    detecciones = []
    for pt in zip(*loc[::-1]):
        score = res[pt[1], pt[0]]
        detecciones.append((pt[0], pt[1], w, h, score))
    # Supresión de no-máximos para evitar solapamientos
    boxes = np.array([[x, y, x+w, y+h, score] for (x, y, w, h, score) in detecciones])
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes[:,:4].tolist(), scores=boxes[:,4].tolist(), score_threshold=threshold, nms_threshold=0.3
    )
    final = [tuple(boxes[i][0:4].astype(int)) + (boxes[i][4],) for i in indices.flatten()]
    return final


def template_matching_multiescala(img_gray, template_gray, scales=np.linspace(0.5, 1.5, 10), threshold=THRESHOLD):
    """
    Busca el template en la imagen a múltiples escalas usando pirámides.
    Devuelve todas las detecciones por encima del umbral.
    """
    h_t, w_t = template_gray.shape
    detecciones = []
    for scale in scales:
        t_scaled = cv2.resize(template_gray, (int(w_t*scale), int(h_t*scale)))
        if img_gray.shape[0] < t_scaled.shape[0] or img_gray.shape[1] < t_scaled.shape[1]:
            continue
        res = cv2.matchTemplate(img_gray, t_scaled, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            score = res[pt[1], pt[0]]
            detecciones.append((pt[0], pt[1], t_scaled.shape[1], t_scaled.shape[0], score))
    # Supresión de no-máximos
    boxes = np.array([[x, y, x+w, y+h, score] for (x, y, w, h, score) in detecciones])
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes[:,:4].tolist(), scores=boxes[:,4].tolist(), score_threshold=threshold, nms_threshold=0.3
    )
    final = [tuple(boxes[i][0:4].astype(int)) + (boxes[i][4],) for i in indices.flatten()]
    return final


def orb_match(img_gray, template_gray, min_matches=10):
    """
    Detecta el logo usando características locales ORB.
    Devuelve bounding boxes de las coincidencias encontradas.
    """
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)
    if des1 is None or des2 is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < min_matches:
        return []
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:min_matches]]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:min_matches]]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is not None:
        h, w = template_gray.shape
        pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        x, y, w_box, h_box = int(dst[:,0,0].min()), int(dst[:,0,1].min()), int(dst[:,0,0].max()-dst[:,0,0].min()), int(dst[:,0,1].max()-dst[:,0,1].min())
        return [(x, y, w_box, h_box, 1.0)]
    return []


def esquinas_harris(img_gray, min_distance=10):
    """
    Detecta esquinas con Harris.
    Devuelve lista de puntos.
    """
    corners = corner_peaks(corner_harris(img_gray), min_distance=min_distance)
    return corners


def esquinas_shi_tomasi(img_gray, min_distance=10):
    """
    Detecta esquinas con Shi-Tomasi.
    Devuelve lista de puntos.
    """
    corners = corner_peaks(corner_shi_tomasi(img_gray), min_distance=min_distance)
    return corners


def hough_lineas(img_gray):
    """
    Detecta líneas usando la transformada de Hough.
    Devuelve lista de líneas (x1, y1, x2, y2).
    """
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return []
    return lines.reshape(-1, 4)


def hough_circulos(img_gray):
    """
    Detecta círculos usando la transformada de Hough.
    Devuelve lista de círculos (x, y, r).
    """
    img_blur = cv2.medianBlur(img_gray, 5)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        return np.uint16(np.around(circles[0,:]))
    return []


def detectar_logo_clasico(img_gray, template_gray, escalas=np.linspace(0.5, 1.5, 8), rotaciones=np.arange(-30, 31, 15), min_matches=10):
    """
    Busca el template en la imagen usando SIFT (o ORB) a múltiples escalas y rotaciones, validando por homografía.
    Devuelve la mejor detección encontrada (bounding box + score de inliers).
    """
    if hasattr(cv2, 'SIFT_create'):
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(1000)
        norm_type = cv2.NORM_HAMMING
    kp1, des1 = detector.detectAndCompute(template_gray, None)
    best = None
    best_inliers = 0
    h_t, w_t = template_gray.shape
    for escala in escalas:
        for angulo in rotaciones:
            # Escala y rota el template
            M_scale = cv2.getRotationMatrix2D((w_t/2, h_t/2), angulo, escala)
            template_warp = cv2.warpAffine(template_gray, M_scale, (w_t, h_t))
            kp1_w, des1_w = detector.detectAndCompute(template_warp, None)
            if des1_w is None or len(kp1_w) < min_matches:
                continue
            kp2, des2 = detector.detectAndCompute(img_gray, None)
            if des2 is None or len(kp2) < min_matches:
                continue
            matcher = cv2.BFMatcher(norm_type, crossCheck=True)
            matches = matcher.match(des1_w, des2)
            if len(matches) < min_matches:
                continue
            src_pts = np.float32([kp1_w[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None and mask is not None:
                inliers = int(mask.sum())
                if inliers > best_inliers:
                    h, w = template_warp.shape
                    pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, M)
                    x, y, w_box, h_box = int(dst[:,0,0].min()), int(dst[:,0,1].min()), int(dst[:,0,0].max()-dst[:,0,0].min()), int(dst[:,0,1].max()-dst[:,0,1].min())
                    best = [(x, y, w_box, h_box, inliers)]
                    best_inliers = inliers
    if best is not None:
        return best
    return []


def main():

    template = cv2.imread(TEMPLATE_PATH)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape

    for nombre_img in imagenes:
        img_path = os.path.join(IMAGES_DIR, nombre_img)
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detecciones_tm = deteccion_multiple(img_gray, template_gray)
        detecciones_tm_multi = template_matching_multiescala(img_gray, template_gray)
        detecciones_orb = orb_match(img_gray, template_gray)
        esquinas_h = esquinas_harris(img_gray)
        esquinas_st = esquinas_shi_tomasi(img_gray)
        lineas = hough_lineas(img_gray)
        circulos = hough_circulos(img_gray)

        mostrar_resultados(img, detecciones_tm, nombre_img + '_tm', esquinas_h, esquinas_st, lineas, circulos, overlays=True)
        mostrar_resultados(img, detecciones_tm_multi, nombre_img + '_tm_multi', esquinas_h, esquinas_st, lineas, circulos, overlays=True)
        mostrar_resultados(img, detecciones_orb, nombre_img + '_orb', esquinas_h, esquinas_st, lineas, circulos, overlays=True)

        deteccion_robusta = detectar_logo_clasico(img_gray, template_gray)

        esquinas_h = esquinas_harris(img_gray)
        esquinas_st = esquinas_shi_tomasi(img_gray)
        lineas = hough_lineas(img_gray)
        circulos = hough_circulos(img_gray)

        mostrar_resultados(img, deteccion_robusta, nombre_img + '_robusto', overlays=False)
        print(f"Procesada {nombre_img} | TM: {len(detecciones_tm)}, TM-multi: {len(detecciones_tm_multi)}, ORB: {len(detecciones_orb)}, Robust matches: {len(deteccion_robusta)}")

if __name__ == "__main__":
    main()
