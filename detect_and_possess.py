import csv
import cv2
import time
import multiprocessing
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sem janela, compatível com multiprocessing
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from preprocessing import preprocess_frame


VIDEO_PATH  = r"C:\Users\enzom\Desktop\TIVI\Jogos\england_epl\2014-2015\2015-02-21 - 18-00 Swansea 2 - 1 Manchester United\1_224p.mkv"
OUTPUT_DIR  = "output"

TEAM_A_NAME = "Manchester United"
TEAM_B_NAME = "Swansea"

NUM_PROCESSOS = 4

PROCESSOS_TESTE = [1, 2, 4]

FRAMES_FORTE = 400

FRAMES_POR_PROCESSO_FRACA = 100

POSSESSION_THRESHOLD = 80

CONF_THRESHOLD = 0.35

FRAME_STEP = 2

BATCH_SIZE = 64

COLOR_BALL     = (0,   255, 255)
COLOR_PLAYER_A = (255,  80,  80)
COLOR_PLAYER_B = (80,   80, 255)

COCO_PERSON = 0
COCO_BALL   = 32

CALIBRATION_FRAMES = 60



def worker_preprocess(frames_bytes: list) -> list:

    resultados = []
    for frame in frames_bytes:
        frame_limpo, mascara = preprocess_frame(frame)
        resultados.append((frame_limpo, mascara))
    return resultados


def preprocessar_lote_paralelo(frames: list, n_processos: int) -> list:

    if n_processos == 1:
        return worker_preprocess(frames)

    tamanho_chunk = max(1, len(frames) // n_processos)
    chunks = [
        frames[i:i + tamanho_chunk]
        for i in range(0, len(frames), tamanho_chunk)
    ]

    with multiprocessing.Pool(processes=n_processos) as pool:
        resultados_chunks = pool.map(worker_preprocess, chunks)

    # Junta os chunks na ordem correta
    resultados = []
    for chunk in resultados_chunks:
        resultados.extend(chunk)
    return resultados


class TeamClusterer:
    def __init__(self):
        self.centroids    = None
        self._calib_hists = []
        self._calib_count = 0
        self._calibrated  = False

    def fit_or_predict(self, frame: np.ndarray, persons: list) -> list:
        if len(persons) == 0:
            return []
        hists = [self._shirt_histogram(frame, box) for box in persons]
        if not self._calibrated:
            self._calib_hists.extend(hists)
            self._calib_count += 1
            if self._calib_count >= CALIBRATION_FRAMES:
                self._run_calibration()
            return self._fallback_kmeans(frame, persons)
        return self._classify(hists)

    def _shirt_histogram(self, frame: np.ndarray, box: tuple) -> np.ndarray:
        x1, y1, x2, y2 = box
        ry1 = y1 + int((y2 - y1) * 0.15)
        ry2 = y1 + int((y2 - y1) * 0.55)
        rx1 = x1 + int((x2 - x1) * 0.15)
        rx2 = x2 - int((x2 - x1) * 0.15)
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return np.zeros(16 * 8, dtype=np.float32)
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 40]), np.array([180, 255, 255]))
        hist = cv2.calcHist([hsv], [0, 1], mask, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist.flatten()

    def _run_calibration(self) -> None:
        print(f"[INFO] Calibrando times com {len(self._calib_hists)} amostras...")
        data = np.array(self._calib_hists, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        _, labels, centers = cv2.kmeans(
            data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        self.centroids   = [centers[0], centers[1]]
        self._calibrated = True
        print("[INFO] Calibração concluída.")

    def _classify(self, hists: list) -> list:
        labels = []
        for h in hists:
            h32 = h.reshape(16, 8).astype(np.float32)
            c0  = self.centroids[0].reshape(16, 8).astype(np.float32)
            c1  = self.centroids[1].reshape(16, 8).astype(np.float32)
            d0  = cv2.compareHist(h32, c0, cv2.HISTCMP_BHATTACHARYYA)
            d1  = cv2.compareHist(h32, c1, cv2.HISTCMP_BHATTACHARYYA)
            labels.append(0 if d0 <= d1 else 1)
        return labels

    def _fallback_kmeans(self, frame: np.ndarray, persons: list) -> list:
        if len(persons) < 2:
            return [0] * len(persons)
        colors = []
        for (x1, y1, x2, y2) in persons:
            ry1  = y1 + int((y2 - y1) * 0.20)
            ry2  = y1 + int((y2 - y1) * 0.60)
            crop = frame[ry1:ry2, x1:x2]
            colors.append(crop.reshape(-1, 3).mean(axis=0).tolist()
                          if crop.size > 0 else [128.0, 128.0, 128.0])
        data     = np.array(colors, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(data, 2, None, criteria, 5,
                                   cv2.KMEANS_RANDOM_CENTERS)
        return labels.flatten().tolist()



def euclidean(p1, p2):
    return float(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))


def assign_possession(ball_center, players):
    if ball_center is None or len(players) == 0:
        return None
    min_dist, min_team = float("inf"), None
    for (cx, cy, team) in players:
        d = euclidean(ball_center, (cx, cy))
        if d < min_dist:
            min_dist, min_team = d, team
    if min_dist > POSSESSION_THRESHOLD:
        return None
    return "A" if min_team == 0 else "B"



def draw_boxes(frame, detections):
    for (x1, y1, x2, y2, label, conf, color) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1-5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    return frame


def draw_possession_bar(frame, pct_a, pct_b):
    h, w    = frame.shape[:2]
    bar_h   = 62
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-bar_h), (w, h), (15, 15, 15), -1)
    frame   = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

    by1, by2 = h-16, h-6
    bx1, bx2 = 16, w-16
    split_x  = bx1 + int((bx2-bx1) * pct_a / 100)
    cv2.rectangle(frame, (bx1, by1), (bx2,     by2), (50, 50, 50),    -1)
    cv2.rectangle(frame, (bx1, by1), (split_x, by2), COLOR_PLAYER_A,  -1)
    cv2.rectangle(frame, (split_x, by1), (bx2,  by2), COLOR_PLAYER_B, -1)

    score        = f"{TEAM_A_NAME}  {pct_a:.0f}%  X  {pct_b:.0f}%  {TEAM_B_NAME}"
    font         = cv2.FONT_HERSHEY_DUPLEX
    fscale, thick = 0.72, 2
    (tw, _)      = cv2.getTextSize(score, font, fscale, thick)[0]
    tx, ty       = (w - tw) // 2, h - bar_h + 38
    cv2.putText(frame, score, (tx+1, ty+1), font, fscale, (0,0,0),       thick+1)
    cv2.putText(frame, score, (tx,   ty),   font, fscale, (255,255,255), thick)
    return frame



def extrair_frames_para_teste(video_path: str, n: int) -> list:
    """Extrai n frames distribuídos uniformemente do vídeo."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // n)
    selecionados = set(range(0, total, step)[:n])
    frames, idx  = [], 0
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in selecionados:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def medir_tempo(frames: list, n_processos: int, repeticoes: int = 3) -> float:
    """Roda o pré-processamento paralelo N vezes e retorna a mediana."""
    tempos = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        preprocessar_lote_paralelo(frames, n_processos)
        tempos.append(time.perf_counter() - t0)
    return float(np.median(tempos))


def rodar_escalabilidade_forte(video_path: str) -> dict:
    print(f"\n[ESCALA FORTE] Problema fixo: {FRAMES_FORTE} frames")
    frames = extrair_frames_para_teste(video_path, FRAMES_FORTE)
    res    = {"processos": [], "tempo": [], "speedup": [], "eficiencia": []}
    t1     = None
    for p in PROCESSOS_TESTE:
        t = medir_tempo(frames, p)
        if t1 is None:
            t1 = t
        s = t1 / t
        e = s / p
        res["processos"].append(p)
        res["tempo"].append(t)
        res["speedup"].append(s)
        res["eficiencia"].append(e)
        print(f"  p={p} | T={t:.3f}s | Speedup={s:.3f} | Eficiência={e:.3f}")
    return res


def rodar_escalabilidade_fraca(video_path: str) -> dict:
    print(f"\n[ESCALA FRACA] Carga por processo: {FRAMES_POR_PROCESSO_FRACA} frames")
    max_frames = FRAMES_POR_PROCESSO_FRACA * max(PROCESSOS_TESTE)
    todos      = extrair_frames_para_teste(video_path, max_frames)
    res        = {"processos": [], "frames_total": [], "tempo": [], "speedup": [], "eficiencia": []}
    t1         = None
    for p in PROCESSOS_TESTE:
        n      = FRAMES_POR_PROCESSO_FRACA * p
        frames = (todos * ((n // len(todos)) + 1))[:n]
        t      = medir_tempo(frames, p)
        if t1 is None:
            t1 = t
        s = t1 / t
        e = s
        res["processos"].append(p)
        res["frames_total"].append(n)
        res["tempo"].append(t)
        res["speedup"].append(s)
        res["eficiencia"].append(e)
        print(f"  p={p} | frames={n} | T={t:.3f}s | Eficiência={e:.3f}")
    return res


def gerar_graficos(forte: dict, fraca: dict, output_dir: str) -> None:
    AZUL    = "#2E75B6"
    LARANJA = "#E36C09"
    VERDE   = "#70AD47"
    CINZA   = "#595959"

    def estilo(ax, titulo, xlabel, ylabel, processos):
        ax.set_title(titulo, fontsize=11, fontweight="bold", color=CINZA)
        ax.set_xlabel(xlabel, fontsize=9, color=CINZA)
        ax.set_ylabel(ylabel, fontsize=9, color=CINZA)
        ax.set_xticks(processos)
        ax.tick_params(colors=CINZA)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Gráfico 1: Escalabilidade Forte ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Escalabilidade Forte — {FRAMES_FORTE} frames (problema fixo)",
                 fontsize=13, fontweight="bold", color=CINZA)
    p = forte["processos"]

    ax = axes[0]
    ax.plot(p, forte["tempo"], "o-", color=AZUL, lw=2, ms=7)
    for xi, yi in zip(p, forte["tempo"]):
        ax.annotate(f"{yi:.2f}s", (xi, yi), xytext=(0,8),
                    textcoords="offset points", ha="center", fontsize=8, color=AZUL)
    estilo(ax, "Tempo de Execução", "Processos (p)", "Tempo (s)", p)

    ax = axes[1]
    ax.plot(p, p,               "--", color=CINZA,   lw=1.5, label="Ideal")
    ax.plot(p, forte["speedup"], "o-", color=LARANJA, lw=2,   ms=7, label="Medido")
    for xi, yi in zip(p, forte["speedup"]):
        ax.annotate(f"{yi:.2f}x", (xi, yi), xytext=(0,8),
                    textcoords="offset points", ha="center", fontsize=8, color=LARANJA)
    estilo(ax, "Speedup  S(p) = T(1)/T(p)", "Processos (p)", "Speedup", p)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.axhline(y=1.0, linestyle="--", color=CINZA, lw=1.5, label="Ideal")
    ax.plot(p, forte["eficiencia"], "o-", color=VERDE, lw=2, ms=7, label="Medido")
    for xi, yi in zip(p, forte["eficiencia"]):
        ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0,8),
                    textcoords="offset points", ha="center", fontsize=8, color=VERDE)
    ax.set_ylim(0, 1.3)
    estilo(ax, "Eficiência  E(p) = S(p)/p", "Processos (p)", "Eficiência", p)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "grafico_escalabilidade_forte.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] grafico_escalabilidade_forte.png salvo")

    # ── Gráfico 2: Escalabilidade Fraca ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Escalabilidade Fraca — {FRAMES_POR_PROCESSO_FRACA} frames/processo",
                 fontsize=13, fontweight="bold", color=CINZA)

    ax = axes[0]
    ax.plot(p, fraca["tempo"], "o-", color=AZUL, lw=2, ms=7)
    ax.axhline(y=fraca["tempo"][0], linestyle="--", color=CINZA, lw=1.5, label="Ideal")
    for xi, yi in zip(p, fraca["tempo"]):
        ax.annotate(f"{yi:.2f}s", (xi, yi), xytext=(0,8),
                    textcoords="offset points", ha="center", fontsize=8, color=AZUL)
    estilo(ax, "Tempo de Execução", "Processos (p)", "Tempo (s)", p)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(y=1.0, linestyle="--", color=CINZA, lw=1.5, label="Ideal")
    ax.plot(p, fraca["speedup"], "o-", color=LARANJA, lw=2, ms=7, label="Medido")
    for xi, yi in zip(p, fraca["speedup"]):
        ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0,8),
                    textcoords="offset points", ha="center", fontsize=8, color=LARANJA)
    ax.set_ylim(0, 1.5)
    estilo(ax, "Speedup Relativo  T(1)/T(p)", "Processos (p)", "Speedup", p)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.axhline(y=1.0, linestyle="--", color=CINZA, lw=1.5, label="Ideal")
    ax.plot(p, fraca["eficiencia"], "o-", color=VERDE, lw=2, ms=7, label="Medido")
    for xi, yi in zip(p, fraca["eficiencia"]):
        ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0,8),
                    textcoords="offset points", ha="center", fontsize=8, color=VERDE)
    ax.set_ylim(0, 1.3)
    estilo(ax, "Eficiência  E(p) = T(1)/T(p)", "Processos (p)", "Eficiência", p)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "grafico_escalabilidade_fraca.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] grafico_escalabilidade_fraca.png salvo")


def salvar_csv_paralelo(forte: dict, fraca: dict, output_dir: str) -> None:
    path = Path(output_dir) / "resultados_paralelo.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "processos", "frames_total", "tempo_s", "speedup", "eficiencia"])
        for i, p in enumerate(forte["processos"]):
            w.writerow(["forte", p, FRAMES_FORTE,
                        f"{forte['tempo'][i]:.4f}",
                        f"{forte['speedup'][i]:.4f}",
                        f"{forte['eficiencia'][i]:.4f}"])
        for i, p in enumerate(fraca["processos"]):
            w.writerow(["fraca", p, fraca["frames_total"][i],
                        f"{fraca['tempo'][i]:.4f}",
                        f"{fraca['speedup'][i]:.4f}",
                        f"{fraca['eficiencia'][i]:.4f}"])
    print(f"[OK] resultados_paralelo.csv salvo")



def run(video_path: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("[INFO] Carregando YOLOv8n pré-treinado (COCO)...")
    model     = YOLO("yolov8n.pt")
    clusterer = TeamClusterer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {width}x{height} | {fps:.1f} FPS | {total} frames")
    print(f"[INFO] Pré-processamento paralelo com {NUM_PROCESSOS} processos\n")

    writer = cv2.VideoWriter(
        str(Path(output_dir) / "resultado.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / FRAME_STEP, (width, height)
    )
    csv_file   = open(Path(output_dir) / "posse_por_frame.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "posse", "pct_a", "pct_b"])

    count_a, count_b = 0, 0
    frame_idx        = 0
    last_frame       = None
    batch_raw        = []   # frames brutos aguardando pré-processamento
    batch_indices    = []   # índices globais dos frames no batch

    def processar_batch(batch_raw, batch_indices):
        """Pré-processa o batch em paralelo e roda YOLO + posse em cada frame."""
        nonlocal count_a, count_b, last_frame

        batch_limpo = preprocessar_lote_paralelo(batch_raw, NUM_PROCESSOS)

        for idx_local, (frame_orig, (frame_clean, _)) in enumerate(
                zip(batch_raw, batch_limpo)):

            fidx = batch_indices[idx_local]

            results = model(frame_clean,
                            classes=[COCO_PERSON, COCO_BALL],
                            conf=CONF_THRESHOLD,
                            verbose=False)

            persons, ball_box, ball_conf = [], None, 0.0
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == COCO_BALL:
                    if conf > ball_conf:
                        ball_box, ball_conf = (x1, y1, x2, y2), conf
                else:
                    persons.append((x1, y1, x2, y2))

            team_labels = clusterer.fit_or_predict(frame_clean, persons)

            ball_center = None
            if ball_box:
                bx1, by1, bx2, by2 = ball_box
                ball_center = ((bx1+bx2)//2, (by1+by2)//2)


            players = [((x1+x2)//2, (y1+y2)//2, t)
                       for (x1,y1,x2,y2), t in zip(persons, team_labels)]
            possession = assign_possession(ball_center, players)
            if possession == "A":
                count_a += 1
            elif possession == "B":
                count_b += 1

            total_def = count_a + count_b
            pct_a = (count_a / total_def * 100) if total_def > 0 else 50.0
            pct_b = (count_b / total_def * 100) if total_def > 0 else 50.0

            csv_writer.writerow([fidx, possession or "none",
                                 f"{pct_a:.1f}", f"{pct_b:.1f}"])

            annotated  = frame_orig.copy()
            detections = []
            for (x1,y1,x2,y2), team in zip(persons, team_labels):
                name  = TEAM_A_NAME if team == 0 else TEAM_B_NAME
                color = COLOR_PLAYER_A if team == 0 else COLOR_PLAYER_B
                detections.append((x1,y1,x2,y2, name, 1.0, color))
            if ball_box:
                bx1,by1,bx2,by2 = ball_box
                detections.append((bx1,by1,bx2,by2,"Ball",ball_conf,COLOR_BALL))

            annotated = draw_boxes(annotated, detections)
            annotated = draw_possession_bar(annotated, pct_a, pct_b)
            if possession:
                name  = TEAM_A_NAME if possession == "A" else TEAM_B_NAME
                color = COLOR_PLAYER_A if possession == "A" else COLOR_PLAYER_B
                cv2.putText(annotated, f"Posse: {name}",
                            (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

            writer.write(annotated)
            last_frame = annotated.copy()

            if fidx % 200 == 0:
                print(f"[INFO] Frame {fidx}/{total} | "
                      f"{TEAM_A_NAME}: {pct_a:.1f}% X {pct_b:.1f}% {TEAM_B_NAME}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_STEP == 0:
            batch_raw.append(frame)
            batch_indices.append(frame_idx)
            if len(batch_raw) >= BATCH_SIZE:
                processar_batch(batch_raw, batch_indices)
                batch_raw.clear()
                batch_indices.clear()
        frame_idx += 1

    if batch_raw:
        processar_batch(batch_raw, batch_indices)

    cap.release()
    writer.release()
    csv_file.close()

    total_def = count_a + count_b
    pct_a = (count_a / total_def * 100) if total_def > 0 else 50.0
    pct_b = (count_b / total_def * 100) if total_def > 0 else 50.0

    if last_frame is not None:
        final = draw_possession_bar(last_frame.copy(), pct_a, pct_b)
        cv2.imwrite(str(Path(output_dir) / "placar_final.png"), final)

    print("\n" + "=" * 50)
    print(f"  RESULTADO FINAL DE POSSE")
    print(f"  {TEAM_A_NAME}: {pct_a:.1f}%  X  {pct_b:.1f}%  {TEAM_B_NAME}")
    print("=" * 50)

    # ── Testes de escalabilidade ──
    print("\n[INFO] Iniciando testes de escalabilidade paralela...")
    forte = rodar_escalabilidade_forte(video_path)
    fraca = rodar_escalabilidade_fraca(video_path)
    gerar_graficos(forte, fraca, output_dir)
    salvar_csv_paralelo(forte, fraca, output_dir)

    print(f"\n[OK] Todos os arquivos salvos em: {output_dir}/")



if __name__ == "__main__":
    multiprocessing.freeze_support()   # necessário no Windows
    run(
        video_path = VIDEO_PATH,
        output_dir = OUTPUT_DIR,
    )