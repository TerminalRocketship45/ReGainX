import os
import numpy as np
import PIL.Image
import PIL.ImageDraw
import skvideo
import skvideo.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FFMPEG_PATH = os.environ.get(
    "FFMPEG_PATH",
    r"C:\Users\rohan\Downloads\ffmpeg-2025-09-04-git-2611874a50-essentials_build"
    r"\ffmpeg-2025-09-04-git-2611874a50-essentials_build\bin",
)
skvideo.setFFmpegPath(FFMPEG_PATH)


def add_text_to_frame(frame: np.ndarray, text: str,
                      pos=(10, 5), color=(255, 255, 255)) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    img = PIL.Image.fromarray(frame)
    PIL.ImageDraw.Draw(img).text(pos, text, fill=color)
    return np.asarray(img)


def render_frame(base_env, step: int, label: str = "") -> np.ndarray:
    frame = base_env.unwrapped.sim.renderer.render_offscreen(
        width=400, height=400, camera_id=0
    )
    t = step * base_env.unwrapped.dt
    overlay = f"{label} t={int(t // 60):02d}:{int(t % 60):02d}"
    return add_text_to_frame(frame, overlay)


def save_video(frames: list, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    arr = np.asarray(frames, dtype=np.float32)
    if arr.max() <= 1.0 and arr.min() >= 0.0 and arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    skvideo.io.vwrite(
        path, arr,
        outputdict={"-pix_fmt": "yuv420p"},
    )


def compute_severity(force_scale: float,
                     activation_slowdown: float,
                     avg_mf: float) -> float:
    """Normalized [0, 1] severity; avg_mf must be pre-normalized to [0, 1] (0=no fatigue, 1=fully fatigued)."""
    f = (0.9 - force_scale) / 0.3          # [0.9, 0.6] -> [0, 1] (lower scale = more impaired)
    s = (activation_slowdown - 1.1) / 0.3  # [1.1, 1.4] -> [0, 1]
    return float(np.clip((f + s + avg_mf) / 3.0, 0.0, 1.0))


def get_angle_bin(target_angle: float, edges: np.ndarray) -> int:
    """0-indexed bin; clips to valid range."""
    return int(np.clip(np.searchsorted(edges[1:], target_angle), 0, len(edges) - 2))


def get_severity_quartile(severity: float, edges: np.ndarray) -> int:
    """0-indexed quartile (0=Q1 mild, 3=Q4 severe); clips to valid range."""
    return int(np.clip(np.searchsorted(edges[1:], severity), 0, len(edges) - 2))


def plot_confusion_matrix(
    matrix: np.ndarray,
    angle_labels: list,
    severity_labels: list,
    title: str,
    save_path: str,
) -> None:
    """
    Blues confusion matrix: rows=angle bins, cols=severity quartiles,
    cells=Pearson correlation (NaN shown as N/A).
    Light blue = low correlation, dark blue = high.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    display = np.nan_to_num(matrix, nan=0.0)
    im = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(severity_labels)))
    ax.set_xticklabels(severity_labels)
    ax.set_yticks(range(len(angle_labels)))
    ax.set_yticklabels(angle_labels)
    ax.set_xlabel("Severity Quartile")
    ax.set_ylabel("Target Angle (rad)")
    ax.set_title(title)

    for i in range(len(angle_labels)):
        for j in range(len(severity_labels)):
            val = matrix[i, j]
            text = "N/A" if np.isnan(val) else f"{val:.2f}"
            color = "white" if display[i, j] > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Pearson r (higher = closer to healthy)")
    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
