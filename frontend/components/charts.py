import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

# ── Light Tessan-inspired chart palette ──────────────────────────
_LIGHT_BG = "#FFFFFF"
_PAGE_BG = "#F6F4EE"
_TEXT = "#0C4B43"
_TEXT_SOFT = "#42675F"
_TEXT_MUTED = "#6F8C85"
_BORDER = "#D7E3DC"
_ACCENT = "#0F5A50"
_ACCENT_2 = "#E7E56A"


def _apply_light_style(fig, ax):
    fig.patch.set_facecolor(_LIGHT_BG)
    ax.set_facecolor(_LIGHT_BG)

    ax.tick_params(colors=_TEXT_SOFT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)

    for spine in ax.spines.values():
        spine.set_color(_BORDER)
        spine.set_alpha(0.9)


def waveform_chart(audio, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    _apply_light_style(fig, ax)

    time = np.linspace(0, len(audio) / sr, num=len(audio))
    ax.plot(time, audio, color=_ACCENT, linewidth=0.8, alpha=0.95)

    ax.set_title("Signal audio", fontsize=12, pad=10)
    ax.set_xlabel("Temps (s)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_xlim(0, time[-1])

    ax.grid(axis="y", color=_BORDER, alpha=0.35, linewidth=0.8)

    fig.tight_layout()
    return fig


def mel_spectrogram(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3))
    _apply_light_style(fig, ax)

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma",   # softer on light UI than inferno
    )

    ax.set_title("Mel-spectrogramme", fontsize=12, pad=10)

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    cbar.outline.set_edgecolor(_BORDER)
    cbar.ax.tick_params(colors=_TEXT_SOFT, labelsize=8)

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════
# DOCTOR charts (light / white background)
# ════════════════════════════════════════════

def waveform_chart_doc(color=_ACCENT, seed=0):
    """Generate a simulated waveform for the doctor UI (demo data)."""
    rng = np.random.RandomState(seed)
    fig, ax = plt.subplots(figsize=(8, 1.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    pts = 400
    x = np.linspace(0, 4.8, pts)
    y = (
        rng.randn(pts) * 0.3
        + np.sin(x * 12) * 0.15
        + np.sin(x * 5) * 0.1
    )

    ax.plot(x, y, color=color, linewidth=0.8, alpha=0.9)
    ax.set_xlim(0, 4.8)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def mel_spectrogram_doc(seed=0):
    """Generate a simulated mel spectrogram for the doctor UI (demo data)."""
    rng = np.random.RandomState(seed + 7)
    fig, ax = plt.subplots(figsize=(8, 2.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    rows, cols = 40, 120
    data = np.zeros((rows, cols))
    for c in range(cols):
        for r in range(rows):
            silence = (25 < c < 32) or (80 < c < 90)
            base = 0.05 if silence else (1 - r / rows) * 0.8 + rng.random() * 0.2
            data[r, c] = max(0, min(1, base - r * 0.008))

    ax.imshow(data, aspect="auto", cmap="magma", interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def radar_chart(data):
    """Radar / spider chart for disease probability profile."""
    labels = list(data.keys())
    values = list(data.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values += values[:1]
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(4, 4))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("white")

    ax.plot(angles, values, color=_ACCENT, linewidth=2)
    ax.fill(angles, values, color=_ACCENT, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=_TEXT_SOFT, fontfamily="sans-serif")

    ax.set_ylim(0, 70)
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels([])

    ax.spines["polar"].set_color(_BORDER)
    ax.grid(color=_BORDER, alpha=0.5)

    fig.tight_layout()
    return fig