"""
Gerador de Áudio Sintético com Ruído Tonal em 1550 Hz
Para garantir que o filtro funcione perfeitamente!
"""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sys

# Configurar encoding UTF-8 para o console do Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("GERADOR DE ÁUDIO COM RUÍDO CONTROLADO")
print("=" * 60)

# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Opção 1: Gerar áudio sintético puro
MODO = "sintetico"  # "sintetico" ou "carregar_existente"

# Se carregar_existente, especifique o arquivo
arquivo_voz = "gravando.wav"  # Sua gravação de voz

# Parâmetros do ruído
FREQUENCIA_RUIDO = 1550  # Hz - PERFEITO para o filtro!
VOLUME_RUIDO = 0.3       # 30% do volume da voz
FS = 8000                # Taxa de amostragem (8kHz = especificação)
DURACAO = 5              # segundos

# ============================================================
# FUNÇÃO 1: GERAR VOZ SINTÉTICA
# ============================================================

def gerar_voz_sintetica(fs, duracao):
    """Gera sinal que simula voz humana (100-1200 Hz)"""
    t = np.arange(0, duracao, 1/fs)
    
    # Fundamental da voz (120 Hz - voz masculina típica)
    fundamental = 120
    voz = 0.5 * np.sin(2 * np.pi * fundamental * t)
    
    # Adicionar harmônicos (formantes da voz)
    # Formantes típicos: 500Hz, 900Hz, 1200Hz
    voz += 0.3 * np.sin(2 * np.pi * 500 * t)
    voz += 0.2 * np.sin(2 * np.pi * 900 * t)
    voz += 0.15 * np.sin(2 * np.pi * 1200 * t)
    
    # Envoltória (simular sílabas)
    envelope = np.ones(len(t))
    silaba_duracao = int(0.3 * fs)  # 300ms por sílaba
    
    for i in range(0, len(t), silaba_duracao):
        # Ataque suave
        ataque = int(0.05 * fs)
        if i + ataque < len(envelope):
            envelope[i:i+ataque] = np.linspace(0, 1, ataque)
        
        # Decay ao final da sílaba
        decay = int(0.1 * fs)
        if i + silaba_duracao - decay < len(envelope):
            envelope[i+silaba_duracao-decay:i+silaba_duracao] = \
                np.linspace(1, 0.3, decay)
    
    voz = voz * envelope
    
    # Normalizar
    voz = voz / np.max(np.abs(voz))
    
    return voz, t

# ============================================================
# FUNÇÃO 2: GERAR RUÍDO TONAL
# ============================================================

def gerar_ruido_tonal(fs, duracao, frequencia, amplitude=0.3):
    """Gera tom senoidal puro"""
    t = np.arange(0, duracao, 1/fs)
    ruido = amplitude * np.sin(2 * np.pi * frequencia * t)
    return ruido

# ============================================================
# GERAÇÃO DO ÁUDIO
# ============================================================

print(f"\n[MODO] {MODO}")
print(f"[CONFIG] Fs={FS}Hz, Duração={DURACAO}s, Ruído={FREQUENCIA_RUIDO}Hz")

if MODO == "sintetico":
    print("\n[1/5] Gerando voz sintética...")
    voz, t = gerar_voz_sintetica(FS, DURACAO)
    print(f"  ✓ Voz gerada: {len(voz)} amostras")
    
elif MODO == "carregar_existente":
    print(f"\n[1/5] Carregando sua gravação: {arquivo_voz}")
    try:
        fs_orig, voz_orig = wavfile.read(arquivo_voz)
        
        # Converter para mono
        if len(voz_orig.shape) == 2:
            voz_orig = np.mean(voz_orig, axis=1)
        
        # Normalizar
        if voz_orig.dtype == np.int16:
            voz_orig = voz_orig.astype(float) / 32768.0
        
        # Reamostrar para 8kHz se necessário
        if fs_orig != FS:
            print(f"  [AVISO] Reamostrando de {fs_orig}Hz para {FS}Hz...")
            from scipy import signal as sp_signal
            num_samples = int(len(voz_orig) * FS / fs_orig)
            voz = sp_signal.resample(voz_orig, num_samples)
        else:
            voz = voz_orig
        
        # Ajustar duração
        samples_desejados = int(DURACAO * FS)
        if len(voz) > samples_desejados:
            voz = voz[:samples_desejados]
        elif len(voz) < samples_desejados:
            voz = np.pad(voz, (0, samples_desejados - len(voz)))
        
        t = np.arange(len(voz)) / FS
        print(f"  ✓ Voz carregada: {len(voz)} amostras")
        
    except Exception as e:
        print(f"  ✗ ERRO ao carregar: {e}")
        print("  Mudando para modo sintético...")
        voz, t = gerar_voz_sintetica(FS, DURACAO)

# Gerar ruído
print(f"\n[2/5] Gerando ruído tonal em {FREQUENCIA_RUIDO} Hz...")
ruido = gerar_ruido_tonal(FS, DURACAO, FREQUENCIA_RUIDO, VOLUME_RUIDO)
print(f"  ✓ Ruído gerado")

# Mixar voz + ruído
print("\n[3/5] Mixando voz + ruído...")
audio_com_ruido = voz + ruido

# Normalizar para evitar clipping
max_val = np.max(np.abs(audio_com_ruido))
if max_val > 1.0:
    audio_com_ruido = audio_com_ruido / max_val * 0.95

print(f"  ✓ Áudio mixado")

# ============================================================
# SALVAR ARQUIVO
# ============================================================

print("\n[4/5] Salvando arquivo WAV...")
audio_int16 = (audio_com_ruido * 32767).astype(np.int16)
wavfile.write('audio_com_ruido_1550Hz.wav', FS, audio_int16)
print("  ✓ Salvo: audio_com_ruido_1550Hz.wav")

# ============================================================
# VISUALIZAÇÃO
# ============================================================

print("\n[5/5] Gerando visualização...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Tempo
ax1.plot(t, audio_com_ruido, 'b-', linewidth=0.5)
ax1.set_xlabel('Tempo (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title(f'Áudio Sintético com Ruído em {FREQUENCIA_RUIDO} Hz')
ax1.grid(True, alpha=0.3)

# Plot 2: FFT
N = len(audio_com_ruido)
fft_vals = np.fft.fft(audio_com_ruido)
fft_freq = np.fft.fftfreq(N, 1/FS)

pos_idx = fft_freq >= 0
fft_freq_pos = fft_freq[pos_idx]
fft_mag = np.abs(fft_vals[pos_idx])

ax2.plot(fft_freq_pos, fft_mag, 'r-', linewidth=1)
ax2.axvline(FREQUENCIA_RUIDO, color='green', linestyle='--', linewidth=2, 
            alpha=0.7, label=f'Ruído: {FREQUENCIA_RUIDO} Hz')
ax2.set_xlabel('Frequência (Hz)')
ax2.set_ylabel('|FFT|')
ax2.set_title('Espectro - Ruído Claramente Visível!')
ax2.set_xlim(0, FS/2)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('preview_audio_ruido.png', dpi=300, bbox_inches='tight')
print("  ✓ Salvo: preview_audio_ruido.png")

print("\n" + "=" * 60)
print("✓ CONCLUÍDO!")
print("=" * 60)
print(f"\nARQUIVO GERADO:")
print(f"  - audio_com_ruido_1550Hz.wav ({DURACAO}s, {FS}Hz)")
print(f"  - preview_audio_ruido.png (visualização)")
print(f"\nCARACTERÍSTICAS:")
print(f"  Voz: 100-1200 Hz (formantes naturais)")
print(f"  Ruído: {FREQUENCIA_RUIDO} Hz (tom puro)")
print(f"  Volume ruído: {VOLUME_RUIDO*100:.0f}% da voz")
print(f"\n🎯 AGORA SIM O FILTRO VAI FUNCIONAR PERFEITAMENTE!")
print(f"   Execute o script principal com: 'audio_com_ruido_1550Hz.wav'")