"""
Script Corrigido para Gerar Gráficos do Relatório
Processamento Digital de Sinais - Prática 3
Compatível com Python 3.13 + Qualquer Taxa de Amostragem
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import subprocess
import tempfile
import sys

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# ============================================================
# FUNÇÃO 1: CARREGAR ÁUDIO (WAV ou M4A)
# ============================================================

def carregar_audio(arquivo):
    """Carrega arquivo WAV ou M4A (usando ffmpeg para M4A)"""
    
    # Verificar se arquivo existe
    if not os.path.exists(arquivo):
        print(f"\n[ERRO] Arquivo não encontrado: {arquivo}")
        print(f"\nArquivos disponíveis nesta pasta:")
        for f in os.listdir('.'):
            if f.endswith(('.wav', '.m4a', '.mp3')):
                print(f"  - {f}")
        raise FileNotFoundError(f"Arquivo não encontrado: {arquivo}")
    
    ext = os.path.splitext(arquivo)[1].lower()
    
    if ext == '.m4a':
        print(f"Convertendo {arquivo} para WAV usando ffmpeg...")
        
        # Criar arquivo temporário
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        try:
            # Converter M4A para WAV mono usando ffmpeg (mantém Fs original)
            comando = [
                'ffmpeg', '-i', arquivo,
                '-ac', '1',  # mono
                '-y',  # sobrescrever
                temp_wav_path
            ]
            
            resultado = subprocess.run(
                comando,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Carregar o WAV temporário
            fs, audio = wavfile.read(temp_wav_path)
            
            # Limpar arquivo temporário
            os.unlink(temp_wav_path)
            
        except FileNotFoundError:
            print("\n[ERRO] ffmpeg não encontrado!")
            print("Por favor, instale o ffmpeg:")
            print("  - Windows: baixe em https://ffmpeg.org/download.html")
            print("  - Linux: sudo apt-get install ffmpeg")
            print("  - Mac: brew install ffmpeg")
            print("\nOu converta seu arquivo M4A para WAV manualmente.")
            raise
        
        except subprocess.CalledProcessError as e:
            print(f"\n[ERRO] ao converter M4A: {e}")
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            raise
    
    elif ext == '.wav':
        fs, audio = wavfile.read(arquivo)
    
    else:
        raise ValueError(f"Formato não suportado: {ext}. Use .wav ou .m4a")
    
    # Processar áudio
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
    
    if audio.dtype == np.int16:
        audio = audio.astype(float) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(float) / 2147483648.0
    else:
        audio = audio.astype(float)
    
    print(f"[OK] Áudio carregado: {len(audio)} amostras, {fs} Hz, {len(audio)/fs:.2f}s")
    return audio, fs


# ============================================================
# FUNÇÃO 2: PROJETO FILTRO FIR
# ============================================================

def projetar_fir(fs, fp1, fr1, fr2, fp2, janela='hamming'):
    """Projeta filtro FIR por janelamento"""
    df = fr1 - fp1
    delta_omega = (2 * np.pi * df) / fs
    N = int(np.ceil(3.14 * np.pi / delta_omega))
    if N % 2 == 0:
        N += 1
    M = (N - 1) // 2
    
    # Resposta ideal bandstop
    wc1 = (2 * np.pi * fr1) / fs
    wc2 = (2 * np.pi * fr2) / fs
    n = np.arange(N) - M
    
    h_ideal = np.zeros(N)
    for i, ns in enumerate(n):
        if ns == 0:
            h_ideal[i] = 1 - (wc2 - wc1) / np.pi
        else:
            h_ideal[i] = (np.sin(wc1 * ns) - np.sin(wc2 * ns)) / (np.pi * ns)
    
    # Aplicar janela
    if janela == 'hamming':
        w = np.hamming(N)
    elif janela == 'hanning':
        w = np.hanning(N)
    elif janela == 'blackman':
        w = np.blackman(N)
    else:
        w = np.ones(N)
    
    h = h_ideal * w
    
    print(f"[OK] FIR: N={N}, M={M}, Janela={janela}")
    return h, N, M


# ============================================================
# FUNÇÃO 3: PROJETO FILTRO IIR
# ============================================================

def projetar_iir(fs, fp1, fs1, fs2, fp2, rp=1, rs=40):
    """Projeta filtro IIR Butterworth"""
    nyquist = fs / 2
    
    # Normalizar frequências
    wp = [fp1/nyquist, fp2/nyquist]
    ws = [fs1/nyquist, fs2/nyquist]
    
    # Verificar limites
    if any(f >= 1 or f <= 0 for f in wp + ws):
        raise ValueError(f"Frequências inválidas! Nyquist={nyquist}Hz. "
                        f"Fp1={fp1}, Fs1={fs1}, Fs2={fs2}, Fp2={fp2}")
    
    # Calcular ordem e coeficientes
    N, wn = signal.buttord(wp, ws, rp, rs)
    b, a = signal.butter(N, wn, btype='bandstop')
    
    # Análise de polos e zeros
    z, p, k = signal.tf2zpk(b, a)
    max_polo = np.max(np.abs(p))
    
    # VERIFICAR ESTABILIDADE
    if max_polo >= 1.0:
        print(f"\n[AVISO] Filtro IIR potencialmente INSTÁVEL!")
        print(f"        |p|max = {max_polo:.4f} >= 1.0")
        print(f"        Ajustando especificações...")
        
        # Tentar com especificações mais relaxadas
        N, wn = signal.buttord(wp, ws, rp, rs-10)
        b, a = signal.butter(N, wn, btype='bandstop')
        z, p, k = signal.tf2zpk(b, a)
        max_polo = np.max(np.abs(p))
        
        if max_polo >= 1.0:
            print(f"[ERRO] Ainda instável após ajuste: |p|max={max_polo:.4f}")
            print(f"       Usando filtfilt que compensa instabilidade marginal")
    
    print(f"[OK] IIR: N={N}, Polos={len(p)}, Zeros={len(z)}, |p|max={max_polo:.4f}")
    return b, a, N, z, p


# ============================================================
# GRÁFICO 1: ANÁLISE DO ÁUDIO (Tempo + FFT)
# ============================================================

def plotar_analise_audio(audio, fs, salvar='grafico1_analise_audio.png'):
    """Gráfico 1: Sinal no tempo + Espectro"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Tempo
    t = np.arange(len(audio)) / fs
    ax1.plot(t, audio, 'b-', linewidth=0.5)
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Sinal de Áudio Original com Ruído - Domínio do Tempo')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: FFT
    N = len(audio)
    fft_vals = np.fft.fft(audio)
    fft_freq = np.fft.fftfreq(N, 1/fs)
    
    pos_idx = fft_freq >= 0
    fft_freq_pos = fft_freq[pos_idx]
    fft_mag = np.abs(fft_vals[pos_idx])
    
    ax2.plot(fft_freq_pos, fft_mag, 'r-', linewidth=1)
    ax2.set_xlabel('Frequência (Hz)')
    ax2.set_ylabel('|FFT|')
    ax2.set_title('Espectro de Frequência - Identificar Ruído')
    ax2.set_xlim(0, fs/2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(salvar, dpi=300, bbox_inches='tight')
    print(f"[OK] Salvo: {salvar}")
    plt.close()


# ============================================================
# GRÁFICO 2: RESPOSTA FIR (Magnitude + Fase)
# ============================================================

def plotar_resposta_fir(h, fs, fr1, fr2, salvar='grafico2_resposta_fir.png'):
    """Gráfico 2: Resposta em frequência FIR"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Calcular resposta
    w, H = signal.freqz(h, worN=8000)
    freq = w * fs / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(H) + 1e-10)
    fase = np.angle(H)
    
    # Magnitude
    ax1.plot(freq, mag_db, 'b-', linewidth=2, label='Filtro FIR')
    ax1.axvline(fr1, color='r', linestyle='--', alpha=0.7, label=f'Fr1={fr1}Hz')
    ax1.axvline(fr2, color='r', linestyle='--', alpha=0.7, label=f'Fr2={fr2}Hz')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Resposta em Frequência - Filtro FIR')
    ax1.set_xlim(0, fs/2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Fase
    ax2.plot(freq, fase, 'g-', linewidth=2)
    ax2.set_xlabel('Frequência (Hz)')
    ax2.set_ylabel('Fase (rad)')
    ax2.set_title('Fase - Filtro FIR (Linear)')
    ax2.set_xlim(0, fs/2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(salvar, dpi=300, bbox_inches='tight')
    print(f"[OK] Salvo: {salvar}")
    plt.close()


# ============================================================
# GRÁFICO 3: COEFICIENTES FIR
# ============================================================

def plotar_coeficientes_fir(h, salvar='grafico3_coeficientes_fir.png'):
    """Gráfico 3: Coeficientes do filtro FIR"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    markerline, stemlines, baseline = ax.stem(h, linefmt='b-', markerfmt='bo', basefmt='k-')
    plt.setp(stemlines, linewidth=1.5)
    plt.setp(markerline, markersize=4)
    
    ax.set_xlabel('n (amostras)')
    ax.set_ylabel('h[n]')
    ax.set_title(f'Coeficientes do Filtro FIR (N={len(h)})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(salvar, dpi=300, bbox_inches='tight')
    print(f"[OK] Salvo: {salvar}")
    plt.close()


# ============================================================
# GRÁFICO 4: RESPOSTA IIR (Magnitude + Fase)
# ============================================================

def plotar_resposta_iir(b, a, fs, fs1, fs2, salvar='grafico4_resposta_iir.png'):
    """Gráfico 4: Resposta em frequência IIR"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Calcular resposta
    w, H = signal.freqz(b, a, worN=8000)
    freq = w * fs / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(H) + 1e-10)
    fase = np.angle(H)
    
    # Magnitude
    ax1.plot(freq, mag_db, 'r-', linewidth=2, label='Filtro IIR')
    ax1.axvline(fs1, color='b', linestyle='--', alpha=0.7, label=f'Fs1={fs1}Hz')
    ax1.axvline(fs2, color='b', linestyle='--', alpha=0.7, label=f'Fs2={fs2}Hz')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Resposta em Frequência - Filtro IIR (Butterworth)')
    ax1.set_xlim(0, fs/2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Fase
    ax2.plot(freq, fase, 'orange', linewidth=2)
    ax2.set_xlabel('Frequência (Hz)')
    ax2.set_ylabel('Fase (rad)')
    ax2.set_title('Fase - Filtro IIR (Não-linear)')
    ax2.set_xlim(0, fs/2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(salvar, dpi=300, bbox_inches='tight')
    print(f"[OK] Salvo: {salvar}")
    plt.close()


# ============================================================
# GRÁFICO 5: POLOS E ZEROS (IIR)
# ============================================================

def plotar_polos_zeros(z, p, salvar='grafico5_polos_zeros.png'):
    """Gráfico 5: Diagrama de polos e zeros"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Círculo unitário
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=1.5, label='Círculo Unitário')
    
    # Zeros e polos
    ax.plot(np.real(z), np.imag(z), 'bo', markersize=10, 
            markerfacecolor='none', markeredgewidth=2, label=f'Zeros ({len(z)})')
    ax.plot(np.real(p), np.imag(p), 'rx', markersize=10, 
            markeredgewidth=2, label=f'Polos ({len(p)})')
    
    # Eixos
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    
    max_polo = np.max(np.abs(p))
    estavel = "ESTÁVEL" if max_polo < 1 else "INSTÁVEL"
    
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginário')
    ax.set_title(f'Diagrama de Polos e Zeros - {estavel} (|p|max={max_polo:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(salvar, dpi=300, bbox_inches='tight')
    print(f"[OK] Salvo: {salvar}")
    plt.close()


# ============================================================
# GRÁFICO 6: COMPARAÇÃO TEMPORAL (Original vs Filtrado)
# ============================================================

def plotar_comparacao_temporal(original, filtrado, fs, tipo_filtro, salvar=None):
    """Gráfico 6: Comparação no tempo"""
    if salvar is None:
        salvar = f'grafico6_comparacao_temporal_{tipo_filtro}.png'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotar apenas 2 segundos para visualização
    max_samples = min(int(2 * fs), len(original))
    t = np.arange(max_samples) / fs
    
    ax.plot(t, original[:max_samples], 'r-', alpha=0.6, linewidth=1, label='Original (com ruído)')
    ax.plot(t, filtrado[:max_samples], 'b-', linewidth=1.5, label=f'Filtrado ({tipo_filtro})')
    
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Comparação Temporal - Filtro {tipo_filtro}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(salvar, dpi=300, bbox_inches='tight')
    print(f"[OK] Salvo: {salvar}")
    plt.close()


# ============================================================
# GRÁFICO 7: COMPARAÇÃO ESPECTRAL (Original vs Filtrado)
# ============================================================

def plotar_comparacao_espectral(original, filtrado, fs, fr1, fr2, tipo_filtro, salvar=None):
    """Gráfico 7: Comparação espectral"""
    if salvar is None:
        salvar = f'grafico7_comparacao_espectral_{tipo_filtro}.png'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # FFT original
    fft_orig = np.fft.fft(original)
    freq_orig = np.fft.fftfreq(len(original), 1/fs)
    pos_idx = freq_orig >= 0
    freq_pos = freq_orig[pos_idx]
    mag_orig = np.abs(fft_orig[pos_idx])
    
    # FFT filtrado
    fft_filt = np.fft.fft(filtrado)
    mag_filt = np.abs(fft_filt[pos_idx])
    
    # Plotar
    ax.plot(freq_pos, mag_orig, 'r-', alpha=0.7, linewidth=1.5, label='Original (com ruído)')
    ax.plot(freq_pos, mag_filt, 'b-', linewidth=2, label=f'Filtrado ({tipo_filtro})')
    ax.axvspan(fr1, fr2, alpha=0.2, color='orange', label='Banda de Rejeição')
    
    ax.set_xlabel('Frequência (Hz)')
    ax.set_ylabel('|FFT|')
    ax.set_title(f'Comparação Espectral - Filtro {tipo_filtro}')
    ax.set_xlim(0, fs/2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(salvar, dpi=300, bbox_inches='tight')
    print(f"[OK] Salvo: {salvar}")
    plt.close()


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():
    """Execução principal - Gera todos os gráficos"""
    
    print("=" * 60)
    print("GERADOR DE GRÁFICOS PARA RELATÓRIO - PRÁTICA 3 PDS")
    print("Compatível com Python 3.13 + Qualquer Taxa de Amostragem")
    print("=" * 60)
    
    # ========== CONFIGURAÇÕES BÁSICAS ==========
    
    # *** ALTERE ESTA LINHA COM O NOME DO SEU ARQUIVO ***
    arquivo_audio = "audio_com_ruido_1550Hz.wav"  # <--- SEU ARQUIVO
    
    janela = 'hamming'  # Tipo de janela FIR
    # ===========================================
    
    # Carregar áudio
    print(f"\n[PASSO 1] Carregando áudio: {arquivo_audio}")
    try:
        audio, fs = carregar_audio(arquivo_audio)
    except Exception as e:
        print(f"\n[ERRO] ao carregar áudio: {e}")
        return
    
    # ========== CONFIGURAÇÃO AUTOMÁTICA DAS FREQUÊNCIAS ==========
    # Ajusta as frequências do filtro com base na taxa de amostragem
    
    # Para Fs = 8000 Hz:  ruído em ~1585 Hz
    # Para Fs = 48000 Hz: ruído em ~9500 Hz (proporcional)
    
    fator = fs / 8000  # Fator de escala
    
    print(f"\n[PASSO 2] Configurando frequências do filtro (Fs={fs}Hz, fator={fator:.1f}x)")
    
    # Frequências base para 8kHz
    fp1_base = 1500
    fr1_base = 1550
    fr2_base = 1620
    fp2_base = 1670
    
    # Escalar para a taxa de amostragem atual
    fp1 = int(fp1_base * fator)
    fr1 = int(fr1_base * fator)
    fr2 = int(fr2_base * fator)
    fp2 = int(fp2_base * fator)
    
    # IIR (mesmos valores)
    fs1 = fr1
    fs2 = fr2
    rp = 1
    rs = 40
    
    print(f"  Frequências ajustadas:")
    print(f"    Fp1 = {fp1} Hz (passagem inferior)")
    print(f"    Fr1 = {fr1} Hz (rejeição inferior)")
    print(f"    Fr2 = {fr2} Hz (rejeição superior)")
    print(f"    Fp2 = {fp2} Hz (passagem superior)")
    print(f"  Banda de rejeição: {fr1}-{fr2} Hz")
    
    # GRÁFICO 1: Análise do áudio
    print("\n[PASSO 3] Gerando Gráfico 1: Análise do Áudio...")
    plotar_analise_audio(audio, fs)
    
    # Projetar FIR
    print(f"\n[PASSO 4] Projetando Filtro FIR...")
    h, N_fir, M = projetar_fir(fs, fp1, fr1, fr2, fp2, janela)
    
    # GRÁFICO 2: Resposta FIR
    print("[PASSO 5] Gerando Gráfico 2: Resposta FIR...")
    plotar_resposta_fir(h, fs, fr1, fr2)
    
    # GRÁFICO 3: Coeficientes FIR
    print("[PASSO 6] Gerando Gráfico 3: Coeficientes FIR...")
    plotar_coeficientes_fir(h)
    
    # Aplicar FIR
    print(f"\n[PASSO 7] Aplicando Filtro FIR...")
    audio_fir = signal.filtfilt(h, [1], audio)
    
    # GRÁFICO 6: Comparação temporal FIR
    print("[PASSO 8] Gerando Gráfico 6: Comparação Temporal FIR...")
    plotar_comparacao_temporal(audio, audio_fir, fs, 'FIR')
    
    # GRÁFICO 7: Comparação espectral FIR
    print("[PASSO 9] Gerando Gráfico 7: Comparação Espectral FIR...")
    plotar_comparacao_espectral(audio, audio_fir, fs, fr1, fr2, 'FIR')
    
    # Projetar IIR
    print(f"\n[PASSO 10] Projetando Filtro IIR...")
    try:
        b, a, N_iir, z, p = projetar_iir(fs, fp1, fs1, fs2, fp2, rp, rs)
    except Exception as e:
        print(f"[ERRO] ao projetar IIR: {e}")
        print("Tentando com especificações mais relaxadas...")
        b, a, N_iir, z, p = projetar_iir(fs, fp1, fs1, fs2, fp2, rp, 30)
    
    # GRÁFICO 4: Resposta IIR
    print("[PASSO 11] Gerando Gráfico 4: Resposta IIR...")
    plotar_resposta_iir(b, a, fs, fs1, fs2)
    
    # GRÁFICO 5: Polos e Zeros
    print("[PASSO 12] Gerando Gráfico 5: Polos e Zeros...")
    plotar_polos_zeros(z, p)
    
    # Aplicar IIR
    print(f"\n[PASSO 13] Aplicando Filtro IIR (com filtfilt para fase zero)...")
    try:
        audio_iir = signal.filtfilt(b, a, audio)
    except Exception as e:
        print(f"[AVISO] Erro ao aplicar filtfilt: {e}")
        print("         Tentando método alternativo (lfilter)...")
        audio_iir = signal.lfilter(b, a, audio)
    
    # GRÁFICO 6: Comparação temporal IIR
    print("[PASSO 14] Gerando Gráfico 6: Comparação Temporal IIR...")
    plotar_comparacao_temporal(audio, audio_iir, fs, 'IIR')
    
    # GRÁFICO 7: Comparação espectral IIR
    print("[PASSO 15] Gerando Gráfico 7: Comparação Espectral IIR...")
    plotar_comparacao_espectral(audio, audio_iir, fs, fr1, fr2, 'IIR')
    
    # Salvar áudios filtrados
    print("\n[PASSO 16] Salvando áudios filtrados...")
    
    # Normalizar e converter para int16
    audio_fir_norm = np.clip(audio_fir, -1, 1)
    audio_iir_norm = np.clip(audio_iir, -1, 1)
    
    audio_fir_int = (audio_fir_norm * 32767).astype(np.int16)
    audio_iir_int = (audio_iir_norm * 32767).astype(np.int16)
    
    wavfile.write('audio_filtrado_FIR.wav', fs, audio_fir_int)
    wavfile.write('audio_filtrado_IIR.wav', fs, audio_iir_int)
    print("[OK] Salvo: audio_filtrado_FIR.wav")
    print("[OK] Salvo: audio_filtrado_IIR.wav")
    
    print("\n" + "=" * 60)
    print("[SUCESSO] Todos os gráficos foram gerados!")
    print("=" * 60)
    print(f"\nRESUMO:")
    print(f"  Taxa de amostragem: {fs} Hz")
    print(f"  Banda de rejeição: {fr1}-{fr2} Hz")
    print(f"  Filtro FIR: Ordem N={N_fir}, M={M}")
    print(f"  Filtro IIR: Ordem N={N_iir}")
    print(f"\nARQUIVOS GERADOS:")
    for i in range(1, 8):
        if i <= 5:
            print(f"  - grafico{i}_*.png")
    print(f"  - grafico6_comparacao_temporal_FIR.png")
    print(f"  - grafico6_comparacao_temporal_IIR.png")
    print(f"  - grafico7_comparacao_espectral_FIR.png")
    print(f"  - grafico7_comparacao_espectral_IIR.png")
    print(f"  - audio_filtrado_FIR.wav")
    print(f"  - audio_filtrado_IIR.wav")
    print("\n[DICA] Abra os arquivos PNG para ver os gráficos!")


if __name__ == "__main__":
    main()
    