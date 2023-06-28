import librosa
import numpy as np
import os
import padasip as pa
import joblib
from scipy.fft import fft, fftfreq, rfft, rfftfreq, irfft
from scipy.io.wavfile import write
from scipy.signal import wiener
from pydub.silence import split_on_silence

SAMPLE_RATE = 16000
WINDOW_SIZE = 0.02
HOP_SIZE = 0.01
NUM_COEFFICIENTS = 13
TEST_PERCENT=0.2

class AudioService:

    @staticmethod
    def fftNoiseReduction(noiseRecutionType, signal, sample_rate):
        N = int(sample_rate*librosa.get_duration(y=signal, sr=sample_rate))
        if noiseRecutionType == 'fft':
            yf = fft(signal)
            xf = fftfreq(N, 1 / sample_rate)
        else:
            yf = rfft(signal)
            xf = rfftfreq(N, 1 / sample_rate)
        points_per_freq = len(xf) / (sample_rate / 2)
        target_idx = int(points_per_freq * 4000)
        yf[target_idx - 1: target_idx + 2] = 0
        new_sig = irfft(yf)
        norm_new_sig = np.int16(new_sig * (32767 / new_sig.max()))
        return norm_new_sig

    @staticmethod
    def zs(a):
        a -= a.mean()
        return a / a.std()

    @staticmethod
    def adaptiveNoiseRecution(noiseRecutionType, signal, sample_rate):
        y = signal.copy()
        y = y.astype("float64")
        y = AudioService.zs(y) / 10
        N = len(y)
        n = 300 # filter size
        D = 200 # signal delay

        q = np.sin(2 * np.pi * 1000 / 99 * np.arange(N) + 10.1 * np.sin(2 * np.pi / 110 * np.arange(N)))
        d = y + q

        # prepare data for simulation
        x = pa.input_from_history(d, n)[:-D]
        d = d[n + D - 1:]
        y = y[n + D - 1:]
        q = q[n + D - 1:]

        # create filter and filter
        if noiseRecutionType == 'lms':
            f = pa.filters.FilterLMS(n=n, mu=0.01, w="random")
        elif noiseRecutionType == 'rls':
            f = pa.filters.FilterRLS(n=n, mu=0.01, w="random")
        elif noiseRecutionType == 'nlms':
            f = pa.filters.FilterNLMS(n=n, mu=0.01, w="random")

        yp, e, w = f.run(d, x)
        return e

    @staticmethod
    def wiener_filter(signal, sample_rate):
        N = int(sample_rate*librosa.get_duration(y=signal, sr=sample_rate))
        filtered_audio = wiener(signal, N)
        return filtered_audio
    
    @staticmethod
    def extract_mfcc_features(audio):
        """Extracts MFCC coefficients as features from the audio data."""
        signal, sample_rate = librosa.load(audio, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=int(WINDOW_SIZE * SAMPLE_RATE), hop_length=int(HOP_SIZE * SAMPLE_RATE), n_mfcc=NUM_COEFFICIENTS)
        mfcc = [ np.hstack([x,np.zeros(150-len(x))]) if len(x) < 150 else x[:150] for x in mfcc]
        return mfcc

    @staticmethod
    def extract_fcc_features(audio):
        signal, sample_rate = librosa.load(audio, sr=SAMPLE_RATE)
        spectrum = np.fft.fft(signal, n=SAMPLE_RATE)
        ceps = np.fft.ifft(np.log(np.abs(spectrum))).real
        return ceps

    @staticmethod
    def extract_fft_features(audio):
        signal, sample_rate = librosa.load(audio, sr=SAMPLE_RATE)
        fft = np.fft.fft(signal, n=SAMPLE_RATE)
        spectrum = np.abs(fft)
        f = np.linspace(0, sample_rate, len(spectrum))
        return f

    @staticmethod
    def extract_stft_features(audio):
        signal, sample_rate = librosa.load(audio, sr=SAMPLE_RATE)
        hop_length = 512
        n_fft = 2048
        hop_length_duration = float(hop_length)/SAMPLE_RATE
        n_fft_duration = float(n_fft)/SAMPLE_RATE
        stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
        if stft.shape[1]<40:
            stft = [list(np.append(x, np.zeros(40-len(x)))) for x in stft]
        else:
            stft = stft[:40]
        return stft

    @staticmethod
    def predictKeras(signal, modelPath):
        # model = keras.models.load_model(modelPath)
        # value = model.predict(signal)
        return 'bed'

    @staticmethod
    def predictScipy(signal, modelPath):
        model = joblib.load(modelPath)
        value = model.predict(signal)

        return valueS

    @staticmethod
    def reduct_noise(noiseRecutionType, audio):
        signal, sample_rate = librosa.load(audio, sr=SAMPLE_RATE)
        if noiseRecutionType == 'fft':
            result = AudioService.fftNoiseReduction(noiseRecutionType,signal, sample_rate)
        elif noiseRecutionType == 'rfft':
            result = AudioService.fftNoiseReduction(noiseRecutionType,signal, sample_rate)
        elif noiseRecutionType == 'wiener_filter':
            result = AudioService.wiener_filter(signal, sample_rate)
        elif noiseRecutionType == 'lms':
            result = AudioService.adaptiveNoiseRecution(noiseRecutionType, signal, sample_rate)
        elif noiseRecutionType == 'rls':
            result = AudioService.adaptiveNoiseRecution(noiseRecutionType, signal, sample_rate)
        elif noiseRecutionType == 'nlms':
            result = AudioService.adaptiveNoiseRecution(noiseRecutionType, signal, sample_rate)
        write(audio, sample_rate, result)
        return librosa.get_duration(y=signal, sr=sample_rate)
    
    @staticmethod
    def prepareAudio(audio, dataType):
        if dataType == 'mfcc':
            return AudioService.extract_mfcc_features(audio)
        elif dataType == 'fcc':
            return AudioService.extract_fcc_features(audio)
        elif dataType == 'fft':
            return AudioService.extract_fft_features(audio)
        elif dataType == 'stft':
            return AudioService.extract_stft_features(audio)

    @staticmethod
    def predict(signal, modelPath, libType):
        if libType == 'keras':
            return AudioService.predictKeras(signal, modelPath)
        elif libType == 'scipy':
            return AudioService.predictScipy(signal, modelPath)