import os
import librosa
from flask import Blueprint, request, jsonify
from app.services.audio import AudioService
from pydub import AudioSegment
from pydub.silence import split_on_silence
AudioController = Blueprint('audio', __name__)

@AudioController.route('/send', methods=['POST'])
def send_audio():
    libType = request.form.get('libType', None)
    typeOfDataset = request.form.get('typeOfDataset', None)
    noiseReduction = request.form.get('noiseReduction', None)
    dataType = request.form.get('dataType', None)
    audio = request.files.get('record')
    audio_path = './audio.mp3'
    if audio:
        audio.save(audio_path)

    model_path = './models/{libType}/{dataType}_{typeOfDataset}.{typeOfFile}'.format(libType=libType, dataType=dataType, typeOfDataset=typeOfDataset, typeOfFile='joblib' if libType=='scipy' else 'h5')
    if noiseReduction is not None:
        sound_duration = AudioService.reduct_noise(noiseReduction, audio_path)
    # reading from audio mp3 file
    sound = AudioSegment.from_file(audio_path)
    new_sound_duration = sound.duration_seconds
    if noiseReduction is not None and sound.duration_seconds!= sound_duration:
        velocidad_X = new_sound_duration/sound_duration
        so = sound.speedup(velocidad_X, 150, 25)
        so.export(audio_path, format = 'mp3')
    #spliting audio files
    audio_chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=sound.dBFS-16)
    results = []
    for i, chunk in enumerate(audio_chunks):
        output_file = "./audio_chunks/chunk{0}.mp3".format(i)
        chunk.export(output_file, format="mp3")
        features = AudioService.prepareAudio(output_file, dataType)
        result = AudioService.predict(features, model_path, libType)
        results.extend(result)

    return jsonify(' '.join(results)), 200
