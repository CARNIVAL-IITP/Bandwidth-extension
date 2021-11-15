from pydub import AudioSegment as am
import os

input_filepath = ''
output_filepath = ''

sound = am.from_file(input_filepath, format='wav', frame_rate=48000)
sound = sound.set_frame_rate(16000)
sound.export(output_filepath, format='wav')