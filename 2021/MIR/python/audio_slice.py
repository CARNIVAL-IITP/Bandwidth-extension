from pydub import AudioSegment
from pydub.utils import make_chunks


folder = './impulse_input/anechoic/fabric/10_5_44.1khz/'
filename = 'fabric_10_5_44100[10_22000].wav'

file = folder + filename


myaudio = AudioSegment.from_file(file , "wav") 
chunk_length_ms = 15000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

for i, chunk in enumerate(chunks):
    # chunk_name = "chunk{0}.wav".format(i) 
    chunk_name = str(i) + '_'+ filename
    print("exporting", chunk_name)
    chunk.export(folder + chunk_name, format="wav")

print('All exported successfully')