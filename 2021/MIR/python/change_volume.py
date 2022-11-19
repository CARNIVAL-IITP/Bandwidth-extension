import os
from os import makedirs

from natsort import natsorted
from pydub import AudioSegment
from tqdm import tqdm


input_clean_path = '' 
output_filepath = '' 

makedirs(output_filepath, exist_ok = True)

if __name__ == '__main__':
    hr_files = os.listdir(input_clean_path)

    hr_files = natsorted(hr_files)
    # hr_files.sort()
    
    hr_file_list = []
    for hr_file in hr_files:
        hr_file_list.append(input_clean_path + hr_file)

    for i in tqdm(range(len(hr_file_list))):
        song = AudioSegment.from_wav(hr_file_list[i])

        # boost volume by 6dB
        louder_song = song + 6

        # reduce volume by 3dB
        quieter_song = song - 12

        head, tail = os.path.split(hr_file_list[i])
        tail = tail.split('.wav')[0]

        outputfile = output_filepath +  tail + '_quieter' + '.wav'
        #save quieter song 
        quieter_song.export(outputfile, format='wav')