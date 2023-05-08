import moviepy.editor as mp
import os
import pandas as pd
import traceback
import speech_recognition as sr
import sys
from pydub import AudioSegment
from pydub.silence import split_on_silence
from os.path import exists
from spacy_embeddings import SpacySimilarity
from cosine_similarity import calculate_cosine_similarity
# from tkinter.filedialog import *

# Initialize recognizer class (for recognizing the speech)
recognizer = sr.Recognizer()
spacy_similarity = SpacySimilarity()
file_path_separator = '/' if os.name == 'posix' else '\\'
base_dir = 'test_dataset' + file_path_separator

def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
                              # experiment with this value for your target audio file
                              min_silence_len=500,
                              # adjust this per requirement
                              silence_thresh=sound.dBFS-14,
                              # keep the silence for 1 second, adjustable as well
                              keep_silence=500,
                              )
    folder_name = base_dir + "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    # just for showing the progress bar
    count = len(chunks)
    size = 30

    def show(j):
        x = int(size*j/count)
        sys.stdout.write("%s[%s%s] %i/%i\r" %
                         (path, "#"*x, "."*(size-x), j, count))
        sys.stdout.flush()
    show(0)

    #####
    whole_text = ""
    # process each chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = recognizer.record(source)
            # try converting it to text
            try:
                text = recognizer.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                # traceback.print_exc()
                # os.remove(chunk_filename)
                show(i)
            else:
                text = f"{text.capitalize()}. "
                whole_text += text
                # os.remove(chunk_filename)
                show(i)
    sys.stdout.write("\n")
    sys.stdout.flush()
    # return the text for all chunks detected
    # os.rmdir(folder_name)
    return whole_text

def get_audio_transcription_using_google(audio_file_path):
    with sr.AudioFile(audio_file_path) as source:
        # Listening the audio file and store in audio_text variable
        audio_text = recognizer.record(source)

        try:
            # Convering audio to text
            text = recognizer.recognize_google(audio_text)

        except:
            traceback.print_exc()
    return text

def convert_videos_into_text():
    video_dir = base_dir + 'video'
    audio_dir = base_dir + 'audio'
    text_dir = base_dir + 'text'
    original_text_dir = base_dir + 'original_text'
    if not os.path.isdir(audio_dir):
        os.mkdir(audio_dir)
    if not os.path.isdir(text_dir):
        os.mkdir(text_dir)

    # Converting video files to audio files
    print('\n#################### Converting video files to audio files ####################\n')
    for video in os.listdir(video_dir):
        video_file = video_dir + file_path_separator + video
        audio_file = audio_dir + file_path_separator + video.split('.')[0] + '.wav'
        if not exists(audio_file):
            mp.VideoFileClip(video_file).audio.write_audiofile(audio_file)

    # Converting audio files into text files
    print('\n#################### DONE | Converting video files to audio files ####################\n')
    
    print('\n#################### Converting audio files to text files ####################\n')
    index = 0
    data_frames = pd.DataFrame()
    for audio in os.listdir(audio_dir):
        file_name = audio.split('.')[0]
        audio_file = audio_dir + file_path_separator + audio
        text_file_path = text_dir + file_path_separator + file_name + '.txt'
        if exists(text_file_path):
            text_file = open(text_file_path, 'r')
            text_content = text_file.read()
        else:
            text_file = open(text_file_path, 'w')
            text_content = get_large_audio_transcription(audio_file)
            text_file.write(text_content)
            text_file.close()

        try:
            print("\n### Successfully converted file [", file_name, "]")
            original_text_file = open(original_text_dir + file_path_separator + file_name + '.txt', 'r')
            original_text = original_text_file.read()
            print("    Similarity (cosine) = ", calculate_cosine_similarity(text_content, original_text))
            print("    Similarity (spacy) = ", spacy_similarity.spacy_similarity(text_content, original_text))
            original_text_file.close()
        except Exception as e:
            print("    Skipping similarity calculation as there is no original text file")
        
        data_frames = data_frames.append(pd.DataFrame(index=[index], data={'title': file_name, 'text': text_content}))
        index = index + 1

    print('\n#################### DONE | Converting audio files to text files ####################')

    return data_frames

def extract_text_into_data_frame():
    text_dir = base_dir + 'text'
    
    data_frames = pd.DataFrame()
    index = 0
    for text in os.listdir(text_dir):
        text_file = text_dir + file_path_separator + text
        text = open(text_file, 'r')
        text_content = text.read()
        text.close()
        data_frames = data_frames.append(pd.DataFrame(index=[index], data={'title': text_file, 'text': text_content}))
        index = index + 1

    print('\n#################### DONE | Converting audio files to text files ####################')

    return data_frames
