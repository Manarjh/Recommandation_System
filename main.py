from video_to_text_converter import convert_videos_into_text
from video_to_text_converter import extract_text_into_data_frame
from lda import generate_lda_model

class Settings:
    debug = False


class LDA:
    num_topics = 3
    num_of_words_topic_description = 10
    passess = 100
    chunksize=10000
    update_every=3


######
data_frames = convert_videos_into_text()
# data_frames = extract_text_into_data_frame()

lda_model = generate_lda_model(data_frames, LDA, Settings)
