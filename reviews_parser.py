import os
import csv
import codecs
from os.path import exists
from scipy import spatial
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# row[0] reviewer comment
# row[3] original ratings
# row[5] calculated score using VADER algorithm

analyzer = SentimentIntensityAnalyzer()
file_path_separator = '/' if os.name == 'posix' else '\\'
base_dir = 'test_dataset' + file_path_separator
extended_reviews_file_path = base_dir + 'reviews_extended.csv'

def generate_extended_reviews_file():
    if exists(extended_reviews_file_path):
        os.remove(extended_reviews_file_path)

    print('Getenrating reviews_extended.csv...')
    with open(base_dir + 'reviews.csv', 'rb') as inp, open(extended_reviews_file_path, 'w', encoding="utf-8") as out:
        writer = csv.writer(out)
        for row in csv.reader(codecs.iterdecode(inp, 'utf-8')):
            extended_row = ["" for x in range(8)]
            extended_row[0] = row[0]
            extended_row[1] = row[1]
            extended_row[2] = row[2]
            extended_row[3] = row[3]
            extended_row[4] = row[4]
            if row[3] == 'rating':
                extended_row[5] = 'new rating'
                extended_row[6] = 'vader score'
                extended_row[7] = 'consine similarity'
            else :
                vader_compound_score = analyzer.polarity_scores(row[0])['compound']
                newRating = vader_score(vader_compound_score)
                extended_row[5] = newRating
                extended_row[6] = vader_compound_score
                extended_row[7] = 1 - spatial.distance.cosine([int(extended_row[3])], [newRating])
            writer.writerow(extended_row)
    print('reviews_extended.csv is generated and located in ' + base_dir)

def vader_score(vader_compound_score):
    if vader_compound_score >= -1 and vader_compound_score < -0.5:
        return 1
    elif vader_compound_score >= -0.5 and vader_compound_score < -0.05:
        return 2
    elif vader_compound_score >= -0.05 and vader_compound_score <= 0.05:
        return 3
    elif vader_compound_score > 0.05 and vader_compound_score < 0.5:
        return 4
    else:
        return 5

if __name__ == '__main__':
    generate_extended_reviews_file()
    source_ratings = []
    vader_ratings = []
    for row in csv.reader(codecs.iterdecode(open(extended_reviews_file_path, 'rb'), 'utf-8')):
        if row[3].isnumeric():
            source_ratings.append(int(row[3]))
            vader_ratings.append(int(row[5]))
    
    similarity = 1 - spatial.distance.cosine(source_ratings, vader_ratings)
    print("\n\n    Cosine Similarity between Original Rating and VADER Rating =", similarity, "\n\n")
