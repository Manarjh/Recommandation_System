import re
import numpy as np 
import spacy
from spacy.lang.en import English
spacy_en = English(disable=['parser', 'ner'])
from spacy.lang.en.stop_words import STOP_WORDS as stopwords_en


class SpacySimilarity(object):

	def __init__(self):
		self.spacy_large_model = spacy.load("en_core_web_lg")	

	def clean_text(self, text):
		try:
			text = str(text)
			text = re.sub(r"[^A-Za-z0-9]", " ", text)
			text = re.sub(r"\s+", " ", text)
			text = text.lower().strip()
		except Exception as e:
			print("\n Error in clean_text --- ", e,"\n ", traceback.format_exc())
			print("\n Error sent --- ", text)
		return text

	def get_lemma_tokens(self, text):
		return " ".join([tok.lemma_.lower().strip() for tok in spacy_en(text) if (tok.lemma_ != '-PRON-' and tok.lemma_ not in stopwords_en and len(tok.lemma_)>1)])

	def cleaning_pipeline(self, text):
		text = self.clean_text(text)
		#text = self.get_lemma_tokens(text)
		return text

	def cos_sim(self, vector_1, vector_2):
		return np.inner(vector_1, vector_2) / (np.linalg.norm(vector_1) * (np.linalg.norm(vector_2)))

	def spacy_similarity(self, sent1, sent2):
		sent1_cleaned = self.cleaning_pipeline(sent1)
		sent2_cleaned = self.cleaning_pipeline(sent2)
		sent1_vector = self.spacy_large_model(sent1_cleaned).vector
		sent2_vector = self.spacy_large_model(sent2_cleaned).vector
		return self.cos_sim(sent1_vector, sent2_vector)

if __name__ == '__main__':
	obj = SpacySimilarity()
	x1 = "Let's get started with our linear algebra review. In this video I want to tell you what are matrices and what are vectors. A matrix is a rectangular array of numbers written between square brackets. So, for example, here is a matrix on the right, a left square bracket. And then, write in a bunch of numbers. These could be features from a learning problem or it could be data from somewhere else, but the specific values don't matter, and then I'm going to close it with another right bracket on the right. And so that's one matrix. And, here's another example of the matrix, let's write 3, 4, 5,6. So matrix is just another way for saying, is a 2D or a two dimensional array. And the other piece of knowledge that we need is that the dimension of the matrix is going to be written as the number of row times the number of columns in the matrix. So, concretely, this example on the left, this has 1, 2, 3, 4 rows and has 2 columns, and so this example on the left is a 4 by 2 matrix - number of rows by number of columns. So, four rows, two columns. This one on the right, this matrix has two rows. That's the first row, that's the second row, and it has three columns. That's the first column, that's the second column, that's the third column So, this second matrix we say it is a 2 by 3 matrix. So we say that the dimension of this matrix is 2 by 3. Sometimes you also see this written out, in the case of left, you will see this written out as R4 by 2 or concretely what people will sometimes say this matrix is an element of the set R 4 by 2. So, this thing here, this just means the set of all matrices that of dimension 4 by 2 and this thing on the right, sometimes this is written out as a matrix that is an R 2 by 3. So if you ever see, 2 by 3. So if you ever see something like this are 4 by 2 or are 2 by 3, people are just referring to matrices of a specific dimension. Next, let's talk about how to refer to specific elements of the matrix. And by matrix elements, other than the matrix I just mean the entries, so the numbers inside the matrix. So, in the standard notation, if A is this matrix here, then A sub-strip IJ is going to refer to the i, j entry, meaning the entry in the matrix in the ith row and jth column. So for example a1-1 is going to refer to the entry in the 1st row and the 1st column, so that's the first row and the first column and so a1-1 is going to be equal to 1, 4, 0, 2. Another example, 8 1 2 is going to refer to the entry in the first row and the second column and so A 1 2 is going to be equal to one nine one. This come from a quick examples. Let's see, A, oh let's say A 3 2, is going to refer to the entry in the 3rd row, and second column, right, because that's 3 2 so that's equal to 1 4 3 7. And finally, 8 4 1 is going to refer to this one right, fourth row, first column is equal to 1 4 7 and if, hopefully you won't, but if you were to write and say well this A 4 3, well, that refers to the fourth row, and the third column that, you know, this matrix has no third column so this is undefined, you know, or you can think of this as an error. There's no such element as 8 4 3, so, you know, you shouldn't be referring to 8 4 3. So, the matrix gets you a way of letting you quickly organize, index and access lots of data. In case I seem to be tossing up a lot of concepts, a lot of new notations very rapidly, you don't need to memorize all of this, but on the course website where we have posted the lecture notes, we also have all of these definitions written down. So you can always refer back, you know, either to these slides, possible coursework, so audible lecture notes if you forget well, A41 was that? Which row, which column was that? Don't worry about memorizing everything now. You can always refer back to the written materials on the course website, and use that as a reference. So that's what a matrix is. Next, let's talk about what is a vector. A vector turns out to be a special case of a matrix. A vector is a matrix that has only 1 column so you have an N x 1 matrix, then that's a remember, right? N is the number of rows, and 1 here is the number of columns, so, so matrix with just one column is what we call a vector. So here's an example of a vector, with I guess I have N equals four elements here. so we also call this thing, another term for this is a four dmensional vector, just means that this is a vector with four elements, with four numbers in it. And, just as earlier for matrices you saw this notation R3 by 2 to refer to 2 by 3 matrices, for this vector we are going to refer to this as a vector in the set R4. So this R4 means a set of four-dimensional vectors. Next let's talk about how to refer to the elements of the vector. We are going to use the notation yi to refer to the ith element of the vector y. So if y is this vector, y subscript i is the ith element. So y1 is the first element,four sixty, y2 is equal to the second element, two thirty two -there's the first. There's the second. Y3 is equal to 315 and so on, and only y1 through y4 are defined consistency 4-dimensional vector. Also it turns out that there are actually 2 conventions for how to index into a vector and here they are. Sometimes, people will use one index and sometimes zero index factors. So this example on the left is a one in that specter where the element we write is y1, y2, y3, y4. And this example in the right is an example of a zero index factor where we start the indexing of the elements from zero. So the elements go from a zero up to y three. And this is a bit like the arrays of some primary languages where the arrays can either be indexed starting from one. The first element of an array is sometimes a Y1, this is sequence notation I guess, and sometimes it's zero index depending on what programming language you use. So it turns out that in most of math, the one index version is more common For a lot of machine learning applications, zero index vectors gives us a more convenient notation. So what you should usually do is, unless otherwised specified, you should assume we are using one index vectors. In fact, throughout the rest of these videos on linear algebra review, I will be using one index vectors. But just be aware that when we are talking about machine learning applications, sometimes I will explicitly say when we need to switch to, when we need to use the zero index vectors as well. Finally, by convention, usually when writing matrices and vectors, most people will use upper case to refer to matrices. So we're going to use capital letters like A, B, C, you know, X, to refer to matrices, and usually we'll use lowercase, like a, b, x, y, to refer to either numbers, or just raw numbers or scalars or to vectors. This isn't always true but this is the more common notation where we use lower case 'Y' for referring to vector and we usually use upper case to refer to a matrix. So, you now know what are matrices and vectors. Next, we'll talk about some of the things you can do with them"
	x2= "Let's get started with a linear algebra review. In this video when i tell you what i'm agencies and one of exes. A matrix. Is a rectangular array of numbers written between square brackets. Example here is a matrix. Play radio let's clear back. And then writing a. Bunch of numbers. And you know this could. Features for machine learning problem what could be data from somewhere else. Example the specific values. Don't matter. Close it with another right back it on the right so that's 1 m. 12th. Symmetric is just another way for saying is it 30 or 2 dimensional. Array. Enzyme. Epistemology we need is that dimensional matrix is going to be. Everton has the number of rows. Times the number of columns in matrix. Completely example. This has one too. 34 rose and has. Two columns and cells example the left i'm going to say this is a 4. By 2. Does number of rows by number of columns 0402 harlem. This one on the right this nucleus has to roll that's a first. That's the second roll and has three columns. That's the first column that's the second column. So. The second matrix crusaders a 253. The dimension of this matrix 2 by 3. Sometimes it also see this written out as. In the kids unless you see the original sr4 by two countries what people sometimes say is that. This matrix is an element of a centaur for by to this thing here at this just means he loves all matrices the lot of dimension 4 by 2. And has written out as a matrix present. So give us something like this in your offer by two r23 people just referring agencies of us pacific. Dimension. Next let's talk about how to refer to specific elements of the matrix and by the numbers. So in the standardisation is a is this an issue. The 8th of july jay is going to refer to the icon emoji. Meaning. Entry in the mate races in the i throw and j cole. Example a11 is it going to refer to the entry in the first row and the first column so that's the first coronavirus column. And so. A11 is going to be good. 14. Barnardo's app. A12. Is going to refer to the entry in the first row. Andy second column. And so they want to. 191. Just come from quickies apples. The cia. Me too. Is going to refer to the entry in the. Roll. And the second column. So that's. Wonderful. 7. Finally. A41. Is going to refer to. 147. And if hopefully you won't be ready to write say what is. 43. Well that refers to the fourth row. And the third column bed has no third column so this is. You can think of this. Sr a430. Cinebee referring. So the matrix gives you a way of letting you quickly organise index and access lots of data. And a lot of concerns about a new notation you don't need to memorize all this but on the course website where we have posted the nationals we also have all of these. So you can always refer back you know either to the slice bottle. A41. Don't worry about them right now. Do the red arrows on the course website newsletter. So the matrix is about 1 as a vector. Evektor turns out to be a special case of a matrix with a vector. Is it matrix that has only one column so you can end by 1 m. Then that's a remember right n is the number of rows. And one here's number colin. The matrix with just one column. Islamic vector. Example of a vector. I guess i have any. Four elements year. So we also called this thing this is that this is a four dimensional. Just means that. This is a vector. For ella. Phone number. Earlier for matrices you sort this location they are 3 by 233 by 2 matrices right. We're going for a further stew to this as a vector in the sets are. Again just music set of all. Four dimensional vectors. Next let's talk about how to refer to elements of the vector. We're going to use the notation why i. To refer to the eiffel amounts of the. Why is inspector why such as the ice. Why was the first album. 460. Y2. Expecting element. 232. Y3. 315. And don't by 12yo for undefined because it's a 4-dimensional. Better. Also turns out that bracket to conventions for how to index into a back. Sometime. People will use one index and sometimes 0 index factors example on the left is a 1. Index factor where the elements you write as y1 y2. Whites by 4. Example example of a zero in vector. Where we start the indexing of the elements from 0. So the animals go from 0 up to 3. And this is a little bit like the arrays of some programming language. The arrays can either be indexed starting for one. The first element of my one. Sometimes the h. 00s sitting on white programme. So it turns out that in most of now by one index version is more common. That's a lot of machines or any application. 0 index of vectors gcse more convenient. So why she usually do is there unless otherwise specified you should have seen their using one index factors and thanks for all the recipes videos on the algebra review i will be using one index effective. Where are the 1 top rubbish in the applications sometimes explicitly. When we need to use the zero index. Finally by convention usually when writing matrices and vectors most people will use uppercase. To refer to matrices solver used cattle miss capital letters r. Cox-2 referred to matrices. And usually what use lowercase like. Xy. To refer to either numbers just wrong and purcell scalars or two vectors this isn't always true but this is the more common notation. How are you use uppercase to refer to a matrix. So. You don't know what are matrices in vector. Next we'll talk about some of the. "
	print("\n spacy cosine similarity = ", obj.spacy_similarity(x1, x2))