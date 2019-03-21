#Python 3.0
import re
import os
import collections
import time
#import other modules as needed

class index:
	def __init__(self,path):

	def buildIndex(self):
		#function to read documents from collection, tokenize and build the index with tokens
		# implement additional functionality to support relevance feedback
		#use unique document integer IDs

	def rocchio(self, query_terms, pos_feedback, neg_feedback, alpha, beta, gamma):
	#function to implement rocchio algorithm
	#pos_feedback - documents deemed to be relevant by the user
	#neg_feedback - documents deemed to be non-relevant by the user
	#Return the new query  terms and their weights
	
	def query(self, query_terms, k):
	#function for exact top K retrieval using cosine similarity
	#Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
	
	def print_dict(self):
    #function to print the terms and posting list in the index

	def print_doc_list(self):
	# function to print the documents and their document id
