#Python 3.7
import re
import os
import collections
import time
import math

from numpy import dot
from numpy.linalg import norm
import numpy as np 

class index:
        #maps term ids to the doc ids and the positions of the term in each document
        posting_list = {}
        #maps document ids to there file name for returning file names
        docID = {}
        #maps terms to term ids
        termID = {}
        
        #this will contain docIDs as keys
        #as values will be a list of the terms in the document
        # and the number of times they appear in the document
        # it is used for vectorizing docs and queries
        doc_to_term = {}

        # Ranks the top documents per word, based on frequency 
        # I choose how it is made 
        champions_list = {}



        #words to ignore
        stop_words = set()
        with open('./../stop-list.txt','r') as f:
            for line in f:
                 for word in line.split():
                     stop_words.add(word)


        def __init__(self,path):
            self.path = path

        def buildIndex(self):
	    #function to read documents from collection, tokenize and build the index with tokens
	    #index should also contain positional information of the terms in the document --- term: [(ID1,[pos1,pos2,..]), (ID2, [pos1,pos2,…]),….]
	    #use unique document IDs
            #ignore stop words 

            path = self.path
            #just for timing
            start_time = time.time()

            #regular expression for parsing, this may need to get more complicated for normalizing, stemming, etc
            p = re.compile('[a-z]+',re.IGNORECASE)
            
            listing = os.listdir(path)
            print('total number of documents: %d'%len(listing))
            for infile in range(len(listing)):
            #for infile in range(1):
                f = open(path + listing[infile])
                self.docID[len(self.docID)] = listing[infile] 
                index = {}
                m = p.findall(f.read().lower())
                
                #adding the docID to a doc_to_term dictionary
                self.doc_to_term[len(self.docID)] = {}


                #this creates the termID dictionary
                #m = total terms per document
                #so this forloop iterates through each documents set of words and constructs a termlist
                for i in range(len(m)):
                    if m[i] not in self.stop_words:
                        if m[i] in self.termID:
                            ID = self.termID[m[i]]
                        else:
                            ID = len(self.termID)
                            self.termID[m[i]]=len(self.termID)
                        


                    #makes a posting list per document because modifying tuples is a bitch
                    #so i make them as lists, then I will cast them as I add them to the final list
                    #if they could just be lists, I wouldnt need to do this... AAAAAAAAA
                        if ID in index:
                            index[ID].append(i)
                        else:
                            index[ID] = [i]

                #Here we take our document posting list, and cast into tuples nad add to the final posting list        
                for i in index:
                    
                    '''
                    # champions list stuff, attempt 1
                    # so far this is not working lol

                    if (len(index[i])/len(index)) > .01:
                        if i in self.champions_list:
                            self.champions_list[i].append(len(self.docID))
                        else:
                            self.champions_list[i] = [len(self.docID)]
                    '''
                    self.doc_to_term[len(self.docID)][i]=len(index[i])
                    if i in self.posting_list:  
                        self.posting_list[i].append((len(self.docID),len(index[i]) ,index[i]))
                        TFID = math.log10(len(listing)/(len(self.posting_list[i])-1))
                        self.posting_list[i][0]=TFID
                    else:
                        TFID = math.log10(len(listing))
                        self.posting_list[i]=[TFID, (len(self.docID),len(index[i]),index[i])]

                
            self.compute_champions_list(10)
            print('Index built in: ', round(time.time()-start_time,4))
            
        

##############################
        
# INEXACT SEARCHES 

##############################


    # Main Inexact Search Function
        def inexact_search(self,query, k=10):
            
            query = self.remove_stop(query)
            query = self.query_to_termID(query)
            
            start_time = time.time()
            
            #champions list section
            docs_to_score = set()
            for termID in query:
                for doc in self.champions_list[termID]:
                    docs_to_score.add(doc)
    
            scores0 = self.score_documents(query,docs_to_score,k)
            if len(scores0) == 0:
                print('\nSorry we didnt find any similar documents :(\n')
            else:
                print('Champions List Results')
                for i in range(len(scores0)):
                    print(self.docID[scores0[i][1]])
                print('Time taken to retrieve query results using the Champions List: ', round(time.time()-start_time,4))
            
            
            start_time = time.time()
            
            #Index Elimination Section
            query = self.index_elimination(query)
            docs_to_score = set()
            for termID in query:
                for doc in self.posting_list[termID]:
                     if type(doc) == tuple:
                        docs_to_score.add(doc[0])
                    
            scores1 = self.score_documents(query,docs_to_score,k)
            if len(scores1) == 0:
                print('\nSorry we didnt find any similar documents :(\n')
            else:
                print('Index Elimination Results')
                for i in range(len(scores1)):
                    print(self.docID[scores1[i][1]])
                print('Time taken to retrieve query results using Index Elimination without the Champions list: ',round(time.time()-start_time,4))
  

            return scores0, scores1
        







#############################
    # CHAMPIONS LIST

    # compute champions list, I am serperating this from the index building based
    # on the idea that I need the complete posting list to create an effective 
    # champions list.

    # I can totally reduce index contruction time complexity by doing this with a tree
    # and by doing this while the posting list is being built.  this will jsut act as
    # a proof of concept/fast testing and then i will impliment a bettter version.
        def compute_champions_list(self, k):
            for term in self.posting_list:
                docs_to_add = []
                if len(self.posting_list[term]) < k+1:
                    for doc in self.posting_list[term]:
                        if type(doc) == tuple:
                            docs_to_add.append(doc[0])
                    self.champions_list[term] = docs_to_add
                
                else:
                    potential_docs=[]
                    for doc in self.posting_list[term]:
                        if type(doc) == tuple:
                            potential_docs.append((doc[0],doc[1]))
                    potential_docs = sorted(potential_docs, key=lambda k:(-k[1],k[0]))
                    for doc in potential_docs:
                        docs_to_add.append(doc[0])
                    self.champions_list[term] = docs_to_add[:k]
                    

#############################

    # INDEX ELIMINATION

    # take round(number of query terms / 2) and only use these for searching
    # choose the query words based on IDF vales. (use the rare half of the words)
        
        def index_elimination(self,query):
            term_idf = []
            for term in query:
                term_idf.append((term,self.posting_list[term][0]))
            
            term_idf = sorted(term_idf, key=lambda k:(-k[1],k[0]))
            terms_to_search = []
            print('!!!!!!!!!!!!!!',math.ceil(len(query)/2)) 
            for i in range(math.ceil(len(query)/2)):
                terms_to_search.append(term_idf[i][0])
            return terms_to_search

#############################

    # SIMPLE CLUSTER PRUNING

# todo .......




#############################

# EXACT SEARCH

#############################

    # takes a query, computes cosine for all documents with at least one term
    # which is similar to the query. and returns the documents in a ranked order
    
    # currently the scores and documents associated with each score is only stored 
    # in a list and then sorted based on score, this is inefficient and should be
    # a tree structure to reduce sort/retrieval complexity
        def exact_search(self, query, k):
            start_time = time.time()
            docs_to_score = set()
            scores = []
            if len(query) ==0:
                print('No documents with any words related, sorry:(')
                return
            query = self.remove_stop(query)
            query = self.query_to_termID(query)
            
            # obtain docs to score
            for termID in query:
                for doc in self.posting_list[termID]:
                    if type(doc) == tuple:
                        docs_to_score.add(doc[0])
        
            scores = self.score_documents(query,docs_to_score,k)

            if len(scores) == 0:
                print('\nSorry we didnt find any similar documents :(\n')
            else:
                for i in range(min(k,len(scores))):
                    print(self.docID[scores[i][1]])
                print('Time taken to retrieve query results from Exact Search: ', round(time.time()-start_time,4))
            return scores[:k]
        


##############################        


        def query_to_termID(self,query):
            ID_query=[]
            for i in query:
                try:
                    ID_query.append(self.termID[i])
                except:
                    pass
            
            return ID_query 

        #takes a query,document, and computes the document and query vectors
        def compute_doc_query_vectors(self, query, docID):
            query_vector =[]
            doc_vector=[]
            for i in self.doc_to_term[docID]:
                TFIDF = (1+math.log10(self.doc_to_term[docID][i])) * self.posting_list[i][0]
                doc_vector.append(TFIDF)
                if i in query:
                    TFIDF = (self.posting_list[i][0])
                    query_vector.append(TFIDF)
                else:
                    query_vector.append(0)
            for i in query:
                if i not in self.doc_to_term[docID]:
                    query_vector.append(self.posting_list[i][0])
                    doc_vector.append(0)
            return query_vector,doc_vector 


        # take two vectors and compute similarities
        def compute_cosine_sim(self, query_vector, doc_vector):
            cos_sim = dot(query_vector, doc_vector)/(norm(query_vector)*norm(doc_vector))
            return cos_sim 


        # take a query and a list of docs, and return top k
        def score_documents(self, query, docs_to_score, k):
            scores = []
            # compute score for each doc
            for docID in docs_to_score:
                query_vector, doc_vector = self.compute_doc_query_vectors(query,docID)
                score = self.compute_cosine_sim(query_vector,doc_vector)
                scores.append((score,docID))

            #sort and return top k docs
            scores = sorted(scores, key=lambda x:(-x[0],x[1]))
            return scores[:k]



############################

        def remove_stop(self,query):
            to_ret = []
            for i in query:
                if i not in self.stop_words:
                    to_ret.append(i)
            return to_ret


###########################

        #function to print the terms and posting list in the index
        def print_dict(self):
            start_time = time.time()
            inv_map = {v: k for k, v in self.termID.items()}
            for i in range(len(self.posting_list)):
                time.sleep(.1)
                print(inv_map[i],self.posting_list[i])
            print('time to process and print: ', time.time()-start_time)



	#print all documents and their document id
        def print_doc_list(self):
            for row in self.docID:
                print(row, self.docID[row])



        #print the document names for the specified document ids
        def print_doc_title(self, docs):
            print("Requsted Documents")
            for doc in docs:
                print(self.docID[doc])



###########################
    
        def examine_champions_list(self):
            information = []
            count = 0
            for i in self.champions_list:
                count += len(self.champions_list[i])
                information.append(len(self.champions_list[i]))
            data = np.array(information) 
            hist,bins=np.histogram(data)
            print('champions list length: ',len(self.champions_list))
            print('total docs in champions list(with overlap): ',count)
            print('average champions list length: %d' % (count/len(self.champions_list)))
            print('Variance of the champions list results: ',np.var(information))
            print('max length of champions list: ', max(information))


##########################

        


#for checking if all TermId's are in the same document
#for exact searching 
def is_equal(items):
    count = 0
    for a in terms:
        if items[0] == a:
            count += 1

    if count == len(item) - 1:
        return True
    else:
        return False






def main():
    ind = './../collection/'
    i = index(ind) 
    print('Index being used: %s' %ind)
    i.buildIndex()
   
   #debugging
    '''
    for x in range(0,100):
        if(len(i.posting_list[x]) ==2):
            print(x, i.posting_list[x],len(i.posting_list[x])) 
    '''
   
    '''
    i.examine_champions_list()
    for x in range(0,40):
        print(i.champions_list[x])
    '''

    while True:
        query = (input('Please enter the query terms:').strip().lower()).split(' ')
        docs0 = i.exact_search(query,5)
        docs1,doc2 = i.inexact_search(query,5)
if __name__ == '__main__':
    main()












