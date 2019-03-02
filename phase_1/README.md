

~merge algorithm

I use a dictionary to keep track of what element of each doc_list I have jsut examined,
and to keep track of the total length so that when I get to the end of any of the doc_lists
i exit because it means that there are no more documents that will match for every term.

the way i iterate through each terms doc_lists is as follows:
1. if all the docIDs which i am examining are the same value, then the each term is in the 
same document and I add that document to a final list.  I then move all of the indexes I am 
examining up by one value and repeat the process of comparing the DocIDS.

2. If the document ids are not all equal, then i move the indexes forward one for all of the 
term_lists which are not the max_value amoung the docIDs i am looking at.  I do this because
the docIDS that are less than the max DocID will never be documents to return because we 
already know that the term_list with the index = max DocID is not in the documents with lower
docIDS. after moving forward all indexes except the documents with the docID equal to the 
max DocID, i move back to the top of the loop and compare the new indexes.


This algorithm is linear in porportion to the total number of DocIDs per term.



the rest of the program is well documented.
