# Python Search-Engine
Search Engine made in Python based on Stack Overflow scraped comments

## Project Overview 

The main goal of this project was to create an information retrieval system in order to retrieve the most related documents from the corpus according to the user’s input.
For the creation of the corpus, I Scraped user Questions from Stack Overflow. In total 5475 User Questions in addition to their answers were scraped from Stack Overflow.
During the scrapping I encountered issues as the website blocked the IP address of the computer that was doing the data scrap because of the high number of requests to the website. 
In order to solve this issue, I incorporated time.sleep() in the code of the scraper in order to pause the code for a number of random seconds (1-4) in order to imitate
the actions of a real person. 
Even though the program became significantly slower the website stopped banning our IP address and we collected the data for 5475 User Questions.
The data were saved in a json file that was used as the main source to create the Corpus. 

The data were processed in order to remove duplicates, more specifically during the scrape many questions were scraped twice. 
In order to remove the duplicates a dictionary was used where the key for every document was the URL of the Stack Overflow Question, and the value was question itself. 
Any duplicates were overridden. In total 2351 Duplicates were removed.

## Vector Space Model

# Processing the data

The Vector Space Model was used as the main information retrieval model for this project. Each document in the Corpus was preprocessed before calculating TF-IDF. 
More specifically any stop words were removed (The nltk library was used in order to create a list of all stopwords) as well as any non-words (numbers dates e.t.c) 
that did not provide any actual value to the semantics of the document (Regex library was used in order to capture only words that contained letters). 
The remaining text was stemmed by using PorterStemmer. The same methodology was also used later in the program in order to process the User’s Query.

# TF-IDF

All the documents and the Query were vectorized by using the TF-IDF methodology. 
Then the inner product between the Vector of the Query and each document vector was calculated and all the results were stored sorted out in descending order. 
In order to choose the most similar to the query results the average inner product was calculated and only the documents that had inner product higher than the 
average are printed for the user.

# Auto Correct 

The Damerau-Levenshtein distance method was used in order to create an autocorrect algorithm for the user Queries. 
A Set of all the unique words in the corpus in addition to all the words in the English vocabulary was created, a total of 236.736 words. 
More specifically first we downloaded all the English Vocabulary from the nltk library. In total 236.736 words were downloaded from nltk.

The idea was that after the user inputs his Query and after processing the query (removing stopwords, non-words, stemming) each remaining word would be passed 
through a function that checks if it is a word or not if the word exists in the English vocabulary, then no further actions required, and the next word of the querywould be checked. 
To accomplish that by avoiding iterating through the whole vocabulary we created a dictionary were the key and the value for each word was the word itself (d={“table”:”table”}). 
Every time a word from the query was to be checked if it exists in the vocabulary, the word was put in the dictionary d, d[“table”]. 
By using try and except method if no error was occurring then the word existed in the vocabulary. Otherwise, (except) the Levenhstein distance between the 
Query word and all the words that have same length (-1,+1) and start with the same letter was calculated. 

