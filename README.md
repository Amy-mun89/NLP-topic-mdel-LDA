# NLP-topic-mdel-LDA

## 1. Dataset

the dataset were gathered from New York Times website, Energy section. (nytimes.com). the Website offers the journals by categories, and I used the category energy. For the text mining, I had to check the structure of website. 
The websiste basically using HTML base, and had four big frames. To create the crawler, I used selenium chrome web driver and python. For the first put the url and access address. In this step, I already put the url which is  energy section so that I can avoid additional step. The journals I wanted to crawl is only for renewable energy, so I used send_keys function from BeautifulSoup. Then make the sorting option as newest. This sorting option was found as Xpath from chrome instpection. 
Then use the selenium to scroll down and at the end download the date, title and headline and save as csv file.


This dataset has date, title and headline of the journals related renewable energy from Dec 11 2020 to Feb 26, 2021, and it has total 110 rows without missing values.
The ‘news’ column is combination of ‘title’ column and ‘headline’ column. for the topic modeling, mostly the ‘news’ column has been used.

## 2. text pre-processing
1) special characters, numbers and punctuation marks are removed. For this step, python replace function has been applied. Every character excludes English al-phabet (a-zA-Z) is replaced to blank. (“ “).

2) Second step is removing the short length words. In this project, the words have less than 3 alphabet character are assumed as not useful information. For example, “if”, “it”, “of”, “at”. For this step, for loop and if statement has been applied. 

3) convert capital letters to lower letters. By this steps, the total number of words can be re-duced. For this step, apply function has been applied

## 3. LDA
LDA is an unsupervised machine learning model that find topics from the literature and one of the representative algorithms of topic modeling. in this code, gensim library has been applied for the model. 
## 4. Visualization
For the visualization of LDA model, pyLDAvis package has been applied.
The distance of each circle shows how different each topic is from each other. If the two circles overlapped, it indicates that these two topics are similar topics

By clicking each circle, each words term frequency is shown as bar chart representation. The blue bar indicates overall term frequency and the red bar indicates estimated term frequency within the selected topic, and the bar chart is sorted by the red line
