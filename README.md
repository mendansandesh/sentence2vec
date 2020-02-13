<h1>Sentence 2 vec</h1>

Simple Python 3.5/3.6 implementation of the algorithm by Sandesh Mendan

- `Categorize.py` estimates the category of the input text by comparing cosine distance from input sentence to each category sentences defined in `categories.txt`

- input sentence could be of plain english text/ web url/ database query statement/ file path

- do not alter 1st word from `categories.txt`; if you need add more keyword in each category sentence add it anywhere after first word position

Sample input of this program:
```
https://www.amazon.in/gp/product/B07S6P5FQ3/ref=ox_sc_act_title_6?smid=A2VQOLRW0XTGMB&psc=1
```
Sample output of this program:
```
DataCategory is: purchase
``` 