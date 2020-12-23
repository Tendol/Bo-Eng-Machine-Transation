from bs4 import BeautifulSoup
import requests
import re
from unicodedata import normalize
import sys 

# get the links of all the pages in the website from start - (excluding end)
def scrap(url, start, end):
    content = []
    for i in range(start, end):
        link = url + "/page/" + str(i) +"/"
        content.append(content_scrap(link))
    return content

# scrap the content from the webpages
def content_scrap(url):
    # url = "https://www.bod.asia/"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')

    atags = soup.find_all('a', {'class':'posts-item-title-link'})
    links = [atag['href'] for atag in atags]

    a = []

    for link in links:
        sub_url = "https://www.bod.asia/"+link
        r = requests.get(sub_url)
        s = BeautifulSoup(r.content, 'html.parser')
        for text in s.find_all('span'):
            line = normalize('NFKD', text.get_text())
            a.append(line)
    return a

# save a scrapped content to file
def save_content(content, filename):
    with open(filename, 'w') as filehandle:
        for sentences in content: 
            for sentence in sentences:
                filehandle.writelines(sentence)
    print('Saved: %s' % filename)

if __name__ == '__main__':
    if(len(sys.argv) < 4):
        url = input("enter the url name: ")
        start = input("start: ")
        end = input("end: ")
        output = input("enter the output file name: ")
    else:
        url = sys.argv[1]
        start = sys.argv[2]
        end = sys.argv[3]
        output = sys.argv[4]

        content = scrap(url, int(start), int(end))
        save_content(content, output)
