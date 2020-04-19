#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import numpy as np
from bs4 import BeautifulSoup
import itertools
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import re
from lxml import html
import math
import time
import sys


# In[50]:


def inside_get_year(url_):
    #url = "https://www.ptt.cc/bbs/Beauty/M.1568553917.A.175.html"
    time.sleep(0.1)
    payload = {
        "from": "/bbs/Gossiping/index.html",
        "yes": "yes"
    }
    rs = requests.session()
    res = rs.post("https://www.ptt.cc/ask/over18", verify = False, data = payload)    
    res = rs.get(url_, verify = False, headers={'Connection':'close'})    
    tree = html.fromstring(res.content)
    a = tree.xpath("//*[@id=\"main-content\"]/div[4]/span[2]")[0]
    return( a.text[-4:] )


# In[3]:


def url_get_date(int_):
    head = "https://www.ptt.cc/bbs/Beauty/index"
    end = ".html"    
    url_ = head + str(int_) + end
    payload = {
        "from": "/bbs/Gossiping/index.html",
        "yes": "yes"
    }
    rs = requests.session()
    res = rs.post("https://www.ptt.cc/ask/over18", verify = False, data = payload)    
    res = rs.get(url_, verify = False, headers={'Connection':'close'})      
    soup = BeautifulSoup(res.text)
    block_l = soup.select(".r-ent")
    for i in block_l:
        try:
            date = i.select(".date")[0].text[1:]
            date = date.replace("/", "")
            #print(date)
            URL = i.select(".title")[0].select("a")[0]["href"]
            head2 = "https://www.ptt.cc"
            year = inside_get_year(head2 + URL)
            #print(year)
            if( len(year + date)==7 ):
                return( int( year + "0" + date) )
            else:
                return( int( year + date) )
            break
        except:
            pass


# In[4]:


#start_time_glo = time.time()
def binary_search(date_, start_, end_ , time_):
    pivot = int((start_ + end_ )/2)
    date = url_get_date( pivot )
    #print(date)
    #print(date_)
    if( time.time() - time_ > 15):
        return(pivot)
    if( date_ < date):
        #print("date_ < date\n")
        return( binary_search(date_, start_, pivot, time_) )
    if( date_ > date):
        #print("date_ > date\n")
        return( binary_search(date_, pivot, end_, time_) )
    if(date_ == date):
        return(pivot)


# In[5]:


def find_start_end():
    start_time_glo = time.time()
    start = binary_search(20171231, 0, 3000, time.time())
    start_time_glo = time.time()
    end = binary_search(20190101, 0, 3000, time.time())
    return( (start, end))


# In[6]:


def num_make_URL(int_):
    head = "https://www.ptt.cc/bbs/Beauty/index"
    end = ".html"
    return(head + str(int_) + end)


# In[7]:


def url_find_block(url_):
    #url = "https://www.ptt.cc/bbs/Beauty/index3057.html"
    try:
        time.sleep(0.1)
        while(True):
            payload = {
                "from": "/bbs/Gossiping/index.html",
                "yes": "yes"
            }
            print(url_)
            rs = requests.session()
            res = rs.post("https://www.ptt.cc/ask/over18", verify = False, data = payload)
            res = rs.get(url_, verify = False, headers={'Connection':'close'}) 
            soup = BeautifulSoup(res.text)
            block_l = soup.select(".r-ent")
            print(url_)
            return(block_l)
            break
    except:
        print("url_find_block: error")
        print(url_)
        print("\n")


# In[8]:


def block_find_data(block_):
    date, title, URL, bao, except_, annoucement = None, None, None, None, False, False
    try:
        date = block_.select(".date")[0].text
        title = block_.select(".title")[0].text
        annoucement = title.startswith('\n[公告]')
        URL = block_.select(".title")[0].select("a")[0]["href"]
        bao = block_.select(".nrec")[0].text
    except:
        except_ = True
    
    return( (date, title, URL, bao, except_, annoucement))


# In[9]:


def data_to_df(block_l_):
    df = pd.DataFrame(list(map(block_find_data, block_l_))) 
    df.columns = ["date","title", "URL",  "bao", "except", "annoucement"]
    return(df)


# In[10]:


def date_adj(str_):
    return(str_.replace("/", ""))


# In[11]:


def title_adj(str_):
    try:
        str_ = re.match(u"^\\n(.*)\\n$", str_).groups()[0]
    except:
        print("title_adj: error")
    return( str_)


# In[12]:


def df_adjust(df):
    df = df[df["except"] == False]
    df = df[df["annoucement"] == False]
    df["URL"] = "https://www.ptt.cc" + df["URL"]
    df["date"] = list(map(date_adj, df["date"].tolist()))
    while( df["date"].tolist()[0] == "1231" ):
        df = df.drop(df.index[0])
    
    while( df["date"].tolist()[-1] == " 101" ):
        df = df.drop(df.index[-1])
    df["title"] = list(map(title_adj, df["title"].tolist()))
    return(df)


# In[1]:


def remove_blank(str_):
    return(str_.replace(" ", ""))


# In[13]:


def all_articles(df):
    buf = df[["date", "title", "URL"]].astype(str)
    buf.date = list(map(remove_blank,  buf.date.tolist()))
    try:
        buf.to_csv('all_articles.txt', sep=',',
                   index = False, header = False)
        print("all_articles: success")
    except:
        print("all_articles: fail")


# In[14]:


def all_popular(df):
    df = df[df["bao"]=="爆"]
    buf = df[["date", "title", "URL"]].astype(str)
    buf.date = list(map(remove_blank,  buf.date.tolist()))
    try:
        buf.to_csv('all_popular.txt', sep=',',
                   index = False, header = False)
        print("all_popular: success")
    except:
        print("all_popular: fail")   


# In[52]:


def crawl():
    print("crawl start")
    page_tuple = find_start_end()
    print(page_tuple)
    URL_list = list(map(num_make_URL, np.arange(page_tuple[0], page_tuple[1])))
    block_list = list(map(url_find_block, URL_list))
    block_list = list(itertools.chain(*block_list))
    df = data_to_df(block_list)
    df = df_adjust(df)
    all_articles(df)
    all_popular(df)
    df.to_csv("HW1-1_3.0.csv")
    return("problem 1 down")


# In[16]:


if( sys.argv[1] == "crawl"):
    crawl()


# In[20]:


def url_find_soup(url_):
    #url = "https://www.ptt.cc/bbs/Beauty/index3057.html"
    time.sleep(0.1)
    try:
        while(True):
            payload = {
                "from": "/bbs/Gossiping/index.html",
                "yes": "yes"
            }
            rs = requests.session()
            res = rs.post("https://www.ptt.cc/ask/over18", verify = False, data = payload)
            res = rs.get(url_, verify = False, headers={'Connection':'close'}) 
            soup = BeautifulSoup(res.text)
            #block_l = soup.select(".push")
            return(soup)
            break
    except:
        print("url_find_block: error")
        print(url_)
        print("\n")


# In[21]:


def push_find_pushtag(push_):
    try:
        return( push_.select(".hl.push-tag")[0].text)
    except:
        print("push_find_pushtag: error:", push_)
        return(None)

def push_find_pushID(push_):
    try:
        return( push_.select(".f3.hl.push-userid")[0].text)
    except:
        print("push_find_pushID: error", push_)
        return(None)


# In[22]:


def tag_to_text(tag_):
    return(tag_.text)

def find_all_href(soup_):
    compare = "(.PNG|.JPEG|.GIF|.JPG|.png|.jpeg|.gif|.jpg)$"
    try:
        all_hreftag = soup_.find_all(href=re.compile(compare))
        return( list(map(tag_to_text , all_hreftag)) )
    
    except:
        print("find_all_href: error")
        return(None)

def find_article_href(soup_):
    try:
        compare = "(.PNG|.JPEG|.GIF|.JPG|.png|.jpeg|.gif|.jpg)$"
        buf = soup_.select("#main-content")[0]
        article_hreftag = buf.find_all(href=re.compile(compare), recursive=False)  
        return( list(map(tag_to_text ,article_hreftag)) )
    
    except:
        print("find_article_href: error")
        return(None)    


# In[23]:


def soup_find_article(soup):
    soup.select("#main-content")[0].text
    article = soup.select("#main-content")[0].text
    article = article.replace("\n", "")
    compare = r"(.*)--※ 發信站"
    buf = re.search( compare, article).groups()[0]
    return(buf)


# In[24]:


def url_find_data(url_):
    print(url_)
    push_tag_l, push_userid_l, all_hreftag = None, None, None
    article_hreftag, article = None, None
    try:
        soup = url_find_soup(url_)
        push = soup.select(".push")
        push_tag_l = list(map(push_find_pushtag, push))
        push_userid_l = list(map(push_find_pushID, push))
        all_hreftag = find_all_href(soup)
        article_hreftag = find_article_href(soup)
        article = soup_find_article(soup)
    except:
        print("url_find_data: error", url_)
    
    return( (push_tag_l, push_userid_l, all_hreftag, article_hreftag, article))    


# In[25]:


def get_data():
    print("get_data start")
    df = pd.read_csv("HW1-1_3.0.csv")
    df = df.drop(df.columns[[0]], axis=1) 
    #testdf = df.head(100)
    start_time = time.time()
    buf_np = list(map(url_find_data, df["URL"].tolist()))
    print(time.time() - start_time)
    np.save('url_data', buf_np)


# In[ ]:


#if( sys.argv[1] == "push"):
#    get_data()


# # hw 2.5

# In[26]:


def make_push_table(np_):
    df_push = pd.DataFrame()
    for i in np_:
        try:
            buf = pd.DataFrame({'push': i[0], 'ID': i[1]})
            df_push  = df_push.append(buf)
        except:
            print(i)
    return(df_push)


# In[27]:


def find_push_boo(df_):
    buf = df_.groupby(['push']).count()
    buf2 = buf.loc[ ['推 ' , '噓 '] ,:]["ID"].tolist()
    return(buf2)


# In[28]:


def create_like_str(int_):
    return("like #" + str(int_))
def create_boo_str(int_):
    return("boo #" + str(int_))


# In[51]:


def push(start_date, end_date):
#start_date = 101
#end_date = 202
    print("push start")
    start_date = int(start_date)
    end_date = int(end_date)
    read_np = np.load('url_data.npy',allow_pickle = True )
    df = pd.read_csv("HW1-1_3.0.csv")
    df = df.drop(df.columns[[0]], axis=1) 
    buf1 = np.array(df.date) >= start_date
    buf2 = np.array(df.date) <= end_date
    legel_index = buf1 * buf2
    legal_np = read_np[legel_index]
    df_push = make_push_table(legal_np)
    buf = df_push[df_push["push"] != "→ "]
    cross_df = pd.crosstab(buf.ID, buf.push, margins=True)
    cross_df["pushID"] = cross_df.index
    push_df = cross_df.sort_values(by = ["推 ", "pushID"], ascending= [False, True])[1:11]
    boo_df = cross_df.sort_values(by = ["噓 ", "pushID"], ascending = [False, True])[1:11]
    buf1 = list(map(create_like_str, np.arange(11)[1:]))
    buf2 = list(map(create_boo_str, np.arange(11)[1:]))
    col1 = ["all like", "all boo"] + buf1 + buf2
    col2 = find_push_boo(df_push) + list(push_df.index) + list(boo_df.index)
    col3 = [" ", " "] + push_df["推 "].tolist() + boo_df["噓 "].tolist()
    col4 = []
    for i in np.arange(len(col2)):
        col4 = col4 + [  " " + str(col2[i]) + " " + str(col3[i]) ]
    #col3 = list(map(str, col3))
    output_df = pd.DataFrame({'name':col1, 'number': col4})
    output_name = "push[%s-%s].txt" % (start_date, end_date)
    output_df.to_csv(output_name, sep = ":",  index = False, header = False)


# In[31]:


if( sys.argv[1] == "push"):
    try:
        push(sys.argv[2], sys.argv[3])
    except:
        get_data()
        push(sys.argv[2], sys.argv[3])


# In[32]:


# HW3


# In[33]:


def one_to_allhref(tuple_):
    return( tuple_[2])


# In[39]:


def popular(start_date, end_date):
    start_date = int(start_date)
    end_date = int(end_date)
    print("popular excute")
    read_np = np.load('url_data.npy',allow_pickle = True )
    df = pd.read_csv("HW1-1_3.0.csv")
    df = df.drop(df.columns[[0]], axis=1) 
    bao_list = np.array(df["bao"])== ["爆"]
    buf1 = np.array(df.date) >= start_date
    buf2 = np.array(df.date) <= end_date
    legel_index = buf1 * buf2 * bao_list
    legal_np = read_np[legel_index]
    href_list = list(map(one_to_allhref, read_np))
    buf = list(itertools.compress(href_list, legel_index))
    merge_href = list(itertools.chain(*buf))
    buf = "number of popular articles: %d" % sum(legel_index)
    output_df = pd.DataFrame({'col1': [buf] + merge_href })
    output_name = "popular[%s-%s].txt" % (start_date, end_date)
    output_df.to_csv(output_name, sep = ",",  index = False, header = False)


# In[40]:


if( sys.argv[1] == "popular"):
    popular(sys.argv[2], sys.argv[3])


# In[41]:


# HW4


# In[45]:


def one_to_article(tuple_):
    return( tuple_[4])


# In[42]:


def article_if_keyword(str_):
    if( str_ == None ):
        return( False)
    else:
        return( keyword_glo in str_)


# In[44]:


def one_to_article_href(tuple_):
    return( tuple_[3])


# In[48]:


def keyword_search(keyword, start_date, end_date):
    print("keyword_search" + " start")
    start_date = int(start_date)
    end_date = int(end_date)
    keyword_glo = str(keyword)
    read_np = np.load('url_data.npy',allow_pickle = True )
    df = pd.read_csv("HW1-1_3.0.csv")
    df = df.drop(df.columns[[0]], axis=1) 
    buf1 = np.array(df.date) >= start_date
    buf2 = np.array(df.date) <= end_date
    article_list = list(map(one_to_article, read_np))
    keyword_list = list(map(article_if_keyword, article_list))
    legel_index = buf1 * buf2 * keyword_list
    a_href_list = list(map(one_to_article_href, read_np))
    buf = list(itertools.compress(a_href_list, legel_index))
    merge_href = list(itertools.chain(*buf))
    print("number of keyword articles: %d" % sum(legel_index))
    output_df = pd.DataFrame({'col1':  merge_href })
    output_name = "keyword(%s)[%s-%s].txt" % (keyword_glo, start_date, end_date)
    output_df.to_csv(output_name, sep = ":",  index = False, header = False)


# In[49]:


if( sys.argv[1] == "keyword"):
    keyword_glo = str(sys.argv[2])
    keyword_search(sys.argv[2], sys.argv[3], sys.argv[4])


# In[ ]:




