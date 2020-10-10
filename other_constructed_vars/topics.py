#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:34:31 2020

@author: jacopo
"""

'''
code to cluster tags in topics
'''

import classes as cl
import pandas as pd
import re
from collections import Counter
import itertools
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt # needs to be imported from plotting graphs, even if not directly called
import matplotlib
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns
sns.set()

directory_data = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/'
#qa_name = 'apple/'
qa_name = 'ell/'

out_dir = directory_data + qa_name
#out_fig = '/home/jacopo/Documenti/scuola/università/TSE/PhD/my_projects/stackoverflow/figures/'
out_fig = '/home/jacopo/Documenti/scuola/università/TSE/PhD/my_projects/stackoverflow/figures_ELL/'

# all questions
allposts = cl.Posts(qa_name, directory_data, out_type='df').posts()
qposts = allposts.loc[allposts['PostTypeId']=='1',]

# recover synonims and substitute with targets
headers = {
    'User-Agent': 'jacrobot',
    'From': 'jak117@hotmail.it'}

#url = 'https://apple.stackexchange.com/tags/synonyms'
url = 'https://ell.stackexchange.com/tags/synonyms'

r = requests.get(url, headers=headers)
content = r.content
soup = BeautifulSoup(content, 'html.parser')
# find num of pages - THIS DOES NOT WORK IF THERE'S NOT ONE BOTTON FOR EACH PAGE. IF IT'S LIKE 1,2,3...50, THEN SEE BELOW UNDER SECTION "### USING TAG DESCRIPTIONS"
numpages = len(soup.find('div',{'class':"s-pagination pager fl"}).find_all('a',{'class':'s-pagination--item'}))
# extract synonims
synonyms_dict = {}
for pagenum in range(1,numpages+1):
    nr = requests.get(url + '?page=%d'%(pagenum))
    nsoup = BeautifulSoup(nr.content, 'html.parser')
    synonyms = nsoup.find_all('tr', {'class': re.compile('synonym-[0-9]+')})
    for syn in synonyms:
        try:
            key = syn.find('a',{'title':re.compile('show questions tagged.+')}).text
            value = syn.find('a',{'title':re.compile('Synonyms for.+')}).text
            synonyms_dict[key] = value
        except:
            continue

pd.to_pickle(synonyms_dict, out_dir + 'tag_synonyms_dict.pkl') 
synonyms_dict = pd.read_pickle(out_dir + 'tag_synonyms_dict.pkl')

# all tags 
qposts.loc[:,'TagList'] = qposts['Tags'].apply(lambda x: re.findall('<(.+?)>',x))

# freq distribution of tags in all questions (SYNONIMS NOT DROPPED)
alltags = [item for sublist in qposts['TagList'] for item in sublist]
alltagsXfreq = Counter(alltags)
alltagsXfreqDF = pd.DataFrame({'tags':list(alltagsXfreq.keys()),'freq':list(alltagsXfreq.values())})
alltagsXfreqDF.loc[:,'tags'] = alltagsXfreqDF['tags'].apply(lambda x: synonyms_dict[x] if x in synonyms_dict.keys() else x )
alltagsXfreqDF = alltagsXfreqDF.groupby('tags')['freq'].sum()

# construct matrix filled with number of co-occurrences
cooccurring = []
for l in qposts['TagList'].tolist():
    cooccurring.extend(list(itertools.permutations(l, 2)))
cooccurring = Counter(cooccurring)
data= pd.DataFrame(cooccurring.keys(), columns=['index','columns'])
data.loc[:,'freq'] = cooccurring.values()

# correct synonyms
data.loc[:,'index'] = data['index'].apply(lambda x: synonyms_dict[x] if x in synonyms_dict.keys() else x )
data.loc[:,'columns'] = data['columns'].apply(lambda x: synonyms_dict[x] if x in synonyms_dict.keys() else x )

# aggregate
data = data.groupby(['index','columns'])['freq'].sum().reset_index()

data.to_csv(out_dir + 'tagsfreq.csv', index=False) # data with frequency a tag is appearing with each other tag 
data = pd.read_csv(out_dir + 'tagsfreq.csv')

############### IDENTIFYING TOPICS ########################################################### 

### HIERARCHICAL CLUSTERING ### 
# drop too rare tags 
quantile25 = alltagsXfreqDF.quantile(0.25)
raretags = alltagsXfreqDF.loc[alltagsXfreqDF<quantile25].index.tolist()
dtaplot = data.loc[(~data['index'].isin(raretags)) & (~data['columns'].isin(raretags))]
dtaplot = dtaplot.set_index(['index','columns']).unstack(fill_value=0)
v = dtaplot.values / dtaplot.sum(axis=1).values
dtaplotrel = pd.DataFrame(v, index=dtaplot.index, columns=dtaplot.columns)

# plot (not nice)
sns.clustermap(dtaplotrel,method='centroid' )

# create hierarchical tree
test = linkage(dtaplot.values, method='centroid')
# plot it
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(test)
# cut tree at arbitrary value to identify topics
a = fcluster(test,1.15)
len(np.unique(a))
adf = pd.DataFrame(a, index=dtaplot.index, columns=['cluster'])
adf.sort_values(by='cluster', inplace=True)


### HIERARCHICAL CLUSTERING using question words as features ###
# first create a df with col1=tag col2=text of question where tag was assigned
# if multiple tag where assigned to a question, assign the question's text to each of the tags
qtextdata = qposts.copy()
qtextdata.loc[:,'numtags'] = qtextdata['TagList'].apply(lambda x: len(x))
maxnumtags = qtextdata['numtags'].max()
dfs = []
for tagnum in range(maxnumtags):
    d = qposts.loc[:,['TagList','Body']]
    d.loc[:,'tag'] = d['TagList'].apply(lambda x: x[tagnum] if len(x)>tagnum else 0) # where tag is zero has to be dropped
    d.loc[:,'text'] = d['Body'].apply(lambda x: BeautifulSoup(x,parser='html').text)
    dfs.append(d[['tag','text']])

# concatenate
dfs = pd.concat(dfs)

# drop if tag == 0
dfs = dfs.loc[dfs['tag']!=0]

# add space at the end of text
dfs.loc[:,'text'] = dfs['text'].apply(lambda x: x+' ')

# substitute synonims
dfs.loc[:,'tag'] =dfs['tag'].apply(lambda x: synonyms_dict[x] if x in synonyms_dict.keys() else x )

# group by tag
dfs = dfs.groupby('tag')['text'].sum() # summing on string it concatenates strings

# vectorize text data
vectorizer = CountVectorizer()
V = vectorizer.fit_transform(dfs.tolist())
V = V.toarray()

# save dataframe
Vdf = pd.DataFrame(V, index=dfs.index)
Vdf.to_csv(out_dir + 'tag_2wordvec.csv') # dataframe with index=tags, columns=words, values= word count in questions

# hierarchical clustering 
hc_V = linkage(V, method='ward')
# plot it
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(hc_V)

clusters = fcluster(hc_V,t=1.15)
len(np.unique(clusters))
hcVdf = pd.DataFrame(clusters, index=dfs.index, columns=['cluster'])
hcVdf.sort_values(by='cluster', inplace=True)



### K-MEANS clustering ###
datakm = data.copy()
datakm.set_index(['index','columns'], inplace=True)
datakm = datakm.unstack().fillna(0)
datakm.columns = datakm.columns.droplevel()
# remove tags appearing once at max
tags2keep = [i for i in datakm.columns if i in alltagsXfreqDF['tags'].tolist()]
datakm = datakm.loc[tags2keep,tags2keep]

# k-means clustering
fit = KMeans(n_clusters=10).fit(datakm.values)
fit = pd.DataFrame(index= datakm.index, data=fit.labels_)

### FIRST TAG IN LIS OF TAGS: often it is object of interest (iphone, ...) (while other tags it's more on the specific problem)
qposts.loc[:,'firsttag'] = qposts['TagList'].apply(lambda x: x[0])
# topics are the tags most frequently occurring as first tag
firstagdf = qposts['firsttag'].value_counts()


### using graphs
datag = data.copy()
datag.loc[:,'forgraph'] = datag.apply(lambda x: (x['index'],x['columns'],{'weight':x['freq']}), axis=1)

G = nx.Graph()
G.add_edges_from(datag['forgraph'].tolist())

### MEASURE OF CENTER IN SUBGRAPHS - topics are the most central
centr_subgraph = nx.subgraph_centrality(G)
centr_subgraphdf = pd.DataFrame({'tag':list(centr_subgraph.keys()), 'subgraphcentrality':list(centr_subgraph.values())})
centr_subgraphdf = centr_subgraphdf.sort_values(by='subgraphcentrality', ascending=False)

### MEASURE OF CENTER IN GRAPH - topics are the most central
centrality = nx.degree_centrality(G)
centrdf = pd.DataFrame({'tag':list(centrality.keys()), 'centrality':list(centrality.values())})
centrdf = centrdf.sort_values(by='centrality', ascending=False)
##########################################################################################
#### FINAL CHOICE FOR MAKING TOPICS !! ###################################################
### PAGE RANKING VALUE - topics are the ones with highest page ranking vlue
pagerk = nx.pagerank(G)
pagerkdf = pd.DataFrame({'tag':list(pagerk.keys()), 'pagerank':list(pagerk.values())})
pagerkdf = pagerkdf.sort_values(by='pagerank', ascending=False)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.linspace(pagerkdf['pagerank'].min(),pagerkdf['pagerank'].max(),num=len(pagerkdf)),pagerkdf['pagerank'].values)
#FOR APPLE after first 4, curve slows down badly. use first 4 as main nodes
#FOR ELL: after first 6, curve slows down badly. use first 6 as main nodes
cn = pagerkdf['tag'].iloc[:6].tolist()
vc = nx.voronoi_cells(G, center_nodes=cn) # keys are tags selected as topics page-rank, values are tags in each topic
pd.to_pickle(vc,out_dir + 'topicsFromTags.pkl') # it doesn't look too good the allocation of tags to topics, but i still believe that is the most appropriate method
topics = pd.read_pickle(out_dir + 'topicsFromTags.pkl')

nodes = []
colors = []
ntopics = len(topics)
topic_names = list(topics.keys())
cmap = matplotlib.cm.get_cmap('Set2')
for i in range(ntopics):
    lists = topics[topic_names[i]]
    nodes.extend(list(lists))
    c = [cmap(i) for j in range(len(lists))]
    colors.extend(c)
cmapkeys = matplotlib.cm.get_cmap('Dark2')
nodes.extend(topic_names)
colors.extend(list(cmapkeys.colors)[:6])

datag['maxw'] = datag.groupby('index')['freq'].transform(max)
datared = datag.loc[datag['freq']==datag['maxw']]
datared['tups'] = datared.apply(lambda x: (x['index'],x['columns']), axis=1)

edges = datared['tups'].tolist()
weights = datared['freq'].astype(float).tolist()

minw = min(weights)
maxw = max(weights)

nx.draw(G, nodelist=nodes, node_color=colors, with_labels=False, node_size=60, 
                 alpha=1, edge_cmap=matplotlib.cm.get_cmap('Wistia'), edgelist=edges,
                 edge_color=weights, edge_vmin=minw, edge_vmax=maxw)
# drow word clouds for each topic
for name in topic_names:
    tags = alltagsXfreqDF.loc[alltagsXfreqDF.index.isin(topics[name]),]
    tags = tags.to_dict()
    from wordcloud import WordCloud
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(tags)
    f = plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    f = f.get_figure()
    f.savefig(out_fig + 'topic_{}.png'.format(name), dpi=500)
    plt.close()
#############################################################################################

### DOMINATING SET 
from networkx.algorithms import approximation
ds = approximation.dominating_set.min_weighted_dominating_set(G)

c = nx.find_cliques(G)
from networkx.algorithms.community import k_clique_communities
k = k_clique_communities(G, 4)
klist = [i for i in k]

### USING TAG DESCRIPTIONS

headers = {
    'User-Agent': 'jacrobot',
    'From': 'jak117@hotmail.it'}

url = 'https://apple.stackexchange.com/tags'
r = requests.get(url, headers=headers)
content = r.content
soup = BeautifulSoup(content, 'html.parser')
# find num of pages
numpages = soup.find('div',{'class':"s-pagination pager fr"}).find_all('a',{'class':'s-pagination--item'})
numpages = max([int(re.findall('[0-9]+',i.text)[0]) for i in numpages if re.search('[0-9]+',i.text)])
# extract synonims
tagdescription_dict = {}
for pagenum in range(1,numpages+1):
    nr = requests.get(url + '?page=%d'%(pagenum))
    nsoup = BeautifulSoup(nr.content, 'html.parser')
    descripts = nsoup.find_all('div', {'class': 's-card js-tag-cell grid fd-column'})
    for tag in descripts:
        try:
            tagname = tag.find('a',{'class':'post-tag'}).text
            descript = tag.find('div',{'class':'grid--cell fc-light mb12 v-truncate4'}).text
            tagdescription_dict[tagname] = descript
        except:
            continue

pd.to_pickle(tagdescription_dict, out_dir + 'tagdescription.pkl')
tagdescription_dict = pd.read_pickle(out_dir + 'tagdescription.pkl')

# create matrix: rows are tag descriptions, columns are words, values are counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(list(tagdescription_dict.values()))

simil = cosine_similarity(X)
dist = cosine_distances(X)

test = linkage(X.todense())
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(test)
a = fcluster(test,1.18)






taglist = list(tagdescription_dict.keys())
tag2similar = {}
topics = []
for tagnum in range(simil.shape[0]):
    row = np.where(simil[tagnum,:]>=0.4,np.array(taglist),'').tolist()
    similar = [i for i in row if i!='']    
    tag2similar[taglist[tagnum]] = similar
    if not similar in topics:
        topics.append(similar)

uniquetopics = Counter(tag2similar.values())

simildf = pd.DataFrame(simil, index=taglist, columns=taglist)

test = linkage(simil)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(test)

dtaplotrel = pd.DataFrame(X.todense(), index=taglist)
sns.clustermap(dtaplotrel)