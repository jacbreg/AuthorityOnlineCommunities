#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:07:08 2017

@author: jacopo
"""
#c = root.xpath("row[@OwnerUserId='3']")

from bs4 import BeautifulSoup
import re
import os
#import xml.etree.ElementTree as ET
from lxml import etree
import pandas as pd
import time

'''
When file is too large, return iterator (unless specific request).
To access elements in iterator:
    for i, j in returned_iterator:
        print(j.attrib)

to parse all elements looking for specific rows:
    for i, j in returned_iterator:
        --- do something ---
        j.clear()
        while j.getprevious() is not None:
            del j.getparent()[0]
'''

class Posts(object):
    def __init__(self, qa_name, directory_data, threshold = 10**10, out_type='list'):
        '''
        out_type = 'list' or 'df' (dataframe)
        '''
        self.qa_name = qa_name
        self.directory_data = directory_data
        self.threshold = threshold
        self.out_type = out_type
        
        if os.stat(self.directory_data + self.qa_name + 'Posts.xml').st_size <=  self.threshold:
            self.tree = etree.parse(self.directory_data + self.qa_name + 'Posts.xml')
        else:
            self.tree = etree.iterparse(self.directory_data + self.qa_name + 'Posts.xml')
    
    def set_outtype(self,out_type):
        self.out_type = out_type
        
    def posts(self, set_condition = False): 

        if os.stat(self.directory_data + self.qa_name + 'Posts.xml').st_size >  self.threshold:
            print('File size is too large, it is returned iterator')
            return self.tree
#            tree = etree.iterparse(self.directory_data + self.qa_name + 'Posts.xml', events=('end',), tag='row')
#            
#            if set_condition == False:
#                return tree
#            else:            
#                posts = []
#                cond = str(input("introduce condition ('tag',value) : "))
#                for i, j in tree:
#                    tup = eval(cond)
#                    if j[tup[0]] == tup[1]:
#                        posts.append(dict(j.attrib))
#                    j.clear()
#                    while j.getprevious() is not None:
#                        del j.getparent()[0]
#                        
#                if self.out_type == 'list':
#                    return posts
#                elif self.out_type =='df':
#                    return pd.DataFrame(posts)
#                else:
#                    raise ValueError('value for out_type not valid')
                
        else:
            root = self.tree.getroot()
            all_rows = []
            for post in root:
                all_rows.append(dict(post.attrib))
                
            if self.out_type== 'list':
                return all_rows
            elif self.out_type=='df':
                return pd.DataFrame(all_rows)
            else:
                raise ValueError('value for out_type not valid')
    '''    
    def user_posts(self, user_id): # user_id is str
        df = pd.DataFrame(self.posts())
        df = df.set_index('Id')
        df = df.loc[df['OwnerUserId']==user_id,]
        
        return df'''
    
    def user_posts(self, user_id):
        
        if os.stat(self.directory_data + self.qa_name + 'Posts.xml').st_size >  self.threshold:
            tree = etree.iterparse(self.directory_data + self.qa_name + 'Posts.xml')
            output = []
            for event, element in tree:
                if 'OwnerUserId' in element.attrib.keys() and element.attrib['OwnerUserId'] == user_id:
                    output.append(dict(element.attrib))
                element.clear() # just for memory usage efficiency
                while element.getprevious() is not None: # just for memory usage efficiency
                    del element.getparent()[0] # just for memory usage efficiency
                    
            if self.out_type == 'list':
                return output
            elif self.out_type == 'df':    
                output_df = pd.DataFrame(output).set_index('Id')
                return output_df
            else:
                raise ValueError('value for out_type not valid')
        else:
            p = [dict(i.attrib) for i in self.tree.xpath("row[@OwnerUserId='%s']"%(user_id))]
            if not p:
                print('user has no posts')
                return
            
            if self.out_type== 'list':
                return p
            elif self.out_type=='df':
                return pd.DataFrame(p)
            else:
                raise ValueError('value for out_type not valid')
                
    def Id(self, post_id):
        ' missing code for data larger than threshold'
        return dict(self.tree.xpath("row[@Id='%s']"%(post_id))[0].attrib)
    
    
class Questions(Posts):
    def __init__(self, qa_name, directory_data, out_type= 'list', with_answers='y'):
        Posts.__init__(self, qa_name, directory_data, out_type=out_type)
        self.with_answers = with_answers
        
    def questions(self, user = None):
        if self.with_answers == 'y':
            if user == None:
                questions = [dict(i.attrib) for i in self.tree.xpath("row[@PostTypeId='1' and @AnswerCount!='0']")]
                if self.out_type == 'list':
                    return questions
                elif self.out_type == 'df':
                    return pd.DataFrame(questions)
                else:
                    raise ValueError("'out_type' input not defined or uncorrect")
            else:
                questions = []
                if self.out_type == 'list':
                    for post in self.user_posts(user):
                        if post['PostTypeId']=='1':
                            questions.append(post)
                elif self.out_type == 'df':
                    data = self.user_posts(user)
                    return data.loc[(data['PostTypeId']=='1') & (data['AnswerCount']!='0'),]
                else:
                    raise ValueError("'out_type' input not defined or uncorrect")
        elif self.with_answers == 'n':
            if user == None:
                questions = [dict(i.attrib) for i in self.tree.xpath("row[@PostTypeId='1']")]
            else:
                questions = []
                if self.out_type == 'list':
                    for post in self.user_posts(user):
                        if post['PostTypeId']=='1':
                            questions.append(post)
                elif self.out_type == 'df':
                    data = self.user_posts(user)
                    return data.loc[data['PostTypeId']=='1',]            
        else:
            raise ValueError("with_answer has to take either 'y' or 'n' value")
        
 
    # subset of questions with accepted answers
    def with_accept(self, user = None):
        if self.out_type == 'list':
            w_accept = []
            for i in self.questions(user):
                if 'AcceptedAnswerId' in i.keys():
                    w_accept.append(i)
            return w_accept
        elif self.out_type == 'df':
            data = self.questions(user)
            return data.dropna(subset=['AcceptedAnswerId']) # to check if this works
        
    
class Answers(Posts):
    def __init__(self, qa_name, directory_data, out_type='list'):
        Posts.__init__(self, qa_name, directory_data, out_type=out_type)
    
    def answers(self):
            answers = [dict(i.attrib) for i in self.tree.xpath("row[@PostTypeId='2']")]
            if self.out_type == 'list':
                return answers
            elif self.out_type == 'df':
                return pd.DataFrame(answers)
            else:
                raise ValueError("'out_type' input not defined or uncorrect")
                
    def user_answers(self, user_id):
            if self.out_type == 'list':
                answers = []
                for post in self.user_posts(user_id):
                    if post['PostTypeId']=='2':
                        answers.append(post)
                return answers
            elif self.out_type == 'df':
                data = self.user_posts(user_id)
                return data.loc[data['PostTypeId']=='2',]
            else:
                raise ValueError("'out_type' input not defined or uncorrect")

    def answers2questions(self, question_id): # answers to specific question
            answers = [dict(i.attrib) for i in self.tree.xpath("row[@PostTypeId='2' and @ParentId=%s]"%(question_id))]
            if self.out_type == 'list':
                return answers
            elif self.out_type == 'df':
                return pd.DataFrame(answers)
            else:
                raise ValueError("'out_type' input not defined or uncorrect")

        
        # subset of answers that have been accepted for some questions
    def accepted(self, question_withaccept = None): # argument can be an already existing object, resulted of the Questions().with_accept() method
        if question_withaccept == None:
            q_withaccept = Questions(self.qa_name, self.directory_data).with_accept()
        else:
            q_withaccept = question_withaccept
        
        if self.out_type == 'list':
            accepted_answers_ids = []
            for q in q_withaccept:
                accepted_answers_ids.append(q['AcceptedAnswerId'])
            accepted_answers = []
            for a in self.answers():
                if a['Id'] in accepted_answers_ids:
                    accepted_answers.append(a)
            return accepted_answers
        elif self.out_type == 'df':
            data_q = q_withaccept['AcceptedAnswerId']
            data_a = self.answers()
            return data_a.loc[data_a['Id'].isin(data_q),]


class Edits(object):
    def __init__(self, qa_name, directory_data, threshold = 10**9, out_type='list'):
        self.directory_data = directory_data
        self.qa_name = qa_name
        self.out_type = out_type
        self.threshold = threshold

        if os.stat(self.directory_data + self.qa_name + 'PostHistory.xml').st_size <=  self.threshold:
            self.tree = etree.parse(self.directory_data + self.qa_name + 'PostHistory.xml')
        else:
            self.tree = etree.iterparse(self.directory_data + self.qa_name + 'PostHistory.xml')
        
    def edits(self): # not coded yet case in which file size is above threshold
        if os.stat(self.directory_data + self.qa_name + 'PostHistory.xml').st_size >  self.threshold:
            print('File size is too large, it is returned iterator')
            return self.tree
        
        else:
            root = self.tree.getroot()
            all_edits = [dict(edit.attrib) for edit in root]
            if self.out_type == 'list':
                return all_edits
            elif self.out_type == 'df':
                return pd.DataFrame(all_edits)
          
    def post_edits(self, post_id, exclude_owner=False):
        edits = [dict(i.attrib) for i in self.tree.xpath("row[@PostId='%s']"%(post_id))]
        if self.out_type == 'list':
            if exclude_owner == True:
                p = Posts(self.directory_data, self.qa_name, threshold = self.threshold, out_type = self.out_type).Id(post_id)
                owner_id = p['OwnerUserId']
                newedits = []
                for edit in edits:
                    if not edit['UserId'] == owner_id:
                        newedits.append(edit)
                edits = newedits
            return edits
        
        elif self.out_type == 'df':
            dfedits = pd.DataFrame(edits)
            if exclude_owner == True:
                p = Posts(self.directory_data, self.qa_name, threshold = self.threshold, out_type = self.out_type).Id(post_id)
                owner_id = p['OwnerUserId']
                dfedits = dfedits[dfedits['UserId'] != owner_id]
    
            return dfedits
        
        else:
            raise ValueError("'out_type' input not defined or uncorrect")
    
    def user_edits(self, user_id, include_creation=False):
        ''' the 'include_creation' argument tells whether creation edits (type 1,2,3) are included (True) or not (False)'''
        if os.stat(self.directory_data + self.qa_name + 'PostHistory.xml').st_size >  self.threshold:
            tree = etree.iterparse(self.directory_data + self.qa_name + 'PostHistory.xml')
            output = []
            for event, element in tree:
                if include_creation == False:
                    if element.attrib['PostHistoryTypeId'] in ['1','2','3']:
                        continue
                if 'UserId' in element.attrib.keys() and element.attrib['UserId'] == user_id:
                    output.append(dict(element.attrib))
                element.clear() # just for memory usage efficiency
                while element.getprevious() is not None: # just for memory usage efficiency
                    del element.getparent()[0] # just for memory usage efficiency
                    
            if self.out_type == 'list':
                return output
            elif self.out_type == 'df':    
                output_df = pd.DataFrame(output).set_index('Id')
                return output_df
            else:
                raise ValueError('value for out_type not valid')
        else:
            if include_creation==False:
                p = [dict(i.attrib) for i in self.tree.xpath("row[@UserId='%s' and not(@PostHistoryTypeId='1') and not(@PostHistoryTypeId='2') and not(@PostHistoryTypeId='3')]"%(user_id))]
            elif include_creation == True:
                p = [dict(i.attrib) for i in self.tree.xpath("row[@UserId='%s']"%(user_id))]
                
            if not p:
                print('user has no edits')
                return pd.DataFrame()
            
            if self.out_type== 'list':
                return p
            elif self.out_type=='df':
                return pd.DataFrame(p)
            else:
                raise ValueError('value for out_type not valid')        
         

# full name, location, age, website, info extracted from about me (num di link?)

class Users(object):
    def __init__(self, qa_name, directory_data, threshold = 10**9, out_type='list'):
        self.qa_name = qa_name
        self.directory_data = directory_data
        self.threshold = threshold
        self.out_type = out_type
        
        if os.stat(self.directory_data + self.qa_name + 'Users.xml').st_size <=  self.threshold:
            self.tree = etree.parse(self.directory_data + self.qa_name + 'Users.xml')
        else:
            self.tree = etree.iterparse(self.directory_data + self.qa_name + 'Users.xml')
        
    def users(self, specific_user=None):
        
        if os.stat(self.directory_data + self.qa_name + 'Users.xml').st_size >  self.threshold:
            print('File size is too large, it is returned iterator')
            return self.tree
#            tree = etree.iterparse(self.directory_data + self.qa_name + 'Users.xml')
#            users = []
#            type_input = int(input('manual (1) or semi-manual (2)?'))
#            
#            if type_input == 1:
#                cond = str(input("introduce condition: "))
#                for i, usr in tree:
#                    usr = usr
#                    if eval(cond):
#                        users.append(dict(usr.attrib))
#                    usr.clear()
#                    while usr.getprevious() is not None:
#                        del usr.getparent()[0]
#                        
#            elif type_input == 2:
#                cond = str(input("introduce condition ('tag',value) : "))
#                for i, usr in tree:
#                    tup = eval(cond)
#                    if usr[tup[0]] == tup[1]:
#                        users.append(dict(usr.attrib))
#                    usr.clear()
#                    while usr.getprevious() is not None:
#                        del usr.getparent()[0]
#            
#            return users
        
        else:
            if specific_user == None:
                users = [dict(i.attrib) for i in self.tree.getroot() if i.attrib['Id']!='-1']
                
                if self.out_type == 'list':
                    return users
                elif self.out_type == 'df':
                    return pd.DataFrame(users)
                else:
                    raise ValueError("'out_type' input not defined or uncorrect")
            
            else:
                user = [dict(i.attrib) for i in self.tree.xpath("row[@Id='%s']"%(specific_user))]

                if self.out_type == 'list':
                    if len(user) == 1:
                        user = user[0]
                    return user
                elif self.out_type == 'df':
                    return pd.DataFrame(user)
                else:
                    raise ValueError("'out_type' input not defined or uncorrect")
        '''
        users = open(directory_data + qa_name + '/Users.xml').read()
        soup = BeautifulSoup(users, 'lxml')
        all_rows = soup.find_all('row')
        
        self.users = []
        for i in all_rows:
            if not i.attrs['id'] == '-1': # removes community user
                self.users.append(i.attrs)
        '''
    def with_fullname(self):
        '''
        a user name is full iff has the pattern; Name Surname
        '''
        import re
        
        u = []
        for i in self.users:
            if re.search('[A-Z][a-z]*\s[A-Z][a-z]*', i['DisplayName']):
                u.append(i)
        return u
    
    def ofPost(self, postid):
        ''' to be coded for big data '''
        post = Posts(qa_name= self.qa_name, directory_data=self.directory_data, out_type='df').Id(postid)
        if not 'OwnerUserId' in post.keys():
            if 'OwnerDisplayName' in post.keys():
                return post['OwnerDisplayName']
            else:
                return None
        return post['OwnerUserId']

class Votes(object):
    def __init__(self, qa_name, directory_data, threshold = 10**9, out_type='list'):
        '''
        out_type = 'list' or 'df' (dataframe)
        '''
        self.qa_name = qa_name
        self.directory_data = directory_data
        self.threshold = threshold
        self.out_type = out_type
        
        if os.stat(self.directory_data + self.qa_name + 'Votes.xml').st_size <=  self.threshold:
            self.tree = etree.parse(self.directory_data + self.qa_name + 'Votes.xml')
        else:
            self.tree = etree.iterparse(self.directory_data + self.qa_name + 'Votes.xml')  
            
    def votes(self):
        
        if os.stat(self.directory_data + self.qa_name + 'Votes.xml').st_size >  self.threshold:
            print('File size is too large, it is returned iterator')
            return self.tree
            
        else:
            root = self.tree.getroot()

            all_rows = []
            for votes in root:
                all_rows.append(dict(votes.attrib))
            
            if self.out_type== 'list':
                return all_rows
            elif self.out_type=='df':
                return pd.DataFrame(all_rows)
            else:
                raise ValueError('value for out_type not valid')     
    

    def user_favorites(self, user_id): 
        ''' this is gives the 'favorites' of the user'''
        
        favorites = [dict(i.attrib) for i in self.tree.xpath("row[@UserId='%s' and @VoteTypeId='5']"%(user_id))]
        
        if self.out_type== 'list':
            return favorites
        elif self.out_type=='df':
            return pd.DataFrame(favorites)
        else:
            raise ValueError('value for out_type not valid')  
    
    def voteHistPost(self, post_id, produceRep = False):
        ''' this is gives the vote history of a given post
        If produceRep==True --> only votes that give reputation to the owner of the post 
        '''
        
        if produceRep==False:
            votes = [dict(i.attrib) for i in self.tree.xpath("row[@PostId='%s']"%(post_id))]
        elif produceRep==True:
            votes = [dict(i.attrib) for i in self.tree.xpath("row[@PostId='%s' and (@VoteTypeId='2' or @VoteTypeId='1' or @VoteTypeId='3')]"%(post_id))]
        else:
            raise ValueError('value for produceRep not valid, set boulean (default False)') 
        
        if self.out_type== 'list':
            return votes
        elif self.out_type=='df':
            return pd.DataFrame(votes)
        else:
            raise ValueError('value for out_type not valid')          
        
    def user_bounties(self, user_id, types='posted'):
        ''' types can be 'posted' i.e. if the bounty as been proposed, or 'gained', if it was obtained bu user_id'''
        
        if types=='posted':
            bounties = [dict(i.attrib) for i in self.tree.xpath("row[@UserId='%s' and @VoteTypeId='8']"%(user_id))]
            
            if self.out_type== 'list':
                return bounties
            elif self.out_type=='df':
                return pd.DataFrame(bounties)
            else:
                raise ValueError('value for out_type not valid')     
        elif types=='gained':
            userposts = list(Posts(self.qa_name, self.directory_data, out_type='df').user_posts(user_id)['Id'])
            bounties = [dict(i.attrib) for i in self.tree.xpath("row[@VoteTypeId='9']")  if dict(i.attrib)['PostId'] in userposts]
            if self.out_type== 'list':
                return bounties
            elif self.out_type=='df':
                return pd.DataFrame(bounties)
            else:
                raise ValueError('value for out_type not valid') 
        else:
            ValueError("value for 'types' not valid")   
        
        
class Badges(object):
    def __init__(self, qa_name, directory_data, threshold = 10**9, out_type='list'):
        '''
        out_type = 'list' or 'df' (dataframe)
        '''
        self.qa_name = qa_name
        self.directory_data = directory_data
        self.threshold = threshold
        self.out_type = out_type

        if os.stat(self.directory_data + self.qa_name + 'Badges.xml').st_size <=  self.threshold:
            self.tree = etree.parse(self.directory_data + self.qa_name + 'Badges.xml')
        else:
            self.tree = etree.iterparse(self.directory_data + self.qa_name + 'Badges.xml')  
        
    def badges(self):
        
        if os.stat(self.directory_data + self.qa_name + 'Badges.xml').st_size >  self.threshold:
            print('File size is too large, it is returned iterator')
            return self.tree
            
        else:
            root = self.tree.getroot()

            all_rows = []
            for badge in root:
                all_rows.append(dict(badge.attrib))
            
            if self.out_type== 'list':
                return all_rows
            elif self.out_type=='df':
                return pd.DataFrame(all_rows)
            else:
                raise ValueError('value for out_type not valid')    
    '''    
    def user_badges(self, user_id): # user_id is str
        start = time.time()
        dfb = pd.DataFrame(self.badges())
        dfb = dfb.set_index('Id')
        dfb = dfb.loc[dfb['UserId']==user_id,]
        end = time.time()
        print(end-start)
        return dfb'''
    
    def user_badges(self, user_id):
        if os.stat(self.directory_data + self.qa_name + 'Badges.xml').st_size >  self.threshold:
            output = []
            for event, element in self.tree:
                if 'UserId' in element.attrib.keys() and element.attrib['UserId'] == user_id:
                    output.append(dict(element.attrib))
                element.clear() # just for memory usage efficiency
                while element.getprevious() is not None: # just for memory usage efficiency
                    del element.getparent()[0] # just for memory usage efficiency
            
            if self.out_type == 'list':
                return output
            elif self.out_type == 'df':    
                output_df = pd.DataFrame(output).set_index('Id')
                return output_df
            else:
                raise ValueError('value for out_type not valid')
            
        else:
            badges = [dict(i.attrib) for i in self.tree.xpath("row[@UserId='%s']"%(user_id))]
            if not badges:
                print('user has no badges')
                return pd.DataFrame()
            
            if self.out_type== 'list':
                return badges
            elif self.out_type=='df':
                return pd.DataFrame(badges)
            else:
                raise ValueError('value for out_type not valid')

class Comments(object):
    def __init__(self, qa_name, directory_data, threshold = 10**9, out_type='list'):
        '''
                out_type = 'list' or 'df' (dataframe)
        '''
        self.qa_name = qa_name
        self.directory_data = directory_data
        self.threshold = threshold
        self.out_type = out_type
        
        if os.stat(self.directory_data + self.qa_name + 'Comments.xml').st_size <=  self.threshold:
            self.tree = etree.parse(self.directory_data + self.qa_name + 'Comments.xml')
        else:
            self.tree = etree.iterparse(self.directory_data + self.qa_name + 'Comments.xml')       
            
    def comments(self):
        if os.stat(self.directory_data + self.qa_name + 'Comments.xml').st_size >  self.threshold:
            print('File size is too large, it is returned iterator')
            return self.tree
            
        else:
            root = self.tree.getroot()

            all_rows = []
            for comment in root:
                all_rows.append(dict(comment.attrib))
            
            if self.out_type== 'list':
                return all_rows
            elif self.out_type=='df':
                return pd.DataFrame(all_rows)
            else:
                raise ValueError('value for out_type not valid')  

    def user_comments(self, user_id):
        if os.stat(self.directory_data + self.qa_name + 'Comments.xml').st_size >  self.threshold:
            output = []
            for event, element in self.tree:
                if 'UserId' in element.attrib.keys() and element.attrib['UserId'] == user_id:
                    output.append(dict(element.attrib))
                element.clear() # just for memory usage efficiency
                while element.getprevious() is not None: # just for memory usage efficiency
                    del element.getparent()[0] # just for memory usage efficiency
            
            if self.out_type == 'list':
                return output
            elif self.out_type == 'df':    
                output_df = pd.DataFrame(output).set_index('Id')
                return output_df
            else:
                raise ValueError('value for out_type not valid')
            
        else:
            comments = [dict(i.attrib) for i in self.tree.xpath("row[@UserId='%s']"%(user_id))]
            if not comments:
                print('user has no comments')
                return pd.DataFrame()
            
            if self.out_type== 'list':
                return comments
            elif self.out_type=='df':
                return pd.DataFrame(comments)
            else:
                raise ValueError('value for out_type not valid')
            
###############################################################################
                # FUNCTIONS
###############################################################################
class SaveFig(object):
    def __init__(self, figure, name, folder_directory):
        if str(type(figure)) == "<class 'matplotlib.figure.Figure'>":
            f = figure
        else:
            f = figure.get_figure()
        f.savefig(folder_directory + '/' + name, dpi=500)

#decode dates: output is datetime object 
def date(string_date):
    dt = pd.to_datetime(string_date)
    return dt

def StackAPIdate(number): # to parse dates from data extracted from StackAPI
    return pd.to_datetime(number, origin='unix', unit='s')

def ymdhms(timestamp):
    return pd.Timestamp(timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute, timestamp.second)

def get_body_as_str(body):
    soup = BeautifulSoup(body, 'lxml')
    if soup.body == None:
        return None
    texts = soup.body.find_all()
    all_text = ''
    for i in texts:
        all_text = all_text + ' ' + i.text
    
    all_text = re.sub("[\[\].*?;:,()\-!<>]", ' ' , all_text)
    all_text = all_text.lower()
    return all_text

# get specific post corresponding to given post id
def post_from_id(post_id, directory_data, qa_name):
    tree = etree.parse(directory_data + qa_name + 'Posts.xml')
    post = [dict(i.attrib) for i in tree.xpath("row[@Id='%s']"%(post_id))][0]
    return post

# convert datetime to numbers (like egen x= group() in stata)
'''input is datetime series'''
def get_date2num(date, freq='D'):
    dr = pd.date_range(start=min(date), end=max(date), freq=freq)
    vrange = [i for i in range(len(dr))]
    dictval = dict(zip(dr,vrange))
#    dictval ={}
#    for i in range(len(dr)):
#        dictval[dr[i]] = vrange[i]
    return dictval
    