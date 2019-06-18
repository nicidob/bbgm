import argparse
import os
import sys
import shutil
import subprocess
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
from selenium import webdriver 

tables = {}


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--folder', default='combine',type=str, help='folder to save combine stats')

args = parser.parse_args()

for folder in [args.folder]:
    try:
        os.mkdir(folder)
        print("Directory {} created".format(folder)) 
    except FileExistsError:
        pass
driver = webdriver.Firefox()  
for year in range(20):
    target = os.path.join(args.folder,str(year) + '.html')

    # get the files
    if not os.path.exists(target):
        driver.get('https://stats.nba.com/draft/combine-anthro/#!?SeasonYear=20{:02d}-{:02d}'.format(year,year+1))
        with open(target, 'w') as f:
            f.write(driver.page_source)
        driver.get('http://google.com')
        
#     # load the data
#     with open(target,'rt') as fp:
#         data = fp.read()

#     # collect all the tables
#     m = re.findall(r'<!--[ \n]*(<div[\s\S\r]+?</div>)[ \n]*-->',data)
#     m2 = re.findall(r'(<div class="table_outer_container">[ \n]*<div class="overthrow table_container" id="div_roster">[\s\S\r]+?</table>[ \n]*</div>[ \n]*</div>)',data)
#     m3 = re.findall(r'(<div class="table_outer_container">[ \n]*<div class="overthrow table_container" id="div_contracts">[\s\S\r]+?</table>[ \n]*</div>[ \n]*</div>)',datac)
#     m = m2 + m + m3 
#     print(target,len(m))
#     tables[team] = {}
#     for test_table in m:
#         try:
#             soup = BeautifulSoup(test_table,features="lxml")
#             table_id = str(soup.find('table').get('id'))

#             if table_id == ['team_and_opponent']:
#                 continue
#             soup.findAll('tr')

#             table_size = {'shooting':2,'pbp':1,'playoffs_shooting':2,'playoffs_pbp':1,'contracts':1}

#             # use getText()to extract the text we need into a list
#             headers = [th.getText() for th in soup.findAll('tr')[table_size.get(table_id,0)].findAll('th')]

#             # exclude the first column as we will not need the ranking order from Basketball Reference for the analysis
#             start_col = 1
#             if table_id in ['contracts','injury']:
#                 start_col = 0

#             headers = headers[start_col:]
#             rows = soup.findAll('tr')[start_col:]
#             player_stats = [[td.getText() for td in rows[i].findAll('td')]
#                         for i in range(len(rows))]

#             if table_id in ['contracts']:
#                 player_status = [[td.get('class') for td in rows[i].findAll('td')]
#                         for i in range(len(rows))]
#                 status_array = []
#                 for status in player_status:
#                     if len(status) > 0:
#                         s2 = [False] + [s[-1] in ['salary-pl','salary-et','salary-tm'] for s in status[1:]]
#                     else:
#                         s2 = np.array([])
#                     status_array.append(s2)
#                 status_array = np.array(status_array)
#                 player_stats_new = []
#                 for a,b in zip(status_array,player_stats):
#                     b_new = []
#                     for c,d in zip(a,b):
#                         b_new.append(d if not c else '')
#                     player_stats_new.append(b_new)
#                 player_stats = player_stats_new
#             if table_id in ['contracts','injury']:
#                 player_names = [[td.getText() for td in rows[i].findAll('th')]
#                             for i in range(len(rows))]
#                 player_stats = [a + b for a,b in zip(player_names[1:],player_stats[1:])]
#             headers[0] = 'Name'
#             stats = pd.DataFrame(player_stats, columns = headers).set_index('Name')
#             if table_id in ['contracts']:
#                 stats = stats.drop(['Player'])
#                 stats = stats.iloc[:stats.index.get_loc('')]

#             # drop nan
#             stats = stats[~ stats.index.isin([None])]
#             # convert to float
#             obj_cols = stats.loc[:, stats.dtypes == object]
#             conv_cols = obj_cols.apply(pd.to_numeric, errors = 'ignore')
#             stats.loc[:, stats.dtypes == object] = conv_cols

#             #print(table_id,stats.index)
#             tables[team][table_id]= stats.fillna('')

#         except:
#             pass
#             #print('FAILED TO PARSE ' +str(soup.find('table').get('id') ))
# with open('combine_{}.pkl'.format(args.year),'wb') as fp:
#     pickle.dump(tables,fp)