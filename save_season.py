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

tables = {}

teams = ['MIL','TOR','PHI','BOS','IND','BRK','ORL','DET','CHO','MIA',\
         'WAS','ATL','CHI','CLE','NYK','GSW','DEN','POR','HOU','UTA',\
         'OKC','SAS','LAC','SAC','LAL','MIN','MEM','NOP','DAL','PHO']
teams = reversed(teams)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--year', default=2019,type=int, help='season to save')
parser.add_argument('--folder', default='teams',type=str, help='folder to save year stats')
parser.add_argument('--cfolder', default='contracts',type=str, help='contracts folder')

args = parser.parse_args()

for folder in [args.folder,args.cfolder]:
    try:
        os.mkdir(folder)
        print("Directory {} created".format(folder)) 
    except FileExistsError:
        pass
    

for team in teams:
    target = os.path.join(args.folder,team + str(args.year) + '.html')
    ctarget = os.path.join(args.cfolder,team + '.html')

    # get the files
    if not os.path.exists(ctarget):
        subprocess.call(['wget','-O',ctarget,
        'https://www.basketball-reference.com/contracts/{}.html'.format(team)])
    if not os.path.exists(target):
        subprocess.call(['wget','-O',target,
        'https://www.basketball-reference.com/teams/{}/{}.html'.format(team,args.year)])

    # load the data
    with open(target,'rt') as fp:
        data = fp.read()
    with open(ctarget,'rt') as fp:
        datac = fp.read()
    # collect all the tables
    m = re.findall(r'<!--[ \n]*(<div[\s\S\r]+?</div>)[ \n]*-->',data)
    m2 = re.findall(r'(<div class="table_outer_container">[ \n]*<div class="overthrow table_container" id="div_roster">[\s\S\r]+?</table>[ \n]*</div>[ \n]*</div>)',data)
    m3 = re.findall(r'(<div class="table_outer_container">[ \n]*<div class="overthrow table_container" id="div_contracts">[\s\S\r]+?</table>[ \n]*</div>[ \n]*</div>)',datac)
    m = m2 + m + m3 
    print(target,len(m))
    tables[team] = {}
    for test_table in m:
        try:
            soup = BeautifulSoup(test_table,features="lxml")
            table_id = str(soup.find('table').get('id'))

            if table_id == ['team_and_opponent']:
                continue
            soup.findAll('tr')

            table_size = {'shooting':2,'pbp':1,'playoffs_shooting':2,'playoffs_pbp':1,'contracts':1}

            # use getText()to extract the text we need into a list
            headers = [th.getText() for th in soup.findAll('tr')[table_size.get(table_id,0)].findAll('th')]

            # exclude the first column as we will not need the ranking order from Basketball Reference for the analysis
            start_col = 1
            if table_id in ['contracts','injury']:
                start_col = 0

            headers = headers[start_col:]
            rows = soup.findAll('tr')[start_col:]
            player_stats = [[td.getText() for td in rows[i].findAll('td')]
                        for i in range(len(rows))]

            if table_id in ['contracts']:
                player_status = [[td.get('class') for td in rows[i].findAll('td')]
                        for i in range(len(rows))]
                status_array = []
                for status in player_status:
                    if len(status) > 0:
                        s2 = [False] + [s[-1] in ['salary-pl','salary-et','salary-tm'] for s in status[1:]]
                    else:
                        s2 = np.array([])
                    status_array.append(s2)
                status_array = np.array(status_array)
                player_stats_new = []
                for a,b in zip(status_array,player_stats):
                    b_new = []
                    for c,d in zip(a,b):
                        b_new.append(d if not c else '')
                    player_stats_new.append(b_new)
                player_stats = player_stats_new
            if table_id in ['contracts','injury']:
                player_names = [[td.getText() for td in rows[i].findAll('th')]
                            for i in range(len(rows))]
                player_stats = [a + b for a,b in zip(player_names[1:],player_stats[1:])]
            headers[0] = 'Name'
            stats = pd.DataFrame(player_stats, columns = headers).set_index('Name')
            if table_id in ['contracts']:
                stats = stats.drop(['Player'])
                stats = stats.iloc[:stats.index.get_loc('')]

            # drop nan
            stats = stats[~ stats.index.isin([None])]
            # convert to float
            obj_cols = stats.loc[:, stats.dtypes == object]
            conv_cols = obj_cols.apply(pd.to_numeric, errors = 'ignore')
            stats.loc[:, stats.dtypes == object] = conv_cols

            #print(table_id,stats.index)
            tables[team][table_id]= stats.fillna('')

        except:
            pass
            #print('FAILED TO PARSE ' +str(soup.find('table').get('id') ))
with open('tables_{}.pkl'.format(args.year),'wb') as fp:
    pickle.dump(tables,fp)