
teams = ["air-force",
"akron",
"alabama",
"alabama-birmingham",
"appalachian-state",
"arizona",
"arizona-state",
"arkansas",
"arkansas-state",
"army",
"auburn",
"ball-state",
"baylor",
"boise-state",
"boston-college",
"bowling-green-state",
"brigham-young",
"buffalo",
"california",
"central-florida",
"central-michigan",
"charlotte",
"cincinnati",
"clemson",
"coastal-carolina",
"colorado",
"colorado-state",
"connecticut",
"duke",
"east-carolina",
"eastern-michigan",
"florida",
"florida-atlantic",
"florida-international",
"florida-state",
"fresno-state",
"georgia",
"georgia-southern",
"georgia-state",
"georgia-tech",
"hawaii",
"houston",
"illinois",
"indiana",
"iowa",
"iowa-state",
"kansas",
"kansas-state",
"kent-state",
"kentucky",
"liberty",
"louisiana-lafayette",
"louisiana-state",
"louisiana-tech",
"louisiana-monroe",
"louisville",
"marshall",
"maryland",
"massachusetts",
"memphis",
"miami-fl",
"miami-oh",
"michigan",
"michigan-state",
"middle-tennessee-state",
"minnesota",
"mississippi",
"mississippi-state",
"missouri",
"navy",
"nebraska",
"nevada",
"nevada-las-vegas",
"new-mexico",
"new-mexico-state",
"north-carolina",
"north-carolina-state",
"north-texas",
"northern-illinois",
"northwestern",
"notre-dame",
"ohio",
"ohio-state",
"oklahoma",
"oklahoma-state",
"old-dominion",
"oregon",
"oregon-state",
"penn-state",
"pittsburgh",
"purdue",
"rice",
"rutgers",
"san-diego-state",
"san-jose-state",
"south-alabama",
"south-carolina",
"south-florida",
"southern-california",
"southern-methodist",
"southern-mississippi",
"stanford",
"syracuse",
"temple",
"tennessee",
"texas",
"texas-am",
"texas-christian",
"texas-state",
"texas-tech",
"texas-el-paso",
"texas-san-antonio",
"toledo",
"troy",
"tulane",
"tulsa",
"ucla",
"utah",
"utah-state",
"vanderbilt",
"virginia",
"virginia-tech",
"wake-forest",
"washington",
"washington-state",
"west-virginia",
"western-kentucky",
"western-michigan",
"wisconsin",
"wyoming"]

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
import itertools
import io

tables = {}


teams = sorted(teams)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--year', default=2018,type=int, help='season to save')
parser.add_argument('--folder', default='cfb_team',type=str, help='folder to save year stats')
parser.add_argument('--rfolder', default='cfb_roster',type=str, help='rosters folder')
parser.add_argument('--ow', action='store_true',help='overwrite existing')
parser.add_argument('--process', action='store_true',help='only process files, no fetching')

args = parser.parse_args()

for folder in [args.folder,args.rfolder]:
    try:
        os.mkdir(folder)
        print("Directory {} created".format(folder))
    except FileExistsError:
        pass


for team in teams:
    target = os.path.join(args.folder,team + str(args.year) + '.html')
    rtarget = os.path.join(args.rfolder,team + str(args.year) + '.html')

    if args.process:
        if not os.path.exists(target):
            continue
    # get the files
    else:
        if args.ow or not os.path.exists(target):
            subprocess.call(['wget','-O',target,
            'https://www.sports-reference.com/cfb/schools/{}/{}.html'.format(team,args.year)])
            fs = os.path.getsize(target)
            if fs < 10:
                os.remove(target)
                continue
        if args.ow or not os.path.exists(rtarget):
            subprocess.call(['wget','-O',rtarget,
            'https://www.sports-reference.com/cfb/schools/{}/{}-roster.html'.format(team,args.year)])
            fs = os.path.getsize(rtarget)
            if fs < 10:
                os.remove(rtarget)
                continue
    # load the data
    try:
        with open(target,'rt') as fp:
            data = fp.read()
        with open(rtarget,'rt') as fp:
            rdata = fp.read()
    except:
        with open(target,'rt',encoding='latin-1') as fp:
            data = fp.read()
        with open(rtarget,'rt',encoding='latin-1') as fp:
            rdata = fp.read()   
    # collect all the tables
    try:
        m = re.findall(r'<!--[ \n]*(<div[\s\S\r]+?</div>)[ \n]*-->',data)
        m2 = re.findall(r'(<div class="table_outer_container">[ \n]*<div class="overthrow table_container" id="div_roster">[\s\S\r]+?</table>[ \n]*</div>[ \n]*</div>)',rdata)
        m3 = re.findall(r'<!--[ \n]*(<div[\s\S\r]+?</div>)[ \n]*-->',rdata)

        m = m2 + m + m3
        print(target,len(m),len(m3))
        tables[team] = {}

        bs = BeautifulSoup(data,features="lxml")

        tables[team]['logo'] = re.findall('(http.*png)',str(bs.find_all('img',{"class": "teamlogo"})[0]))[0]
        tables[team]['name'] = re.findall('<title>{} (.*) Stats'.format(args.year),data)[0]
        tables[team]['conf'] = re.findall('<a href="/cfb/conferences/(.*)/{}.html">(.*)</a>'.format(args.year),data)[0]
    except:
        continue
    for test_table in m:
        try:
            soup = BeautifulSoup(test_table,features="lxml")
            table_id = str(soup.find('table').get('id'))

            if table_id in ['team_and_opponent','team_td_log','opp_td_log']:
                continue
            soup.findAll('tr')

            table_size =  {'defense_and_fumbles':1,'passing':1, 'rushing_and_receiving' :1,'returns' :1,'kicking' :1,'defense' :1,'kicking_and_punting':1,'scoring':1}

            # use getText()to extract the text we need into a list
            headers = [th.getText() for th in soup.findAll('tr')[table_size.get(table_id,0)].findAll('th')]

            # exclude the first column as we will not need the ranking order from Basketball Reference for the analysis
            start_col = 1
            if table_id in ['contracts','injury','on_off','on_off_p','roster']:
                start_col = 0

            headers = headers[start_col:]
            rows = soup.findAll('tr')[start_col:]
            player_stats = [[td.getText() for td in rows[i].findAll('td')]
                        for i in range(len(rows))]

            if table_id in ['contracts','roster']:
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
            if table_id in ['contracts','injury','on_off','on_off_p','roster']:
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

            stats = stats.fillna('')

            if True and 'on_off' in table_id:
                stats = stats.iloc[~ stats.index.get_loc('Player')]
                stats = stats.loc[~ (stats.Split == '')]
                stats.index = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in [_ for _ in stats.index if _!='']))

            #print(table_id,stats.index)
            tables[team][table_id]= stats
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(table_id)
            print(e)
            print(headers)
            raise
            print('FAILED TO PARSE ' +str(soup.find('table').get('id') ))
with open('cfb_tables_{}.pkl'.format(args.year),'wb') as fp:
    pickle.dump(tables,fp)
