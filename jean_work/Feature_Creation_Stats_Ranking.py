#!/usr/bin/env python
# coding: utf-8

# In[278]:


import pandas as pd
from IPython.display import clear_output


# In[233]:


data_players=pd.read_csv(r'processed_data/player_profiles_clean.csv').iloc[:,1:]
all_games=pd.read_csv(r'processed_data/all_match_91_2020.csv')
ranking_all=pd.read_csv(r'processed_data/ranking_91_2020_clean.csv').iloc[:,1:]


# In[193]:


len_ref=len(all_games)
len_ref


# First, need to group 'duplicates' in the dataframe

# In[218]:


data_players=data_players.drop_duplicates(['player_name','year'])
ranking_all=ranking_all.groupby(['unique_id','player_name','year','week_title']).mean().reset_index()


# # Create the ranking of the player for the given match

# In[234]:


all_games=all_games.merge(ranking_all[['unique_id','rank_number']],how='left',left_on='id_winner', right_on='unique_id').rename({'rank_number':'rank_winner'},axis=1).drop('unique_id',axis=1)
all_games=all_games.merge(ranking_all[['unique_id','rank_number']],how='left',left_on='id_loser', right_on='unique_id').rename({'rank_number':'rank_loser'},axis=1).drop('unique_id',axis=1)


# # Add the height, birthdate, weight, player_hand, handedness, backhand, turned_pro

# In[295]:


data_demo_players=data_players.drop_duplicates('player_name')

all_games=all_games.merge(data_demo_players[['turned_pro','birthdate','player_hand','player_ht','weight_kg','handedness','backhand','player_name']]
                ,how='left',left_on='winner', right_on='player_name').rename({'turned_pro':'turned_pro_winner',
                                                                            'birthdate':'birthdate_winner',
                                                                            'player_hand':'player_hand_winner',
                                                                            'player_ht':'winner_height',
                                                                            'weight_kg':'winner_weight',
                                                                            'handedness':'winner_handedness',
                                                                            'backhand':'winner_backhand'},axis=1).drop('player_name',axis=1)

all_games=all_games.merge(data_demo_players[['turned_pro','birthdate','player_hand','player_ht','weight_kg','handedness','backhand','player_name']]
                ,how='left',left_on='loser', right_on='player_name').rename({'turned_pro':'turned_pro_loser',
                                                                            'birthdate':'birthdate_loser',
                                                                            'player_hand':'player_hand_loser',
                                                                            'player_ht':'loser_height',
                                                                            'weight_kg':'loser_weight',
                                                                            'handedness':'loser_handedness',
                                                                            'backhand':'loser_backhand'},axis=1).drop('player_name',axis=1)


# # Adding weighted average features based on time before. 
# 
# here I did an exponential average with a time of 2. <span class="burk">That can be changed, as well as the sommothing parameter (2 by default in python.</span>
# 
# ## Games features

# In[205]:


def exponential_avg_stats_players(feature_name,player_name,time_period):
    data=data_players[data_players['player_name']==player_name]
    return (data[feature_name].ewm(span=time_period).mean())

exponential_avg_stats_players('ace','piros_z.',2)


# In[223]:


list_features_exp_avg=['ace',
       'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved',
       'bpFaced']

for feature in list_features_exp_avg:
    for player in data_players['player_name'].unique():
        data_players.loc[data_players['player_name']==player,'exp_avg_'+feature]=exponential_avg_stats_players(feature,player,3)


# We can now add them to the dataset with the game. 

# In[239]:


all_games=pd.merge(all_games,data_players[['exp_avg_'+list_features_exp_avg[k] for k in range (len(list_features_exp_avg))]+['year','player_name']],
                how='left',left_on=['year','winner'], right_on=['year','player_name']).drop('player_name',axis=1)
for col in ['exp_avg_'+list_features_exp_avg[k] for k in range (len(list_features_exp_avg))]:
    all_games=all_games.rename({col:col+'_winner'},axis=1)
    
all_games=pd.merge(all_games,data_players[['exp_avg_'+list_features_exp_avg[k] for k in range (len(list_features_exp_avg))]+['year','player_name']],
                how='left',left_on=['year','loser'], right_on=['year','player_name']).drop('player_name',axis=1)
for col in ['exp_avg_'+list_features_exp_avg[k] for k in range (len(list_features_exp_avg))]:
    all_games=all_games.rename({col:col+'_loser'},axis=1)    


# In[231]:


for col in ['exp_avg_'+list_features_exp_avg[k] for k in range (len(list_features_exp_avg))]:
    all_games=all_games.rename({col:col+'_winner'},axis=1)


# ## Ranking features

# In[262]:


def exponential_avg_ranking_players(player_name,time_period):
    data=ranking_all[ranking_all['player_name']==player_name]
    return (data['rank_number'].ewm(span=time_period).mean())

exponential_avg_ranking_players('piros_z.',10)


# <span class="burk">Warning: We should work only on the loser players instead of the ranking dataset, since the ranking dataset has many much players than the all_games (13000 vs 3500).</span> 

# In[284]:


k=1
for player in all_games['loser'].unique():
    if k%100==0:
        clear_output(wait=True)
        print(k)
    ranking_all.loc[ranking_all['player_name']==player,'exp_avg_ranking']=exponential_avg_ranking_players(player,5)
    k+=1
    


# In[287]:


all_games.columns


# In[289]:


all_games=all_games.merge(ranking_all[['unique_id','exp_avg_ranking']],how='left',left_on='id_winner',right_on='unique_id').rename({'exp_avg_ranking':'exp_avg_ranking_winner'},axis=1).drop('unique_id',axis=1)
all_games=all_games.merge(ranking_all[['unique_id','exp_avg_ranking']],how='left',left_on='id_loser',right_on='unique_id').rename({'exp_avg_ranking':'exp_avg_ranking_loser'},axis=1).drop('unique_id',axis=1)


# # Final Dataset obtaining 

# In[298]:


all_games.columns


# # Other features we could add in the future
# 
# With Noé, we also thought about other features that we could add in the future. We prefered keeping them in mind but not using them, in order to work on a first dataset, and then improve it. 
# Here is a (non exhoaustive) list of them: 
# - Nationality/location of the tournament
# - Nb tourney won by each pleayer
# - slope of the ranking
# - relative difference within a year of ranking.
