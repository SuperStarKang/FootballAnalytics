import pandas as pd
import numpy as np
import os

def calculate_match_ppda(match_events):
	"""
	단일 경기 데이터프레임을 받아 두 팀의 PPDA 계산

	Args:
		match_events (pd.DataFrame): 단일 경기의 이벤트 데이터.

	Returns:
		dict: {team1_id: ppda_value1, team2_id: ppda_value2)
	"""

	# 1. 경기에 참여한 두 팀의 ID 확인
	teams_in_match = match_events['team_id'].unique()
	team1_id, team2_id = teams_in_match[0], teams_in_match[1]

	ppda_results = {}

	# 2. 각 팀의 관점에서 PPDA 계산 (코드를 두 번 반복)
	for my_team_id, opponent_id in [(team1_id, team2_id), (team2_id, team1_id)]:
		
		# 3. 분자 계산 (상대팀의 수비 진영 패스)
		opponent_passes_condition = (
			(match_events['team_id'] == opponent_id) &
			(match_events['event_type'].isin(['Pass', 'Free kick'])) &
			(match_events['start_x'] < 62.4)
		)
		num_opponent_passes = len(match_events[opponent_passes_condition])

		# 4. 분모 계산 (우리팀의 상대 진영 수비 액션)
		my_team_events_in_opp_half = match_events[
			(match_events['team_id'] == my_team_id) & 
			(match_events['start_x'] > 41.6)
		]

		is_interception = my_team_events_in_opp_half['tags'].apply(lambda tags_list: 'interception' in tags_list)
		is_won_duel = my_team_events_in_opp_half['tags'].apply(lambda tags_list: 'won' in tags_list)
		is_sliding_tackle = my_team_events_in_opp_half['tags'].apply(lambda tags_list: 'sliding_tackle' in tags_list)

		defensive_actions = my_team_events_in_opp_half[
			(my_team_events_in_opp_half['sub_event_type'].isin(['Foul', 'Hand foul', 'Late card foul', 'Violent foul'])) |
			(is_interception) |
			((my_team_events_in_opp_half['sub_event_type'] == 'Ground defending duel') & is_won_duel) |
			(is_sliding_tackle)
		]
		num_defensive_actions = len(defensive_actions)

		# 5. PPDA 최종 계산
		if num_defensive_actions == 0:
			ppda = np.nan
		else:
			ppda = num_opponent_passes / num_defensive_actions
		
		# 결과 저장
		ppda_results[my_team_id] = ppda
		
	return ppda_results