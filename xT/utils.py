import socceraction.spadl as spadl
import pandas as pd
import matplotsoccer as mps # 축구 이벤트 데이터 시각화

# 모든 액션이 왼쪽에서 오른쪽으로 진행되도록 좌표 변환(공격 방향 통일)
def play_left_to_right(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
	"""
		모든 액션이 동일한 경기 방향(왼쪽에서 오른쪽)으로 수행되도록 변환합니다.

		이 함수는 각 액션의 시작 및 종료 위치를 변경하여,
		마치 첫 액션을 수행한 팀이 항상 왼쪽에서 오른쪽으로 플레이하는 것처럼 만듭니다.

		Parameters
		----------
		actions : pd.DataFrame
			한 경기의 액션 데이터프레임.
		home_team_id : int
			홈팀의 ID.

		Returns
		-------
		pd.DataFrame
			모든 액션이 왼쪽에서 오른쪽으로 수행되도록 좌표가 변환된 데이터프레임.
	"""

	away_idx = actions.team_id != home_team_id

	for col in ["start_x", "end_x"]:
		actions.loc[away_idx, col] = spadl.config.field_length - actions[away_idx][col].values
	for col in ["start_y", "end_y"]:
		actions.loc[away_idx, col] = spadl.config.field_width - actions[away_idx][col].values

	return actions

def nice_time(row):
	minute = int((row.period_id-1)*45 +row.time_seconds // 60)
	second = int(row.time_seconds % 60)
	return f"{minute}m{second}s"

def plot_actions(a: pd.DataFrame, g: pd.Series, df_players: pd.DataFrame, df_teams: pd.DataFrame) -> None:
	home_team_name = df_teams[df_teams.team_id == g.home_team_id].team_name.values[0]
	away_team_name = df_teams[df_teams.team_id == g.away_team_id].team_name.values[0]

	minute = int((a.period_id.values[0]-1) * 45 + a.time_seconds.values[0] // 60)
	game_info = f"{g.game_date} {home_team_name} {g.home_score}-{g.away_score} {away_team_name} {minute + 1}'"
	print(game_info)

	a["player_name"] = a.player_id.map(df_players.set_index("player_id").player_name)
	a["team_name"] = a.team_id.map(df_teams.set_index("team_id").team_name)
	a["type_name"] = a.type_id.map(spadl.config.actiontypes_df().type_name.to_dict())
	a["result_name"] = a.result_id.map(spadl.config.results_df().result_name.to_dict())
	a["nice_time"] = a.apply(nice_time, axis=1)

	if "xT_value" in a.columns:
		labels = a[["nice_time", "type_name", "player_name", "team_name", "xT_value"]]
		labeltitle = ["time", "actiontype", "player", "team", "xT_value"]
	else:
		labels = a[["nice_time", "type_name", "player_name", "team_name"]]
		labeltitle = ["time", "actiontype", "player", "team"]

	away_idx = a.team_id != g.home_team_id
	for col in ["start_x", "end_x"]:
		a.loc[away_idx, col] = spadl.config.field_length - a[away_idx][col].values
	for col in ["start_y", "end_y"]:
		a.loc[away_idx, col] = spadl.config.field_width - a[away_idx][col].values

	mps.actions(
		location=a[["start_x", "start_y", "end_x", "end_y"]],
		action_type=a.type_name,
		team= a.team_name,
		result= a.result_name == "success",
		label=labels,
		labeltitle=labeltitle,
		zoom=False,
		figsize=6,
		color="green"
	)

# xT 모델 학습 과정(Value Iteration)의 각 단계별 xT 값 변화를 3D Surface 플롯으로 시각화하는 함수
def visualize_surface_plots(xTModel):
	"""Visualizes the surface plot of each iteration of the model.

	See https://plot.ly/python/sliders/ and https://karun.in/blog/expected-threat.html#visualizing-xt
	NOTE: y-axis is mirrored in plotly.
	"""
	camera = dict(
		up=dict(x=0, y=0, z=1),
		center=dict(x=0, y=0, z=0),
		eye=dict(x=-2.25, y=-1, z=0.5),
	)

	max_z = np.around(xTModel.xT.max() + 0.05, decimals=1)

	layout = go.Layout(
		title="Expected Threat",
		autosize=True,
		width=500,
		height=500,
		margin=dict(l=65, r=50, b=65, t=90),
		scene=dict(
			camera=camera,
			aspectmode="auto",
			xaxis=dict(),
			yaxis=dict(),
			zaxis=dict(autorange=False, range=[0, max_z]),
		),
	)

	fig = go.Figure(layout=layout)

	for i in xTModel.heatmaps:
		fig.add_trace(go.Surface(z=i))

	# Make last trace visible
	for i in range(len(fig.data) - 1):
		fig.data[i].visible = False
	fig.data[len(fig.data) - 1].visible = True

	# Create and add slider
	steps = []
	for i in range(len(fig.data)):
		step = dict(method="restyle", args=["visible", [False] * len(fig.data)])
		step["args"][1][i] = True  # Toggle i'th trace to "visible"
		steps.append(step)

	sliders = [
		dict(
			active=(len(fig.data) - 1),
			currentvalue={"prefix": "Iteration: "},
			pad={"t": 50},
			steps=steps,
		)
	]

	fig.update_layout(sliders=sliders)
	fig.show()