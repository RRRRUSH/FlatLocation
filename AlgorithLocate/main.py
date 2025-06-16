import time

import numpy as np
import math

import matplotlib.pyplot as plt

from tools.csv import read_csv, reload, read_real_loc
from tools.date import get_date, date_formt, sec2date
from tools.plot import (point, clear, set_axes, deg2rad,
						show_beacon, show_circle_line, fig_init, show_params)

from loc.points import get_points
from loc.scores import get_score

from configs import ALIVE_CIRCLE, ALLTIME_DATA
from configs import DBID, AXLIM, FIGSIZE

# 修改信标位置
from configs import BEACON_LOCATION_1011 as BEACON_LOCATION


tag = (0, 0)
history= None

def get_dis(loc, tag):
	return math.sqrt((loc[0]-tag[0])**2+(loc[1]-tag[1])**2)

# cal
def cal(param, poplis):
	# update ALIVE_CIRCLE
	global history
	for i in poplis:
		ALIVE_CIRCLE.pop(i)

	points = get_points(param)

	if len(points) != 0:
		scores = get_score(points, history)

		max3 = np.argsort(scores)[-3:]
		for index, point_info in enumerate(points):
			loc = point_info[0], point_info[1]

			if index == max3[-1]:
				history = point_info
				color = 'red'; text_loc = (0.1, 0.25); rank = '0'
			elif index == max3[0]:
				color = 'orange'; text_loc = (0.1, 0.2); rank = '1'
			elif index == max3[1]:
				color = 'orange'; text_loc = (0.1, 0.15); rank = '2'
			else:
				color = 'gray'
			
			if color != 'gray':
				ax_param_info.text(
					text_loc[0], text_loc[1], f'rank{rank} ({round(loc[0], 2)}, {round(loc[1], 2)})',
					fontsize=12, ha='left', va='center')
			point(ax_circle, loc, color=color, text=np.round(scores[index], 2))


def update(val):
	clear([ax_circle, ax_param_info])

	# 根据 val 将当前的所有数据加入 VAL_TIMELINE
	# val 值小于 len(VAL_TIMELINE) 时，利用 VAL 中的circle进行计算
	# cal 大于 len 时，继续递增
	# ALLTIME_DATA[val] = {}

	set_axes(ax_circle, AXLIM)

	sim_time = str(beg + sec2date(val))[:-3]
	text.set_text(f'Time: {sim_time}')

	# 步进较小，需要找出最近的一条记录
	closest_time = min(
		TIMEKEY_DATA.keys(),
		key=lambda x: abs(date_formt(x) - date_formt(sim_time, "%Y-%m-%d %H:%M:%S.%f"))
	)

	# closest_time =  date_formt(sim_time, "%Y-%m-%d %H:%M:%S.%f")

	record = TIMEKEY_DATA.get(closest_time)
	beg_time = time.time()

	if record is not None and record['TYPE'] != 'motionState':

		db_lis = record['AroundList']
		beacon_id = record['JZID'].replace(' ', '')

		for db in db_lis:
			if db['DBID'] == DBID:
				end_time = date_formt(closest_time) + sec2date(60) - sec2date(int(db['toLastTime']))

				angle = int(db.get('angle', -1))
				if angle == -1:
					# id radius true_angle measure_angle
					args = {
						'key': beacon_id,
						'dis': int(db['dis']),
						'time': int(db['toLastTime']),
						'measure_angle': -1,
						'true_angle': -1,
					}

				else:
					if angle < 60 or angle > 300:
						# cal angle
						measure_angle = min(360 - angle, angle)
						true_angle = (angle + BEACON_LOCATION[beacon_id]['aspect'] - 90) % 360

						args = {
							'key': beacon_id+"_angle",
							'dis': -1,
							'time': int(db['toLastTime']),
							'measure_angle': deg2rad(measure_angle),
							'true_angle': deg2rad(true_angle),
						}
					else:
						if beacon_id+"_angle" in ALIVE_CIRCLE.keys():
							ALIVE_CIRCLE.pop(beacon_id+"_angle")
						continue

				assert args is not None, "Error args is None"
				ALIVE_CIRCLE[args['key']] = {
					'center': BEACON_LOCATION[beacon_id]['loc'], 'radius': args['dis'],
					'true_angle': args['true_angle'], 'time': args['time'], 'measure_angle': args['measure_angle'],
					'end_time': end_time, 'color': BEACON_LOCATION[beacon_id]['color'],
				}

	pop_lis, param_dic = show_circle_line(ax_circle, ALIVE_CIRCLE, BEACON_LOCATION, sim_time)

	try:
		global time_index
		if date_formt(sim_time, "%Y-%m-%d %H:%M:%S.%f") > real_loc[time_index]['time']:
			time_index += 1
		point(ax_circle, real_loc[time_index]['loc'], color='green', text='')
	except IndexError:
		timer.stop()

	# cal
	cal(list(param_dic.values()), pop_lis)
	print(f"{time.time() - beg_time:.6f} seconds")

	# show result
	show_params(ax_param_info ,param_dic)

	fig.canvas.draw_idle()


def timer_update():
	slider.set_val(slider.val+0.1)

	if slider.val > timeline:
		timer.stop()


paused = False
def pause(event):
	global paused
	paused = not paused

	if paused:
		timer.stop()
	else:
		timer.start()


def main(time):
	global fig, ax_beacon, ax_circle, slider, text, ax_param_info
	fig, ax_beacon, ax_circle, slider, text, btn, ax_param_info = fig_init(
		figsize=FIGSIZE, axlim=AXLIM, timeline=time
	)

	# binding
	slider.on_changed(update)
	btn.on_clicked(pause)

	# show beacon
	show_beacon(ax_beacon, BEACON_LOCATION)

	# Timer
	global timer
	timer = fig.canvas.new_timer(interval=1)
	timer.add_callback(timer_update)
	timer.start()

	plt.show()


def run():
	data = read_csv(r"./data/20241011/json_1400_lhz.csv")
	global time_index, real_loc
	time_index = 0
	real_loc = read_real_loc(r"./data/20241011/1400_lhz.csv", '2024-10-11 14:00:00')

	# pre data
	global beg, timeline, TIMEKEY_DATA
	beg, end, timeline = get_date(data)

	TIMEKEY_DATA = reload(data)

	main(timeline)


if __name__ == "__main__":
	run()
