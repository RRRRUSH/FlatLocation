import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

from tools.date import date_formt


def set_axes(ax, axlim):
	# set x axis and y axis
	ax.set_xlim(axlim); ax.set_ylim(axlim)
	ax.set_aspect('auto'); ax.invert_yaxis()

	# show grid
	ax.grid(True)


def circle(ax, center, radius, text, color):
	# plt.Circle 不会自动将对象添加到 axes 中
	circle = plt.Circle(center, radius, linewidth=1, fill=False, color=color)
	ax.add_artist(circle)
	ax.text(center[0], center[1]+radius, round(text, 1), fontsize=8, color=color, ha='center', va='top')


def point(ax, loc, color, text):
	# plt.scatter 自动将对象添加到当前活动的 axes 中，若调用 add_artist 则会出现异常
	ax.scatter(loc[0], loc[1], s=8, color=color)
	ax.text(loc[0], loc[1], text, fontsize=8, color=color, ha='center', va='bottom')


def deg2rad(angle):
	return angle * np.pi / 180


def rad2deg(angle):
	return angle * 180 / np.pi


def line(ax, beacon_loc, length, angle, color):
	ax.plot(
		[beacon_loc[0], beacon_loc[0] + length * np.cos(angle)],
		[beacon_loc[1], beacon_loc[1] + length * np.sin(angle)],
		linewidth=1, linestyle='--', color=color
	)
	# ax.text(beacon_loc[0], beacon_loc[1], rad2deg(angle), fontsize=8, color=color, ha='center', va='bottom')


def clear(ax_lis):
	for ax in ax_lis:
		ax.clear()
		ax.axis('off')


def show_beacon(ax, loc):
	for beacon_id in loc.keys():
		point(ax, loc[beacon_id]['loc'], color=loc[beacon_id]['color'], text=beacon_id)


def show_circle_line(ax, alive_circle, beacon_loc, sim_time):
	pop_lis, param_dic = list(), dict()
	for beacon_id in alive_circle.keys():
		alive_circle[beacon_id]['time'] += 0.1
		if alive_circle[beacon_id]['end_time'] >= date_formt(sim_time, "%Y-%m-%d %H:%M:%S.%f"):
			if beacon_id.endswith('_angle'):
				line(
					ax, beacon_loc[beacon_id[:-6]]['loc'], 50,
					alive_circle[beacon_id]['true_angle'], beacon_loc[beacon_id[:-6]]['color']
				)
				param_dic[beacon_id] = [
					alive_circle[beacon_id]['center'][0], alive_circle[beacon_id]['center'][1], -1,
					alive_circle[beacon_id]['true_angle'], round(alive_circle[beacon_id]['time'], 1),
					alive_circle[beacon_id]['measure_angle']]

			else:
				circle(
					ax, center=alive_circle[beacon_id]['center'], radius=alive_circle[beacon_id]['radius'],
					text=alive_circle[beacon_id]['time'], color=alive_circle[beacon_id]['color']
				)
				param_dic[beacon_id] =  [
					alive_circle[beacon_id]['center'][0], alive_circle[beacon_id]['center'][1], alive_circle[beacon_id]['radius'],
					-1, round(alive_circle[beacon_id]['time'], 1), -1]
		else:
			pop_lis.append(beacon_id)
	return pop_lis, param_dic


def show_params(ax, param_dic):
	param_text_show = ''
	for pkey in param_dic.keys():
		param_text_show += (f'{pkey}: {param_dic[pkey][0], param_dic[pkey][1]}, {param_dic[pkey][2]}, ' +
							f'{round(param_dic[pkey][3],2)}, {param_dic[pkey][4]}, {round(param_dic[pkey][5],2)}\n')
	ax.text(0.1, 0.5, param_text_show, fontsize=12, ha='left', va='center')


def hide_spine(ax_ls):
	for ax in ax_ls:
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.set(xticks=[], yticks=[])


def fig_init(figsize=(10, 10), axlim=(-40, 60), timeline=0):
	fig, axes = plt.subplots(figsize=figsize)
	axes.axis('off')

	grid = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[8, 1])

	# left subplot is beacon axes
	ax_beacon = fig.add_subplot(grid[:, 0])
	set_axes(ax_beacon, axlim)

	# param axes
	ax_param_info = fig.add_subplot(grid[0, 1])

	# Button widget
	ax_btn = fig.add_subplot(grid[1, 1])
	btn = Button(ax_btn, 'Pause/Continue')

	# hide_spine([ax_param_info])

	# circle axes
	ax_circle = fig.add_axes(ax_beacon.get_position(), frameon=False)

	# left bottom width height (0,1)
	slider_ax = plt.axes((0.15, 0.05, 0.5, 0.02))
	slider = Slider(slider_ax, 'Time line', 0, timeline, valinit=0, valstep=0.1)

	# slider text
	ax_text = plt.axes((0.55, 0.05, 0.5, 0.02))
	slider_text = ax_text.text(0.55, 0.5, '', transform=ax_text.transAxes, ha='center', va='center')
	ax_text.axis('off')

	return fig, ax_beacon, ax_circle, slider, slider_text, btn, ax_param_info
