import bisect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.animation import FuncAnimation

class Plotter:
    def __init__(self, figsize=(12, 6),
                 angle_result: dict=None, score_result: dict=None, colors=None,
                 antenna_nums=6):

        assert angle_result.keys() == score_result.keys()
        self.angle_result = angle_result
        self.score_result = score_result

        self.timestamps = list(self.angle_result.keys()) if self.angle_result else []
        self.antenna_nums = antenna_nums

        self.fig = plt.figure(figsize=figsize)

        self.layout = [
            (0.05, 0.1, 0.4, 0.8),
            (0.55, 0.1, 0.4, 0.8)
        ]
        self.ax_lim = [
            (-3, 5), # xlim
            (-5, 5)  # ylim
        ]

        self.is_playing = False
        self.current_index = 0
        self.colors = colors if colors else [
            "#FF355E", "#190033", "#0066FF", "#CC6600", "#FF00CC", "#00FEFE"]

        self.ax_left = self.fig.add_axes(self.layout[0])
        self.ax_left.set_xlim(*self.ax_lim[0])
        self.ax_left.set_ylim(*self.ax_lim[1])
        self.ax_left.set_aspect('equal')
        self.ax_left.grid(True)
        self.ax_left.set_xlabel('X')
        self.ax_left.set_ylabel('Y')

        self.ax_right = self.fig.add_axes(self.layout[1])
        self.ax_right.axis('off')

        self.fig.canvas.mpl_connect('close_event', self.on_close)

        # 添加控件
        self._add_widgets()
        self._draw_static()

        self.update_plot(self.current_index)

        self.ani = FuncAnimation(
            self.fig,
            self.animate,
            interval=300,
            blit=False,
            save_count=len(self.timestamps)
        )

    def add_angle(self, angle, length=1.0, color='blue', label=None, score=0):
        """绘制测量角度线"""
        rad = np.deg2rad(angle)
        x = length * np.cos(rad)
        y = length * np.sin(rad)
        self.ax_left.plot([0, x], [0, y], color=color, label=label)
        self.ax_left.text(x, y, s=f"{round(score, 3)}", color=color)
        if label:
            self.ax_left.legend(loc="upper right")

    def add_item(self, index: int):

        length = 3
        ts_now = self.timestamps[index]
        angles_table = ["ID,SRC,RSSI,PDOA,AOA,STD,TIME,MARK,SYS_AOA".split(',')]
        scores_table = ["ID,SCOPE,TIME,STD,DENSITY,HISTORY,RSSI,RESULT".split(',')]

        for i, angle_meas in enumerate(self.angle_result[ts_now]):
            ant_id, rssi, aoa = angle_meas[0], angle_meas[1], angle_meas[-1]

            if i != 0:
                length = 3 * self.score_result[ts_now][i][-1]

            self.add_angle(aoa, color=self.colors[int(ant_id)], length=length,
                              label=f"{i} {int(ant_id)} {round(aoa, 3)}", score=i)
            angles, scores = (np.round(self.angle_result[ts_now][i], 2).tolist(),
                              np.round(self.score_result[ts_now][i], 3).tolist())
            angles.insert(0, i)
            scores.insert(0, i)
            angles_table.append(angles)
            scores_table.append(scores)
        self._create_tables(
            angles_table, bbox=[-0.32, 0.2, 0.7, 0.7],
            colWidths=[0.003, 0.007, 0.007, 0.01, 0.01, 0.007, 0.01, 0.007, 0.01])
        self._create_tables(
            scores_table, bbox=[0.4, 0.2, 0.7, 0.7],
            colWidths=[0.002, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007]
        )

    def update_plot(self, index):
        self.clear()
        # 将时间戳信息显示在图表的左上方，移出图表内部
        self.ax_left.text(0, 1.05, f"[{index}] <<CURRENT TIMESTAMP>>: {self.timestamps[index]}",
                     transform=self.ax_left.transAxes, fontsize=10, verticalalignment='top')
        self.add_item(index)
        self.update()

    def _add_widgets(self):
        """添加按钮控件"""
        # 新增时间戳跳转控件
        self.ax_textbox = plt.axes((0.8, 0.9, 0.15, 0.075))
        self.textbox = TextBox(self.ax_textbox, 'Go to ts:')

        self.btn_play = Button(plt.axes((0.65, 0.9, 0.07, 0.075)), 'Play')
        self.btn_prev = Button(plt.axes((0.45, 0.9, 0.07, 0.075)), 'Previous')
        self.btn_next = Button(plt.axes((0.55, 0.9, 0.07, 0.075)), 'Next')

        self.btn_play.on_clicked(self.on_play)
        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)

        self.textbox.on_submit(self.on_goto)

    def _create_tables(self, data, bbox, colWidths):
        ax_scores_table = self.ax_right.table(
            data, cellLoc='center', edges='closed', bbox=bbox, colWidths=colWidths
        )
        ax_scores_table.auto_set_font_size(False)
        ax_scores_table.set_fontsize(9)

    def _draw_static(self, radius=1.0, color='gray'):
        """绘制基站朝向"""
        angles = np.linspace(np.pi/self.antenna_nums, 2 * np.pi + np.pi/self.antenna_nums, self.antenna_nums+1)[:-1]
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        self.ax_left.fill(x, y, color=color, alpha=0.3)

    def on_close(self, event):
        """窗口关闭事件处理"""
        print("Window closed, exiting program.")
        plt.close('all')
        exit(0)

    def on_play(self, event):
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Stop' if self.is_playing else 'Play')
        if self.is_playing:
            self.ani.event_source.start()
        else:
            self.ani.event_source.stop()
        plt.draw()

    def on_prev(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot(self.current_index)

    def on_next(self, event):
        if self.current_index < len(self.timestamps) - 1:
            self.current_index += 1
            self.update_plot(self.current_index)

    def on_goto(self, event=None):
        try:
            input_ts = int(self.textbox.text.strip())
        except ValueError:
            print("Invalid timestamp input.")
            return

        for i in range(len(self.timestamps)):
            if self.timestamps[i] == input_ts:
                self.current_index = i

                self.update_plot(self.current_index)
                print(f"Jumped to timestamp: {self.timestamps[self.current_index]} (index: {self.current_index})")
                break
            else:
                print("Invalid timestamp")

    def animate(self, frame):

        if self.is_playing and self.current_index < len(self.timestamps) - 1:
            self.current_index += 1
            self.update_plot(self.current_index)

    def clear(self):
        self.ax_left.clear()
        self.ax_right.clear()

        self.ax_left.set_xlim(*self.ax_lim[0])
        self.ax_left.set_ylim(*self.ax_lim[1])
        self.ax_left.set_aspect('equal')
        self.ax_left.grid(True)
        self.ax_left.set_xlabel('X')
        self.ax_left.set_ylabel('Y')

        self.ax_right.axis('off')

        self._draw_static(radius=0.3)

    def update(self):
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__=="__main__":
    angle_result = {
        1.0: np.array([
            [0, 20, 3, 4, 5, 60, 1, 120],
            [1, 30, 4, 5, 6, 120, -1, 300]
        ]),
        2.0: np.array([
            [0, 25, 3, 4, 5, 60, 1, 110],
            [1, 35, 4, 5, 6, 120, -1, 310]
        ])
    }
    score_result = {
        1.0: np.array([
            ["0, 20, 3, 4, 5, 60, 0.8"],
            ["1, 30, 4, 5, 6, 120, 0.9"]
        ]),
        2.0: np.array([
            ["0, 25, 3, 4, 5, 60, 0.7"],
            ["1, 35, 4, 5, 6, 120, 0.8"]
        ])
    }

    plotter = Plotter(angle_result=angle_result, score_result=score_result)
    plotter.show()