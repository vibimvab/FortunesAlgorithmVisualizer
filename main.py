import tkinter as tk
import numpy as np
from fortunes import FortunesAlgorithm

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
SAMPLE_COUNT = 600


class VoronoiVisualizer:
    def __init__(self, master):
        self.master = master
        self.mode = "edit"       # "edit" / "view" / "completed"
        self.sites = []
        self.sites_added = False
        self.sweep_y = 100

        # 버튼 UI
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(side='top', fill='x')
        self.edit_btn = tk.Button(self.button_frame, text="점 입력 모드", command=self.set_edit_mode)
        self.edit_btn.pack(side='left')
        self.view_btn = tk.Button(self.button_frame, text="점 추가", command=self.generate_points)
        self.view_btn.pack(side='left')
        self.view_btn = tk.Button(self.button_frame, text="초기화", command=self.reset)
        self.view_btn.pack(side='left')

        self.view_btn = tk.Button(self.button_frame, text="시각화 모드", command=self.set_view_mode)
        self.view_btn.pack(side='left')
        self.view_btn = tk.Button(self.button_frame, text="완성본 보기", command=self.show_completed)
        self.view_btn.pack(side='left')

        # 캔버스
        self.canvas = tk.Canvas(master, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg='white')
        self.canvas.pack()

        # 이벤트 바인딩
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # fortune's algorithm
        self.fa = None

        self.draw()

    def set_edit_mode(self):
        self.mode = "edit"
        self.sweep_y = 100
        self.draw()

    def generate_points(self):
        # if self.mode == "edit":
        for _ in range(5):
            self.sites.append((np.random.randint(WINDOW_WIDTH / 10, WINDOW_WIDTH * 9 / 10),
                               np.random.randint(WINDOW_HEIGHT / 10, WINDOW_HEIGHT * 8 / 10)))
        self.draw()
        self.sites_added = True

    def reset(self):
        if not self.mode == "edit":
            return

        self.sites = []
        self.fa = None
        self.draw()

    def set_view_mode(self):
        if not self.sites:
            return
        self.mode = "view"
        self.sweep_y = 50

        # 여기서 Fortune’s 알고리즘을 실행하고 snapshot 저장 가능 (다음 단계)
        if self.sites_added:
            self.fa = FortunesAlgorithm(self.sites)
            self.fa.run()
        self.draw()
        self.sites_added = False

    def show_completed(self):
        if not self.sites:
            return

        self.mode = "completed"

        if self.sites_added:
            self.fa = FortunesAlgorithm(self.sites)
            self.fa.run()
        self.draw()


    def on_click(self, event):
        if self.mode == "edit":
            self.sites.append((event.x, event.y))
            self.draw()
            self.sites_added = True

    def on_mouse_move(self, event):
        if self.mode == "view":
            self.sweep_y = event.y
            self.draw()

    def draw(self):
        self.canvas.delete("all")

        # sites
        for x, y in self.sites:
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")

        if self.mode == 'completed':
            # rays
            self.fa.draw_rays(canvas=self.canvas, sweep_y=float('inf'))

        elif self.mode == "view":
            # rays
            self.fa.draw_rays(canvas=self.canvas, sweep_y=self.sweep_y)

            # sweep line
            self.canvas.create_line(0, self.sweep_y, WINDOW_WIDTH, self.sweep_y, fill="red", dash=(4, 2))

            # beach line
            self.fa.draw_beachline(canvas=self.canvas, sweep_y=self.sweep_y)


    def compute_parabola_y(self, x, fx, fy, ly):
        if fy == ly:
            return np.full_like(x, np.inf)
        return ((x - fx)**2) / (2 * (fy - ly)) + (fy + ly) / 2

    def draw_parabola(self, x_vals, y_vals):
        points = []
        for x, y in zip(x_vals, y_vals):
            if 0 <= y <= WINDOW_HEIGHT:
                points.append((x, y))

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            self.canvas.create_line(x1, y1, x2, y2, fill="blue", smooth=True)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Voronoi Diagram Visualizer")
    app = VoronoiVisualizer(root)
    root.mainloop()
