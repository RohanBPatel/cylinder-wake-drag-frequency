import re
import os
import subprocess
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from scipy.fft import fft, fftfreq
from scipy.signal import welch, find_peaks
from scipy.optimize import curve_fit
from scipy.stats import t, norm
from scipy.spatial import cKDTree
from pydmd import DMD
from pydmd.plotter import plot_eigs, plot_summary, plot_snapshots_2D
import matplotlib.patches as mpatches

prog_path = Path(os.path.abspath(""))
fig_path = prog_path / "Figures"
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

plt.rcParams["figure.dpi"] = 125
plt.rcParams["image.cmap"] = "gray"

def get_num(name) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1

def show_image(arr):
    plt.imshow(arr, cmap="gray", vmin=np.min(arr), vmax=np.max(arr))

def mouse_event(event):
   print(f"{int(event.xdata)}\t{int(event.ydata)}")

def add_filtered_handles(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

rho = 1.001e3 # kg/m^3
mu = 5.84e-3 # Pa.s

class Test:
    calibration_df = pd.DataFrame.from_dict({
        1: {"im_num": 0, "points": np.array([[15,  858], [1285, 899]]), "length_cm": 6},
        2: {"im_num": 0, "points": np.array([[144, 673], [1422, 696]]), "length_cm": 6},
        3: {"im_num": 0, "points": np.array([[123, 622], [1388, 666]]), "length_cm": 6}
    }, orient="index")

    calibration_df["length_mm"] = calibration_df["length_cm"] * 10
    calibration_df["length_px"] = calibration_df["points"].apply(lambda p: np.linalg.norm(p[1] - p[0]))
    calibration_df["alpha"] = calibration_df["length_mm"] / calibration_df["length_px"]
    calibration_scales = calibration_df["alpha"].to_dict()

    all_measurements_df = pd.concat(
        pd.read_excel("Visual_Measurements.xlsx", sheet_name=["Rohan", "Logan", "Blake"]),
        names=["Sheet"]  
    ).reset_index() # level=0

    all_measurements_df = all_measurements_df.dropna(axis="index", thresh=4)

    all_measurements_df["Path"] = all_measurements_df["Path"].apply(lambda p: Path(str(p).strip()))

    ps = all_measurements_df[["Line Segment Start X", "Line Segment Start Y", "Line Segment End X", "Line Segment End Y"]].values
    all_measurements_df["l_px"] = np.linalg.norm(ps[:, 2:4] - ps[:, 0:2], axis=1)

    all_measurements_df["Mid X"] = all_measurements_df[["Line Segment Start X", "Line Segment End X"]].sum(axis=1) // 2
    all_measurements_df["Mid Y"] = all_measurements_df[["Line Segment Start Y", "Line Segment End Y"]].sum(axis=1) // 2

    # all_measurements_df["Day"] = all_measurements_df["Path"].apply(
    #     lambda path: get_num(path.parts[0])
    # )
    # all_measurements_df["alpha"] = all_measurements_df["Day"].map(calibration_scales)

    all_measurements_df["Folder"] = all_measurements_df["Path"].apply(lambda path: path.parent)
    measured_folders = list(map(str, all_measurements_df["Folder"].unique()))

    cylinder_override = {
        r"T8_Open_Lab_3\5_100":  np.array([390.0, 361.5, 55.0]),
        r"T8_Open_Lab_3\5_150":  np.array([390.0, 365.0, 70.0]),
        r"T8_Open_Lab_2\5_220":  np.array([324.0, 385.0, 67.0]), # refine
        r"T8_Open_Lab_2\10_220": np.array([350.0, 410.0, 116.0]) # refine
    }

    grit_to_roughness = {100: 32.42, 150: 16.89, 220: 14.26, 400: 10.0} # micrometer, 400 is a guess

    def __init__(self, folder_name: str, frame_rate=200, t_exposure=5000):
        """
        frame_rate (Hz)
        t_exposure (microsecond)
        """
        self.folder_name = folder_name
        self.folder = prog_path / folder_name
        self.measurements_df = Test.all_measurements_df[Test.all_measurements_df["Folder"] == self.folder.relative_to(prog_path)].copy()
        self.test_day = get_num(self.folder.parent.name)
        self.frame_rate = frame_rate
        self.t_exposure_mus = t_exposure # micros second
        self.t_exposire_ms = self.t_exposure_mus / 1e3 # millisecond

        self.file_prefix  = self.folder_name.replace("\\", "_")

        # self.out_fig_path = fig_path / self.folder_name
        # self.out_fig_path.mkdir(exist_ok=True)

        subfolder_name = self.folder.name

        if subfolder_name == "10_100_2":
            subfolder_name = "10_100"
        elif subfolder_name == "15_220_exposure_10000":
            subfolder_name = "15_220"

        self.alpha = Test.calibration_scales[self.test_day]
        self.nominal_diameter_mm, self.grit = map(int, str(subfolder_name).split("_"))
        self.nominal_diameter_px = self.nominal_diameter_mm / self.alpha
        self.roughness = Test.grit_to_roughness.get(self.grit)
        self.nominal_rel_roughness = (self.roughness / 1e3) / self.nominal_diameter_mm

        files = list(self.folder.iterdir())
        self.df = pd.DataFrame(files, columns=["File"])
        self.df = self.df[self.df["File"].astype(str).str.lower().str.endswith(".tiff")]
        self.df["Number"] = self.df["File"].apply(lambda f: get_num(f.name))
        self.df.sort_values("Number", ignore_index=True, inplace=True)

        self.df["Image_BGR"] = self.df["File"].apply(
            lambda file: cv.imread(file, cv.IMREAD_COLOR_RGB)
        )

        self.df["Image_Gray"] = self.df["File"].apply(
            lambda file: cv.imread(file, cv.IMREAD_GRAYSCALE)
        )

        # self.background_model_gray = np.median(np.stack(self.df["Image_Gray"].values), axis=0).astype(np.uint8)
        self.background_model_gray = np.mean(np.stack(self.df["Image_Gray"].values), axis=0).astype(np.uint8)
        self.background_model_color = cv.cvtColor(self.background_model_gray, cv.COLOR_GRAY2RGB)

        self.find_cylinder()
        self.rel_roughness = (self.roughness / 1e3) / self.D_mm
        self.find_probe_mask()
        self.find_wake_rect_bounds()
        self.find_pathlines()
        self.process_measurements()
        self.dmd_analysis()

        if self.folder_name in Test.measured_folders:
            self.fit_wake()
            self.fit_wake_2()

    @staticmethod
    def show_calibration_image(day: int) -> None:
        row = Test.calibration_df.loc[day]
        calibration_path = prog_path / f"T8_Open_Lab_{day}" / "calibration"
        p = row["points"]
        for path in calibration_path.iterdir():
            if get_num(path.name) == row["im_num"]:
                im = cv.imread(path)
                show_image(im)
                plt.plot(p[:, 0], p[:, 1], lw=2)
                plt.show()
                return
        return
    
    @staticmethod
    def show_scaled_and_centered_image(image, scale=1.0, c_x=None, c_y=None, ax=None, figsize=(8, 6)):
        img_h, img_w = image.shape
        if c_x is None:
            c_x = img_w / 2
        if c_y is None:
            c_y = img_h / 2

        left   = (0     - c_x) * scale
        right  = (img_w - c_x) * scale
        top    = (0     - c_y) * scale
        bottom = (img_h - c_y) * scale

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.imshow(
            image,
            extent=[left, right, bottom, top],
            origin="upper",
            interpolation="bilinear",
            aspect="equal"
        )

        return fig, ax

    def find_cylinder(self, save=False, show=False) -> None:
        diam_limits = np.array([int(self.nominal_diameter_px*0.9), int(self.nominal_diameter_px*1.1)])

        circles = cv.HoughCircles(
            image=self.background_model_gray, 
            method=cv.HOUGH_GRADIENT, 
            dp=1,
            minDist=300,
            param1=300, # 300
            param2=10, # 10-20, smaller -> more circles
            minRadius=diam_limits[0] // 2,
            maxRadius=diam_limits[1] // 2
        )
        
        self.cylinder = Test.cylinder_override.get(self.folder_name)

        if self.cylinder is None:
            if show:
                print(f"Found {circles.shape[1]} circles")
                print(f"Diameter Search Limits (mm): {diam_limits*self.alpha}")
                print(f"Nominal Diameter (mm): {self.nominal_diameter_mm:.4f}")
                print(f"Found Diameter (mm):   {circles[0,0,2]*2*self.alpha:.4f}")

            if circles.shape[1] == 1:
                self.cylinder = circles[0,0].astype(np.float64)

        self.D_px = self.cylinder[2] * 2
        self.D_mm = self.alpha * self.D_px
        self.D_m  = self.D_mm / 1e3

        if not save and not show:
            return
        
        background_circles = self.background_model_color.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv.circle(background_circles, (i[0], i[1]), i[2], (255, 100, 100), 3)
        cv.circle(background_circles, (int(self.cylinder[0]), int(self.cylinder[1])), int(self.cylinder[2]), (50, 255, 50), 3)

        plt.imshow(background_circles)
        plt.title("Finding Cylinder")
        plt.tight_layout()
        if save:
            plt.savefig(fig_path / f"{self.file_prefix}_cylinder_circle.png")
        if show:
            plt.show()
        else:
            plt.close()

    def find_probe_mask(self, save=False, show=False) -> None:
        blurred = cv.GaussianBlur(self.background_model_gray, (5, 5), 0)
        ret, mask = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY_INV)
        # ret, mask = cv.threshold(self.background_model_gray, 100, 255, cv.THRESH_BINARY_INV)

        kernel = np.ones((10, 10), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel) # remove small specks
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel) # fill holes

        # erode to expand the black area around the probe
        mask = cv.erode(mask, np.ones((10,10), np.uint8), iterations=2)

        # mask = cv.convexHull(mask, clockwise=False, returnPoints=True)

        self.probe_mask = mask

        mask_view = cv.bitwise_and(self.background_model_gray, self.background_model_gray, mask=self.probe_mask)

        if not save and not show:
            return

        plt.figure()
        plt.imshow(mask_view, cmap="gray")
        plt.title(f"Probe mask is black. (threshold = {ret})")
        plt.tight_layout()
        if save:
            plt.savefig(fig_path / f"{self.file_prefix}_probe_mask.png")
        if show:
            plt.show()
        else:
            plt.close()

    def find_wake_rect_bounds(self) -> None:
        c_x, c_y, r = self.cylinder.astype(int)

        wake_upper_bound = c_y - (r * 5 // 4) # r
        wake_lower_bound = c_y + (r * 5 // 4) # r

        wake_left_bound_from_cylinder = c_x + r

        probe_mask_x = self.probe_mask[wake_upper_bound:wake_lower_bound].min(axis=0)
        wake_left_bound_from_probe = np.nonzero(255-probe_mask_x[::-1])[0][0]

        wake_left_bound = max(wake_left_bound_from_cylinder, wake_left_bound_from_probe)
        wake_right_bound = self.background_model_gray.shape[1]
        self.wake_rect_bounds = np.array((wake_upper_bound, wake_lower_bound, wake_left_bound, wake_right_bound))

        mask = np.full_like(self.background_model_gray, False)
        mask[self.wake_rect_bounds[0]:self.wake_rect_bounds[1], self.wake_rect_bounds[2]:self.wake_rect_bounds[3]] = True
        self.wake_rect_mask = mask#.astype(bool)

    def update_U_inf_Re(self, U_inf, U_inf_std) -> None:
        self.U_inf = U_inf
        self.U_inf_std = U_inf_std

        self.Re = rho * self.U_inf * self.D_m / mu
        self.Re_std = rho * self.U_inf_std * self.D_m / mu

    def process_measurements(self) -> None:
        self.measurements_df["x_m"] = (self.measurements_df["Mid X"] - self.cylinder[0]) * self.alpha / 1e3
        self.measurements_df["y_m"] = (self.measurements_df["Mid Y"] - self.cylinder[1]) * self.alpha / 1e3
        self.measurements_df["x/D"] = (self.measurements_df["Mid X"] - self.cylinder[0]) / self.D_px
        self.measurements_df["y/D"] = (self.measurements_df["Mid Y"] - self.cylinder[1]) / self.D_px
        # self.measurements_df["x/D"] = self.transform_coords(self.measurements_df["Mid X"], self.cylinder[0])
        # self.measurements_df["y/D"] = self.transform_coords(self.measurements_df["Mid Y"], self.cylinder[1])

        self.measurements_df["dist_to_cylinder_px"] = np.linalg.norm(self.measurements_df[["Mid X", "Mid Y"]].values - self.cylinder[0:2], axis=1)

        self.measurements_df["V"] = self.alpha * self.measurements_df["l_px"] / self.t_exposire_ms # mm/ms, m/s

        upper_edge = self.wake_rect_bounds[0] - 15
        lower_edge = self.wake_rect_bounds[1] + 15

        above_wake_mask = (self.measurements_df["Line Segment Start Y"] < upper_edge) & (self.measurements_df["Line Segment End Y"] < upper_edge)
        below_wake_mask = (self.measurements_df["Line Segment Start Y"] > lower_edge) & (self.measurements_df["Line Segment End Y"] > lower_edge)
        outside_wake_mask = above_wake_mask | below_wake_mask
        # 2 diameters away from cylinder center
        away_from_cylinder_mask = self.measurements_df["dist_to_cylinder_px"] / self.nominal_diameter_px > 3

        self.measurements_df["Free Stream?"] = outside_wake_mask & away_from_cylinder_mask

        U_infs = self.measurements_df[self.measurements_df["Free Stream?"]]["V"]

        if np.any(self.measurements_df["Free Stream?"]):
            self.update_U_inf_Re(np.mean(U_infs), np.std(U_infs))
        else:
            self.update_U_inf_Re(np.max(self.measurements_df["V"]), np.nan)

    def find_pathlines(self) -> None:
        # all_frames = np.stack(self.df["Image_Gray"])
        # diffs = np.abs(imgs - self.background_model_gray)
        diffs = np.stack([cv.absdiff(img, self.background_model_gray) for img in self.df["Image_Gray"]])
        max_projection = np.max(diffs, axis=0)

        # boost contrast
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.accum_pathlines = clahe.apply(max_projection)

    def show_pathlines(self) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(9, 6))
        plt.imshow(self.accum_pathlines, cmap="gray") # "hot" colormap also looks good for pathlines
        # plt.colorbar(label="Intensity")
        # plt.title("Total Accumulated Pathlines (Max Projection)")
        return fig, ax
    
    def show_measurements(self, df=None, fig=None, ax=None):
        if df is None:
            df = self.measurements_df
        if fig is None or ax is None:
            fig, ax = self.show_pathlines()

        ps = df[["Free Stream?", "Line Segment Start X", "Line Segment End X", "Line Segment Start Y", "Line Segment End Y"]].values
        for p in ps:
            color = "red"
            label = "N/A"
            if p[0]:
                color = "green"
                label = "Free Stream"
            ax.plot([p[1], p[2]], [p[3], p[4]], lw=1.2, alpha=0.3, color=color, label=label)
        
        return fig, ax        

    def wake_model(self, coords, U_o, C_D):
        """Eq (3)"""
        x, y = coords
        deficit = 1.2 * np.sqrt(C_D * self.D_m / x)
        spread  = np.exp((-13 * y**2) / (C_D * self.D_m * x))
        return U_o * (1 - deficit * spread)

    @staticmethod
    def compute_density_weights(x, y, radius):
        """
        weight each point by 1/n_neighbors so dense clusters don't dominate the fit
        each spatial region contributes equally
        """
        coords = np.column_stack([x, y])
        tree   = cKDTree(coords)
        counts = np.array([len(tree.query_ball_point(p, radius)) for p in coords])
        return 1.0 / counts

    def fit_wake(self, weight=False):
        df_fit = self.measurements_df[self.measurements_df["x/D"] > 0]
        x = df_fit["x_m"].values # m
        y = df_fit["y_m"].values # m
        V = df_fit["V"].values   # m/s

        if weight:
            weights = Test.compute_density_weights(x / self.D_m, y / self.D_m, radius=0.5)
            popt, pcov = curve_fit(self.wake_model, (x, y), V, p0=[self.U_inf, 1.0], sigma=1.0 / weights, absolute_sigma=False)
        else:
            popt, pcov = curve_fit(self.wake_model, (x, y), V, p0=[self.U_inf, 1.0])
        
        U_inf_fit, self.C_D_fit = popt
        U_inf_std, self.C_D_std = np.sqrt(np.diag(pcov))

        self.update_U_inf_Re(U_inf_fit, U_inf_std)

        U_pred = self.wake_model((df_fit["x_m"].array, df_fit["y_m"].array), self.U_inf, self.C_D_fit)
        correlation_matrix = np.corrcoef(U_pred, df_fit["V"])
        self.r_squared = correlation_matrix[0, 1]**2

        self.cmap = plt.cm.viridis.copy()
        self.cmap.set_under(self.cmap(0.0))
        self.c_norm = colors.Normalize(vmin=0.0, vmax=self.U_inf, clip=True)
    
    def half_width(self, x):
        """Table 1, returns b (m)"""
        return 0.23 * np.sqrt(self.C_D_fit * x)
    
    def wake_edge(self, x):
        """Table 1, returns y (m)"""
        return 0.59 * np.sqrt(self.C_D_fit * self.D_m * x)
    
    def wake_model_2(self, coords, U_o, A, alpha, B, beta): 
        """
        general power law
        Eq (3) is the special case alpha = 0.5, beta = 0.5
        Converging wake: beta < 0.5
        """
        x, y = coords
        deficit = A * x**(-alpha)
        sigma   = B * x**beta
        spread  = np.exp(-(y)**2 / (2 * sigma**2))
        return U_o * (1 - deficit * spread)
    
    def fit_wake_2(self):
        df_fit = self.measurements_df[self.measurements_df["x/D"] > 0]
        x = df_fit["x_m"].values # m
        y = df_fit["y_m"].values # m
        V = df_fit["V"].values   # m/s

        popt, pcov = curve_fit(
            self.wake_model_2, (x, y), V, p0=[self.U_inf, 1.0, 0.5, 1.0, 0.5], 
            bounds=([0, 0, 0, 0, 0], [5.0, np.inf, 2, np.inf, 2])
        )
        self.popt_2 = popt
        self.pcov_2 = np.sqrt(np.diag(pcov))

        U_pred = self.wake_model_2((df_fit["x_m"].array, df_fit["y_m"].array), *self.popt_2)
        correlation_matrix = np.corrcoef(U_pred, df_fit["V"])
        self.r_squared_2 = correlation_matrix[0, 1]**2

    def show_vel_field(self, model_num=1, nx=200, ny=200, legend=True, colorbar=True, ax=None, norm=None, cmap=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if norm is None:
            norm = self.c_norm

        if cmap is None:
            cmap = self.cmap
        
        ax.invert_yaxis()
        ax.set_aspect("equal")

        img_h, img_w = self.background_model_gray.shape

        right  = (img_w - self.cylinder[0]) / self.D_px
        top    = (0     - self.cylinder[1]) / self.D_px
        bottom = (img_h - self.cylinder[1]) / self.D_px

        xD = np.linspace(0.001, right, nx)
        yD = np.linspace(top, bottom, ny)
        Xd, Yd = np.meshgrid(xD, yD)

        X = Xd * self.D_m
        Y = Yd * self.D_m

        if model_num == 1:
            U = self.wake_model((X, Y), self.U_inf, self.C_D_fit)
        elif model_num == 2:
            U = self.wake_model_2((X, Y), *self.popt_2)

        df_fit = self.measurements_df[self.measurements_df["x/D"] > 0]

        # R = 0.5
        # cylinder_mask = (Xd**2 + Yd**2) <= R**2

        circle = plt.Circle((0, 0), 0.5, color="black", zorder=10)
        ax.add_patch(circle)

        cf = ax.contourf(Xd, Yd, U, levels=np.linspace(0, np.max(U), 10), cmap=cmap, norm=norm, extend="min")
        if colorbar:
            fig.colorbar(cf, ax=ax, extend="min", label="Velocity $U$ (m/s)")

        ax.scatter(df_fit["x/D"], df_fit["y/D"], c=df_fit["V"],
            s=18,
            linewidths=1.0,
            label="Measurements",
            edgecolors="white",
            norm=norm,
            cmap=cmap
        )

        if False and model_num == 1:
            x_range = np.linspace(0, self.measurements_df["x/D"].max(), 200)
            b = self.half_width(x_range)
            y_edge = self.wake_edge(x_range)

            kw_b    = dict(color="#e03a3a", linewidth=1.4, linestyle="--", zorder=6)
            kw_edge = dict(color="#3abf5e", linewidth=1.2, linestyle=(0, (6, 3)), zorder=6)

            ax.plot(x_range, b, label=r"Wake half-width $b(x)$", **kw_b)
            ax.plot(x_range, -b, **kw_b)
            ax.plot(x_range, y_edge, label=r"Wake edge (1 $\%$)",**kw_edge)
            ax.plot(x_range, -y_edge, **kw_edge)

        ax.set_xlim(left=0)
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        # ax.set_title(f"Wake Model Velocity Field ($R^2 = {self.r_squared if model_num == 1 else self.r_squared_2:.3f}$)")
        ax.set_title(f"$R^2 = {self.r_squared if model_num == 1 else self.r_squared_2:.3f}$")
        if legend:
            ax.legend(framealpha=0.85)

        return fig, ax

    def show_wake(self, ax=None, norm=None, cmap=None, scatter=False, legend=False):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if norm is None:
            norm = self.c_norm

        if cmap is None:
            cmap = self.cmap

        # ax.invert_yaxis()
        ax.set_aspect("equal")

        # img = self.accum_pathlines.copy()
        # img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        # cv.circle(img, (int(self.cylinder[0]), int(self.cylinder[1])), int(self.cylinder[2]), (50, 255, 50), 3)

        Test.show_scaled_and_centered_image(self.accum_pathlines, scale=1/self.D_px, c_x=self.cylinder[0], c_y=self.cylinder[1], ax=ax)

        img_h, img_w = self.background_model_gray.shape

        right  = (img_w - self.cylinder[0]) / self.D_px
        top    = (0     - self.cylinder[1]) / self.D_px
        bottom = (img_h - self.cylinder[1]) / self.D_px

        xD = np.linspace(0.5, right, 200)
        yD = np.linspace(top, bottom, 200)
        Xd, Yd = np.meshgrid(xD, yD)

        X = Xd * self.D_m
        Y = Yd * self.D_m

        U = self.wake_model((X, Y), self.U_inf, self.C_D_fit)
        # R = 0.5
        # cylinder_mask = (Xd**2 + Yd**2) <= R**2
        # U = np.where(cylinder_mask, 0.0, U)

        def fmt(level):
            x = 100 * level / self.U_inf
            s = f"{x:.1f}"
            if s.endswith("0"):
                s = f"{x:.0f}"
            return rf"{s} $\%$"
        
        cs = ax.contour(Xd, Yd, U, levels=self.U_inf*np.array([0.5, 0.9, 0.99]), zorder=np.inf, cmap=cmap, norm=norm)
        ax.clabel(cs, cs.levels, fmt=fmt, fontsize=10)

        if scatter:
            mappable = ax.scatter(
                self.measurements_df["x/D"],
                self.measurements_df["y/D"],
                c=self.measurements_df["V"],
                cmap=cmap,
                norm=norm,
                alpha=0.6,
            )
            fig.colorbar(mappable, ax=ax, label="V")

        if False:
            x_range = np.linspace(0.1, self.measurements_df["x/D"].max(), 100)
            b = self.half_width(x_range)
            y_edge = self.wake_edge(x_range)

            ax.plot(x_range, b, 'r--', label="Wake boundary $b(x)$")
            ax.plot(x_range, -b, 'r--')
            ax.plot(x_range, y_edge, 'g--', label="Wake Edge (1%)")
            ax.plot(x_range, -y_edge, 'g--')
            ax.legend()

        circle = plt.Circle((0, 0), 0.5, color="red", alpha=0.4)
        ax.add_patch(circle)

        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        if legend:
            ax.legend()
        return fig, ax

    def prepare_frames(self, scaling: float | None = None, bounds: tuple | None = None) -> np.ndarray:
        """
        scale, crop, and stack
        bounds = (upper bound, lower bound, left bound, right bound)
        returns shape (frame number, y pixels, x pixels)
        """
        if scaling is None:
            all_frames = np.stack(self.df["Image_Gray"])
            if bounds is not None:
                all_frames = all_frames[:, bounds[0]:bounds[1], bounds[2]:bounds[3]]
        else:
            all_frames = []
            for img in self.df["Image_Gray"]:
                cur_img = img
                if bounds is not None:
                    # need to resize with bounds
                    cur_img = img[bounds[0]:bounds[1], bounds[2]:bounds[3]]
                all_frames.append(cv.resize(
                    cur_img, 
                    (0, 0), 
                    fx=scaling, 
                    fy=scaling, 
                    interpolation=cv.INTER_AREA
                ))
            all_frames = np.stack(all_frames)

        return all_frames

    def freq_analysis(self):
        """Returns PSD (in wake, basline / outside wake and probe, wake - baseline)"""
        scaling = 0.2
        signal = self.prepare_frames(scaling, None)
        self.all_frames = signal
        # all_frames_wake = all_frames[:, self.wake_rect_mask]
        # all_frames_probe = all_frames[:, self.probe_mask]

        N = signal.shape[0]
        T = 1.0 / self.frame_rate # period (seconds)

        # reshape to (time, pixels)

        wake_rect_mask = cv.resize(
            self.wake_rect_mask, (0, 0), fx=scaling, fy=scaling, interpolation=cv.INTER_AREA
        ).astype(bool).reshape(-1)
        probe_mask = cv.resize(
            self.probe_mask, (0, 0), fx=scaling, fy=scaling, interpolation=cv.INTER_AREA
        ).astype(bool).reshape(-1)
        signal = signal.reshape(N, -1)

        # apply window along time axis
        # window = np.hanning(N)[:, None, None]
        # signal = all_frames * window

        # print(f"{signal.shape = }")
        # print(f"{wake_rect_mask.shape = }")
        # print(f"{probe_mask.shape = }")

        # FFT along time axis
        # yf = fft(signal, axis=0, norm="ortho")[:N//2]
        # xf = fftfreq(N, T)#[:N//2]

        # power spectrum
        # power = np.abs(yf)
        xf, power = welch(signal, self.frame_rate, window="hann", axis=0) # psd
        self.xf = xf
        self.power = power
        # print(f"{power.shape = }")
        
        power_outside_wake_and_probe = power[:, ~(wake_rect_mask | probe_mask)]
        power_wake = power[:, wake_rect_mask]

        # print(f"{power_outside_wake_and_probe.shape = }")
        # print(f"{power_wake.shape = }")

        power_outside_wake_and_probe_mean = power_outside_wake_and_probe.mean(axis=1)
        power_wake_mean = power_wake.mean(axis=1)
        # print(f"{power_outside_wake_and_probe_mean.shape = }")
        # print(f"{power_wake_mean.shape = }")

        power_wake_minus_baseline_mean = power_wake_mean - power_outside_wake_and_probe_mean

        # t = np.arange(N) / self.frame_rate

        return power_wake_mean, power_outside_wake_and_probe_mean, power_wake_minus_baseline_mean
    
    def freq_heatmap(self):
        """should be run with %matplotlib qt"""
        h_w = self.all_frames.shape[1]
        w_w = self.all_frames.shape[2]

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.25) # space for slider

        initial_idx = 5
        img_data = self.power[initial_idx, :].reshape(h_w, w_w)
        im = ax.imshow(img_data, cmap="hot", aspect="auto")
        cbar = fig.colorbar(im, ax=ax, label="Power")

        ax.set_title(f"Frequency: {self.xf[initial_idx]:.2f} Hz")
        ax.axis("off")

        ax_freq = plt.axes([0.2, 0.1, 0.6, 0.03]) # [left, bottom, width, height]
        slider = Slider(
            ax=ax_freq,
            label="Target Freq (Hz) ",
            valmin =self.xf[0],
            valmax =self.xf[-1],
            valinit=self.xf[initial_idx],
            valstep=self.xf[1]-self.xf[0] # Match the frequency resolution
        )

        def update(val):
            idx = np.argmin(np.abs(self.xf - val))
            
            new_data = self.power[idx, :].reshape(h_w, w_w)
            
            im.set_data(new_data)
            
            im.set_clim(vmin=np.min(new_data), vmax=np.percentile(new_data, 99.5))
            
            ax.set_title(f"Frequency Spatial Map: {self.xf[idx]:.2f} Hz")
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show(block=True)

    def dmd_analysis(self, subtract_mean=True):
        X = self.prepare_frames(scaling=0.5, bounds=self.wake_rect_bounds)
        n_frames, height, width = X.shape
        X_flat = X.reshape(n_frames, -1).T.astype(np.float64)
        dt = 1 / self.frame_rate

        if subtract_mean:
            X_mean = np.mean(X_flat, axis=1, keepdims=True)
            X_dynamic = X_flat - X_mean
        else:
            X_dynamic = X_flat

        # lower rank to act as a low-pass filter
        # svd_rank = -1 to use all

        results = []
        for svd_rank in range(4, int(n_frames*0.12)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dmd = DMD(svd_rank=svd_rank)
                dmd.fit(X_dynamic)

            freqs = np.abs(np.imag(np.log(dmd.eigs) / (2 * np.pi * dt)))
            amps = np.abs(dmd.amplitudes)

            l1_norms = np.sum(np.abs(dmd.modes.real), axis=0)

            # plt.figure()
            # plt.stem(freqs, l1_norms)
            # plt.title(f"L1 {svd_rank=}")
            # plt.show()

            valid = (freqs >= 1) & (freqs <= 99) # no 0 or 100 Hz

            freqs = freqs[valid]
            amps = amps[valid]
            l1_norms = l1_norms[valid]

            if len(freqs) == 0:
                continue

            # idx = np.argsort(l1_norms)[:2]
            idx = np.argsort(l1_norms)[:1]
            # idx = np.argsort(l1_norms)[0]

            for i in idx:
                results.append({
                    "svd_rank":  svd_rank,
                    "frequency": freqs[i],
                    "amplitude": amps[i],
                    "L1":        l1_norms[i]
                })

        self.dmd_df = pd.DataFrame(results)
        self.dmd_df.drop_duplicates(subset=["svd_rank", "frequency"])

        # print(type(self.U_inf))

        self.dmd_df["St"] = self.dmd_df["frequency"] * self.D_m / self.U_inf

        # mode_data = dmd.modes[:, m_idx].reshape(shape).real

    def animate_images(self, extension="gif"):
        def escape(p):
            s = p.as_posix()
            s = s.replace("\\", "/") # normalize
            s = s.replace("'", r"'\''") # escape single quotes for ffmpeg
            return s

        with open("frames.txt", "w") as f:
            for p in self.df["File"]:
                f.write(f"file '{escape(p)}'\n")

        assert extension in ("gif", "mp4")

        command = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-r", "14", # playback fps
            "-i", "frames.txt",
            "-vf", "crop=iw:800:0:0", # crop height from 0-1080 px to 0-800 px 
        ]

        if extension == "mp4":
            command += [
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p"
            ]

        command.append(str(fig_path / "Videos" / (self.file_prefix + f".{extension}")))

        subprocess.run(command, stderr=subprocess.PIPE, text=True) # check=True

    def interactive(self, frame_num=0):
        row = self.df[self.df["Number"] == frame_num]
        cur_file = row["File"].item().relative_to(prog_path)
        print(cur_file)

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.3)

        ax_lower = fig.add_axes([0.25, 0.05, 0.6, 0.03])
        lower_slider = Slider(
            ax=ax_lower, label="Lower Brightness Scaler",
            valmin=0, valmax=255, valinit=30,
        )

        ax_upper = fig.add_axes([0.25, 0.2, 0.6, 0.03])
        upper_slider = Slider(
            ax=ax_upper, label="Upper Brightness Scaler",
            valmin=0, valmax=255, valinit=65
        )

        # button = Button(ax, 'Reset')

        cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
        img = ax.imshow(
            cv.bitwise_and(row["Image_Gray"].item(), row["Image_Gray"].item(), mask=self.probe_mask), 
            # row["Image_Gray"].item(), 
            cmap="gray", vmin=lower_slider.val, vmax=upper_slider.val)
        
        cur_df = self.measurements_df[self.measurements_df["Path"] == cur_file]
        self.show_measurements(cur_df, fig, ax)

        add_filtered_handles(ax)
        
        def update(val):
            img.set_clim(vmin=lower_slider.val, vmax=upper_slider.val)
            fig.canvas.draw_idle()

        lower_slider.on_changed(update)
        upper_slider.on_changed(update)
        
        plt.show(block=True)

    def to_dict(self):
        return dict(
            D_nominal_mm  = self.nominal_diameter_mm,
            D             = self.D_mm,
            grit          = self.grit,
            rel_roughness = self.rel_roughness,
            U_inf         = self.U_inf,
            Re            = self.Re,
            Re_std        = self.Re_std,
            C_d           = self.C_D_fit,
            C_d_std       = self.C_D_std,
            r_squared     = self.r_squared,
            freq          = self.dmd_df["frequency"],
            St            = self.dmd_df["St"],
            rank          = self.dmd_df["svd_rank"]
        )