from multiprocessing import Queue, Process
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
import numpy as np
import time
import random


class PlotProcess(Process):
    def __init__(self, out_queue):
        super().__init__()
        self.out_queue = out_queue
        # We need persistent lists to store the "jittered" coordinates
        # so they don't change every frame.
        self.x_visual = []
        self.y_visual = []

    def run(self) -> None:
        # --- 1. Dark Theme Configuration ---
        plt.style.use('dark_background')

        # Setup figure
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('black')

        # Raw values (integers) for statistics and trend lines
        x_values = []
        y_values = []

        # Start animation
        ani = animation.FuncAnimation(fig, self.update_plot,
                                      fargs=(ax, self.out_queue, x_values, y_values),
                                      interval=500, cache_frame_data=False)
        plt.show()

    def update_plot(self, frame, ax, output_queue, x_values, y_values):
        # Process all available data in the queue
        while not output_queue.empty():
            try:
                newx, newy, refresh, xtitle, ytitle = output_queue.get_nowait()
            except:
                break

            if newy is None and newx is None:
                plt.close()
                return

            if refresh:
                y_values.clear()
                x_values.clear()
                self.x_visual.clear()
                self.y_visual.clear()

            # --- JITTER CALCULATION (Only for NEW points) ---
            if len(newx) > 0:
                new_x_arr = np.array(newx, dtype=float)
                new_y_arr = np.array(newy, dtype=float)

                # Determine where to apply jitter based on axis titles
                # (Radius is discrete, so we jitter it to show density)
                jitter_x = np.zeros(len(newx))
                jitter_y = np.zeros(len(newy))

                if 'radius' in xtitle.lower():
                    jitter_x = np.random.uniform(-0.25, 0.25, size=len(newx))
                elif 'radius' in ytitle.lower():
                    jitter_y = np.random.uniform(-0.25, 0.25, size=len(newy))

                # Store the visual coordinates permanently
                self.x_visual.extend(new_x_arr + jitter_x)
                self.y_visual.extend(new_y_arr + jitter_y)

                # Store raw coordinates for stats/trend lines
                x_values.extend(newx)
                y_values.extend(newy)

        if not x_values:
            return

        # --- Plotting Logic ---
        ax.clear()

        # Styling constants
        color_bg = 'black'
        color_grid = '#333333'
        color_text = 'white'
        color_scatter = '#00E5FF'  # Cyan
        color_bar = '#FF4081'  # Neon Pink
        color_trend = '#FFFF00'  # Yellow

        ax.set_facecolor(color_bg)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color=color_grid)

        data = pd.DataFrame({'X': x_values, 'Y': y_values})

        # MODE 1: Distribution (Bar Plot)
        if 'none' in ytitle:
            counts = data['X'].value_counts().sort_index()
            df_counts = pd.DataFrame({'Radius': counts.index, 'Count': counts.values})

            bars = sns.barplot(x='Radius', y='Count', data=df_counts,
                               color=color_bar, ax=ax,
                               edgecolor='black', linewidth=1)

            for bar in bars.patches:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontweight='bold',
                            fontsize=9, color=color_text)

            ax.xaxis.get_major_locator().set_params(integer=True)

        # MODE 2: Scatter Plot (Radius vs Intensity)
        else:
            # Use the PERSISTENT visual lists (with fixed jitter)
            ax.scatter(self.x_visual, self.y_visual,
                       alpha=0.7, s=60, color=color_scatter,
                       edgecolors='white', linewidth=0.5,
                       zorder=3, label='Data Points')

            # Trend line (Calculated on RAW x_values for mathematical accuracy)
            if len(x_values) >= 2:
                try:
                    z = np.polyfit(x_values, y_values, 1)
                    p = np.poly1d(z)

                    x_trend = np.linspace(min(x_values), max(x_values), 100)
                    ax.plot(x_trend, p(x_trend),
                            color=color_trend, linewidth=2,
                            linestyle='--', alpha=0.9,
                            label='Trend', zorder=4)
                except:
                    pass

            ax.legend(loc='upper right', framealpha=0.2, labelcolor='white')

        # --- Labels & Formatting ---
        ax.set_xlabel(xtitle, fontsize=12, fontweight='bold', color=color_text)
        ax.set_ylabel(ytitle, fontsize=12, fontweight='bold', color=color_text)

        title = f'Real-time Analysis\n(n={len(x_values)} samples)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color=color_text)

        ax.tick_params(axis='both', colors=color_text, labelsize=10)

        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(1.5)

        # Stats Box
        if len(x_values) > 0:
            textstr = f'Count: {len(x_values)}\n'
            textstr += f'Mean Radius: {np.mean(x_values):.1f}\n'
            if 'none' not in ytitle:
                textstr += f'Mean Intensity: {np.mean(y_values):.1f}'

            props = dict(boxstyle='round', facecolor='#222222',
                         alpha=0.9, edgecolor='gray', linewidth=1)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=props, fontfamily='monospace', color=color_text)

        plt.tight_layout()


# --- Main Execution (Debug) ---
# if __name__ == '__main__':
#     data_queue = Queue()
#     plot_process = PlotProcess(data_queue)
#     plot_process.start()
#
#     print("Debug Started. Generating Discrete Radius (3-20) and Intensity...")
#
#     try:
#         step = 0
#         mode = "scatter"
#
#         while True:
#             step += 1
#
#             # Switch modes every 100 steps
#             if step % 100 == 0:
#                 refresh = True
#                 if mode == "scatter":
#                     mode = "distribution"
#                     print(f"[{step}] Switching to Radius Distribution Mode...")
#                     xt = "time"
#                     yt = "none"
#                 else:
#                     mode = "scatter"
#                     print(f"[{step}] Switching to Radius vs Intensity Mode...")
#                     xt = "time"
#                     yt = "Intensity (a.u.)"
#             else:
#                 refresh = False
#
#             # --- Generate Data ---
#             # 1. Radius: Discrete 3 to 20
#             new_radii = [random.randint(3, 15) for _ in range(random.randint(1, 80))]
#             new_radii2 = [random.randint(15, 20) for _ in range(random.randint(1, 20))]
#             new_radii.extend(new_radii2)
#
#             # 2. Intensity: Correlated with Radius + Noise
#             new_intensities = []
#             for r in new_radii:
#                 # Intensity drops as radius increases
#                 base_intensity = 1500
#                 noise = random.uniform(-500, 500)
#                 new_intensities.append(base_intensity + noise)
#
#             if mode == "distribution":
#                 xt = "Radius (px)"
#                 yt = "none"
#                 data_queue.put((new_radii, [0] * len(new_radii), refresh, xt, yt))
#             else:
#                 xt = "Radius (px)"
#                 yt = "Intensity (a.u.)"
#                 data_queue.put((new_radii, new_intensities, refresh, xt, yt))
#
#             time.sleep(0.5)
#
#     except KeyboardInterrupt:
#         data_queue.put((None, None, False, "", ""))
#         plot_process.join()
if __name__ == '__main__':
    data_queue = Queue()
    plot_process = PlotProcess(data_queue) # Uncomment if class is available
    plot_process.start()                   # Uncomment if class is available

    print("Debug Started. Generating 5000 samples. Peak Radius 5-8...")

    try:
        step = 0
        mode = "scatter"

        # Configuration for generation
        NUM_SAMPLES = 500
        MEAN_INTENSITY = 1500

        while True:
            step += 1

            # Switch modes every 100 steps
            if step % 100 == 0:
                refresh = True
                if mode == "scatter":
                    mode = "distribution"
                    print(f"[{step}] Switching to Radius Distribution Mode...")
                else:
                    mode = "scatter"
                    print(f"[{step}] Switching to Radius vs Intensity Mode...")
            else:
                refresh = False

            # --- Generate Data (Vectorized with Numpy) ---

            # 1. Radius: Gaussian distribution to peak around 5-8
            # loc=6.5 places the center between 5 and 8.
            # scale=2.5 ensures the "bell" covers 3-20 but tapers off.
            raw_radii = np.random.normal(loc=6.5, scale=2.5, size=NUM_SAMPLES)

            # Clip to ensure hard limits [3, 20] and round to nearest integer
            radii_np = np.clip(raw_radii, 3, 20).round().astype(int)

            # 2. Intensity: Constant Mean independent of Radius
            # Using Normal distribution for natural noise around the mean
            intensity_noise = np.random.normal(loc=0, scale=300, size=NUM_SAMPLES)
            intensities_np = MEAN_INTENSITY + intensity_noise

            # Convert to standard Python lists for Queue compatibility
            new_radii = radii_np.tolist()
            new_intensities = intensities_np.tolist()

            # --- Queue Handling ---
            if mode == "distribution":
                xt = "Radius (px)"
                yt = "Count"  # Changed from 'none' to represent histogram/dist better
                # For distribution mode, we usually just need the X values,
                # but preserving your original structure:
                data_queue.put((new_radii, [0] * len(new_radii), refresh, xt, yt))
            else:
                xt = "Radius (px)"
                yt = "Intensity"
                data_queue.put((new_radii, new_intensities, refresh, xt, yt))
            if step == 10:
                time.sleep(10)
            time.sleep(0.5)

    except KeyboardInterrupt:
        data_queue.put((None, None, False, "", ""))
        # plot_process.join()
        print("\nStopped.")

# if __name__ == '__main__':
#     # Create communication queue
#     data_queue = Queue()
#
#     # Start the plotting process
#     plot_process = PlotProcess(data_queue)
#     plot_process.start()
#
#     print("Lysozome Simulation Started. Press Ctrl+C to stop.")
#
#     try:
#         # Parameters from Code2
#         n_timepoints = 5
#         n_lysosomes = 1000  # Points per batch
#         base_intensity = 1000
#         intensity_increase_rate = 10
#         noise_std = 15
#
#         while True:
#             # Loop through time points 1 to 5 (mimicking Code2's structure)
#             for t in range(1, n_timepoints + 1):
#                 # --- 1. Generate Intensity (Y) Data (Logic from Code2) ---
#                 # Mean intensity increases linearly with time
#                 current_rate = np.random.normal(intensity_increase_rate, 0.1)
#                 mean_intensity = base_intensity + (current_rate * t)
#
#                 # Generate individual lysosome intensities with noise
#                 # We use numpy here for speed, similar to Code2
#                 new_y = np.random.normal(mean_intensity, noise_std, n_lysosomes)
#                 # new_y = np.maximum(new_y, 0)  # Ensure non-negative
#
#                 # --- 2. Generate Time (X) Data with Jitter (Logic from Code2) ---
#                 # Code2 uses: x_jitter = np.random.normal(t, 0.1, n_lysosomes)
#                 # We generate this HERE so the plot process renders it exactly like Code2
#                 new_x = np.random.normal(t, 0.1, n_lysosomes)
#
#                 # --- 3. Control Logic ---
#                 # If t=1, we refresh (clear) the plot to start a new cycle
#                 refresh = (t == 1)
#
#                 xt = "Time Point"
#                 yt = "Lysosome Intensity"
#
#                 # Send to queue
#                 # PlotProcess expects lists or arrays
#                 data_queue.put((new_x, new_y, refresh, xt, yt))
#
#                 print(f"Generated Timepoint {t}: Mean Intensity ~{mean_intensity:.1f}")
#
#                 # Sleep to visualize the 'step-by-step' appearance
#                 time.sleep(1.5)
#
#             print("Cycle complete. Restarting in 3 seconds...")
#             time.sleep(3)
#
#     except KeyboardInterrupt:
#         print("\nStopping...")
#         data_queue.put((None, None, False, "", ""))  # Poison pill
#         plot_process.join()
#         print("Done.")

