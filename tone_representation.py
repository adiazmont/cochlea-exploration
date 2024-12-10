import cochlea
import thorns as th
import thorns.waves as wv
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from cochlea.stats import calc_modulation_gain
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import sys
import numpy as np


def signals_for_dbspl():
    # Signal parameters
    fs = 100e3
    duration = 0.1
    freq = 1000

    # List of dB SPL values to plot
    db_spl_values = list(range(-10, 61, 10))

    # Time axis
    time = [i / fs for i in range(int(fs * duration))]

    # Plot the signals for each dB SPL value
    plt.figure(figsize=(10, 6))

    for db_spl in db_spl_values:
        # Generate signal for the current dB SPL value
        sound = wv.ramped_tone(fs=fs, freq=freq, duration=duration, dbspl=db_spl)

        # Plot the signal
        plt.plot(time, sound, label=f"{db_spl} dB SPL")

    # Add labels and legend
    plt.title("Signals for Different dB SPLs")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file instead of displaying it
    plt.savefig("signals_for_dbspl.png", format="png")

    # Optionally, close the plot to free up memory
    plt.close()


# Function to generate ANF spike trains and save raster plots
def process_dbspl(db_spl):
    # Signal parameters
    fs = 100e3
    duration = 0.1
    freq = 1000

    # Generate signal for the current dB SPL value
    sound = wv.ramped_tone(fs=fs, freq=freq, duration=duration, dbspl=db_spl)

    # Run the ANF model to get the neural spike trains
    anf_trains = cochlea.run_zilany2014(
        sound,
        fs,
        anf_num=(3000, 5000, 4000),  # Number of fibers (HSR#, MSR#, LSR#)
        cf=100,  # Characteristic frequency (1000 Hz)
        seed=0,  # Random seed
        species='cat'  # Species type
    )

    # Plot the raster plot for the current dB SPL
    plt.figure(figsize=(10, 6))
    th.plot_raster(anf_trains)  # Plot the raster plot of the neural spike trains

    # Add title and labels
    plt.title(f"ANF Raster Plot at {db_spl} dB SPL")
    plt.xlabel("Time (s)")
    plt.ylabel("ANF Fiber Index")

    # Save the raster plot to a PNG file
    filename = f"3KHSR_5KMSR_4KLSR_anf_raster_{db_spl}dB_SPL.png"
    # plt.savefig(filename, format="png")
    plt.show()
    # Close the plot to free up memory
    plt.close()

def anf_raster_plots():
    # List of dB SPL values to plot
    db_spl_values = list(range(60, 70, 10))
    # Parallelize the execution across multiple cores
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit the task for each dB SPL value
        executor.map(process_dbspl, db_spl_values)


def calc_mod_gain():

    gains = calc_modulation_gain(
        model=cochlea.run_zilany2014,
        model_pars={'species': 'cat'}
    )

    gains.plot(logx=True)
    plt.show()

def tones():

    # Parameters
    sample_rate = 44100  # Sampling frequency (samples per second)
    duration = 2.0  # Duration in seconds
    carrier_freq = 440  # Carrier frequency in Hz
    modulation_freq = 10000  # Modulation frequency in Hz
    modulation_depth = 0.  # Modulation depth (0 to 1)

    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Generate carrier wave
    carrier = np.sin(2 * np.pi * carrier_freq * t)

    # Generate modulator wave (scaled to 0.5 to 1.0 for modulation depth)
    modulator = 1 + modulation_depth * np.sin(2 * np.pi * modulation_freq * t)

    # Apply amplitude modulation
    am_tone = carrier * modulator

    # Play the sound
    # sd.play(am_tone, sample_rate)
    # sd.wait()  # Wait until playback is finished

    # Plot the waveform
    plt.figure(figsize=(12, 6))

    # First subplot for modulator and carrier
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    plt.plot(t[:1000], modulator[:1000], 'r', label="Modulator")  # Plot the first 1000 samples
    plt.plot(t[:1000], carrier[:1000], 'y', label="Carrier")  # Plot the first 1000 samples
    plt.title("Modulator and Carrier Signals")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    # Second subplot for the amplitude modulated tone
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    plt.plot(t[:1000], am_tone[:1000], 'b', label="AM Tone")  # Plot the first 1000 samples
    plt.title("Amplitude Modulated Tone")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("mod0_carrier_and_AM_signals.png", format="png")
    plt.show()


def visualize_superimposed_tones():
    # Parameters
    sample_rate = 44100  # Sampling frequency (samples per second)
    duration = 2.0  # Duration in seconds
    carrier_freq = 440  # Carrier frequency in Hz
    modulation_frequencies = [1, 10, 100, 1000, 10000]  # Modulation frequencies in Hz
    modulation_depth = 0.5  # Modulation depth (0 to 1)

    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Initialize plot
    plt.figure(figsize=(12, 6))

    # Colors for different modulation frequencies
    colors = ['blue', 'green', 'orange', 'red', 'purple']

    # Generate and plot tones
    for mod_freq, color in zip(modulation_frequencies, colors):
        # Generate modulator wave
        modulator = 1 + modulation_depth * np.sin(2 * np.pi * mod_freq * t)

        # Generate carrier wave
        carrier = np.sin(2 * np.pi * carrier_freq * t)

        # Apply amplitude modulation
        am_tone = carrier * modulator

        # Plot the waveform (first 1000 samples for clarity)
        plt.plot(t[:1000], am_tone[:1000], label=f'fm={mod_freq} Hz', color=color, linewidth=1.5)

    # Customize plot
    plt.title("Superimposed Amplitude Modulated Tones with Varying Modulation Frequencies", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend(title="Modulation Frequency")
    plt.grid()
    plt.show()


def animate_tones():
    # Parameters
    sample_rate = 44100  # Sampling frequency (samples per second)
    duration = 0.02  # Duration in seconds
    carrier_freq = 440  # Carrier frequency in Hz
    modulation_frequencies = [1, 10, 100, 1000, 10000]  # Modulation frequencies in Hz
    modulation_depth = 1  # Modulation depth (0 to 1)
    colors = ['blue', 'green', 'orange', 'red', 'purple']  # Colors for each modulation frequency

    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Generate amplitude-modulated tones
    tones = []
    for mod_freq in modulation_frequencies:
        # Generate modulator wave
        modulator = 1 + modulation_depth * np.sin(2 * np.pi * mod_freq * t)
        # Generate carrier wave
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        # Apply amplitude modulation
        tones.append(carrier * modulator)

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Amplitude Modulated Tones with Varying Modulation Frequencies\nSample rate: 44100 Hz, 2 s, Carrier frequency: 440 Hz, Modulation depth: 1", fontsize=14)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.grid()

    # Plot settings
    line, = ax.plot([], [], lw=2)

    # Create a legend outside the plot area
    legend_handles = [plt.Line2D([0], [0], color=color, lw=2, label=f'{freq} Hz')
                      for freq, color in zip(modulation_frequencies, colors)]
    ax.legend(handles=legend_handles, title="Modulation Frequencies", loc="upper left",
              bbox_to_anchor=(1.05, 1), fontsize=10)

    # Adjust the figure to fit the legend
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    # Set the axis limits
    ax.set_xlim(0, duration)  # Display the entire 2 seconds
    ax.set_ylim(-2.5, 2.5)

    # Animation update function
    def update(frame):
        mod_freq = modulation_frequencies[frame]
        color = colors[frame]
        line.set_data(t, tones[frame])  # Use the full duration
        line.set_color(color)
        return line,

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(modulation_frequencies), interval=1000, blit=True
    )

    # Save the animation as a gif (optional)
    ani.save('amplitude_modulated_tones_002sec_cf440_md1.gif', writer='pillow', fps=1)

    plt.show()

def species_center_frequencies(num=15000, species='human'):
    # characteristic frequencies (min_cf, max_cf, num_cf)
    """
    _greenwood_pars  = {
    'human': {'A': 165.4, 'a': 60, 'k': 0.88, 'length': 35e-3},
    'cat': {'A': 456, 'a': 84, 'k': 0.8, 'length': 25e-3},

    'macaque': {'A': 0.36, 'a': 0.082, 'k': 0.85, 'length': 25.6e-3}

    the number of inner hair cells is relatively conserved among mammals
}
    :param num:
    :param species:
    :return:
    """
    if species is 'cat':
        return 1, 60000, 3500
    elif species is 'macaque':
        return 20, 45000, 3500
    else: # is human
        return 20, 20000, 3500

def species_nerve_fibers(num=15000, species='human'):
    # (HSR#, MSR#, LSR#)
    if species is 'cat':
        return 20000, 5000, 3000
    elif species is 'macaque':
        return 20000, 7000, 3000
    else: # is human
        return 20000, 10000, 3000

def map_tone_to_spike_train():
    return

def tones_to_spike_trains():
    # Define the tones to characterize

    # Define containers to record results

    # Define number of seeds == number of tests == number of models

    # Define hair cells
    # Define characteristic frequencies (min_cf, max_cf, num_cf)

    return


def test_greenwood():
    x = np.linspace(0, 35e-3, 2000)

    freq = cochlea.greenwood(x, A=165, a=60, k=1)

    plt.plot(x, freq)
    plt.show()
    return
# Call the function
# animate_tones()

# Call the function
# tones()
# calc_mod_gain()
# process_dbspl(60)
test_greenwood()
