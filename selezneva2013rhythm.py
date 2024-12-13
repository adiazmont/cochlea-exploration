import numpy as np
import random
import sounddevice as sd
random.seed(42)


# Helper functions
def db_to_amplitude(db):
    """Convert decibels to linear amplitude."""
    return 10 ** (db / 20)


def generate_harmonic_tone_with_envelope(f0, partials, duration, fs, ramp_duration, envelope_type):
    """Generate a harmonic complex tone with specified amplitude envelope."""
    t = np.linspace(0, duration / 1000, int(fs * (duration / 1000)), endpoint=False)
    signal = np.zeros_like(t)

    # Define the amplitude envelope
    if envelope_type == "A":
        amplitudes = [db_to_amplitude(-12 * (partials - n)) for n in range(partials)]  # Increasing amplitude
    elif envelope_type == "B":
        amplitudes = [db_to_amplitude(-12 * n) for n in range(partials)]  # Decreasing amplitude
        amplitudes = [db_to_amplitude(3) * amp for amp in amplitudes]  # Equal loudness adjustment for B-tones

    # Generate the harmonic components
    for n, amp in enumerate(amplitudes, start=1):
        signal += amp * np.sin(2 * np.pi * n * f0 * t)

    # Normalize the signal
    signal /= np.max(np.abs(signal))

    # Apply linear onset and offset ramps
    ramp_samples = int(fs * (ramp_duration / 1000))
    ramp = np.linspace(0, 1, ramp_samples)
    envelope = np.ones_like(signal)
    envelope[:ramp_samples] = ramp  # Onset ramp
    envelope[-ramp_samples:] = ramp[::-1]  # Offset ramp
    signal *= envelope

    return signal, t


def create_tone_sequence(tone, durations, num_short, num_medium, num_long, iois, randomize=False):
    """Generate a sequence of tones based on durations and IOIs."""
    sequence = []
    onset_times = []
    current_time = 0

    # Regular pattern: Two short tones (50 ms) followed by one long tone (200 ms)
    for _ in range(num_long):  # Each long tone corresponds to a full pattern (2 short + 1 long)
        # Add two short tones
        for _ in range(2):
            sequence.append((tone[0], durations[0]))  # Short tone
            onset_times.append(current_time)
            current_time += iois[0]  # IOI for short tones
        # Add one long tone
        sequence.append((tone[2], durations[2]))  # Long tone
        onset_times.append(current_time)
        current_time += iois[2]  # IOI for long tones

    # Randomize sequence if specified
    if randomize:
        zipped = list(zip(sequence, onset_times))
        random.shuffle(zipped)
        sequence, onset_times = zip(*zipped)

    return list(sequence), list(onset_times)


def superimpose_sequences(base_sequence, distractor_sequence, distractor_onset):
    """
    Superimpose a distractor sequence (B-tones) onto a base sequence (A-tones) starting at distractor_onset.
    After the distractor_onset, the B-tones are inserted every 200 ms after each A-tone pulse.
    The distractor onsets are no longer relevant, as they are replaced by the 200 ms spacing rule.
    Returns the combined sequence and the new onsets.

    TODO: check that this is working correctly
    """
    combined_sequence = base_sequence[:]  # Start with the base sequence
    new_onsets = [onset for _, onset in base_sequence]  # Start with the original onsets

    # Initialize the distractor sequence at the distractor_onset
    current_distractor_index = 0

    # Loop over the base sequence, and after each tone, insert a B-tone from the distractor sequence
    for tone, onset in base_sequence:
        if onset >= distractor_onset:  # Start adding B-tones after 4.8s
            if current_distractor_index < len(distractor_sequence):
                # Get the B-tone from the distractor sequence (no duration randomization)
                b_tone, _ = distractor_sequence[current_distractor_index]

                # Insert the B-tone 200 ms after the current A-tone
                new_onset = onset + 200  # B-tone onset is 200 ms after A-tone
                combined_sequence.append((b_tone, new_onset))  # Add B-tone to the sequence
                new_onsets.append(new_onset)  # Add the new onset for the B-tone
                current_distractor_index += 1  # Move to the next B-tone in the distractor sequence

    # Sort the combined sequence by onset time to ensure correct ordering
    combined_sequence.sort(key=lambda x: x[1])
    new_onsets.sort()  # Sort the new onsets to match the combined sequence order

    return combined_sequence, new_onsets

def generate_audio_from_sequence_simple(sequence, onsets, fs):
    """Combine a sequence of tones and their onsets into a single audio signal."""
    total_duration = max(onsets[-1] + len(sequence[-1][0]) / fs, (onsets[-1] + 200) / 1000)  # Calculate total duration
    audio_signal = np.zeros(int(total_duration * fs), dtype=np.float64)  # Create empty signal

    for (tone, duration), onset in zip(sequence, onsets):
        start_idx = int(onset * fs / 1000)  # Convert onset time from ms to samples
        end_idx = start_idx + len(tone)    # Calculate end index for the tone
        audio_signal[start_idx:end_idx] += tone[:len(audio_signal[start_idx:end_idx])]  # Add tone to the signal

    # Normalize the signal to avoid clipping
    audio_signal /= np.max(np.abs(audio_signal))
    return audio_signal

def generate_audio_from_sequence_complex(sequence, onsets, fs):
    """Combine a sequence of tones and their onsets into a single audio signal."""
    total_duration = max(onsets[-1] + len(sequence[-1][0]) / fs, (onsets[-1] + 200) / 1000)  # Calculate total duration
    audio_signal = np.zeros(int(total_duration * fs), dtype=np.float64)  # Create empty signal

    for (tone, duration), onset in zip(sequence, onsets):
        start_idx = int(onset * fs / 1000)  # Convert onset time from ms to samples
        end_idx = start_idx + len(tone)    # Calculate end index for the tone

        # Extract the waveform (first part of the tuple)
        tone_waveform = np.array(tone[0], dtype=np.float64)

        # Prevent out-of-bounds errors by restricting end_idx to the length of audio_signal
        if end_idx > len(audio_signal):
            end_idx = len(audio_signal)

        # Add tone to the audio signal, ensuring proper alignment
        audio_signal[start_idx:end_idx] += tone_waveform[:end_idx - start_idx]

    # Normalize the signal to avoid clipping
    audio_signal /= np.max(np.abs(audio_signal))
    return audio_signal


def test():
    # Parameters
    fs = 44100  # Sampling rate
    f0 = 440  # A-tone fundamental frequency
    partials = 10  # Number of harmonics
    ramp_duration = 5  # Linear ramp in ms
    a_tones = [generate_harmonic_tone_with_envelope(f0, partials, d, fs, ramp_duration, "A")[0] for d in [50, 100, 200]]
    b_tones = [generate_harmonic_tone_with_envelope(f0, partials, d, fs, ramp_duration, "B")[0] for d in [50, 100, 200]]

    # Simple sequences
    regular_sequence, regular_onsets = create_tone_sequence(a_tones, [50, 100, 200], 50, 0, 25, [400, 400, 400])
    irregular_sequence, irregular_onsets = create_tone_sequence(a_tones, [50, 100, 200], 50, 0, 25, [400, 400, 400],
                                                                randomize=True)

    # Distractor sequence
    distractor_sequence, distractor_onsets = create_tone_sequence(b_tones, [50, 100, 200], 42, 0, 21, [400, 400, 400],
                                                                randomize=True)

    # Superimposed sequences
    complex_regular_sequence, complex_regular_onsets = superimpose_sequences(list(zip(regular_sequence, regular_onsets)),
                                                     list(zip(distractor_sequence, distractor_onsets)), 4800)


    complex_irregular_sequence, complex_irregular_onsets = superimpose_sequences(list(zip(irregular_sequence, irregular_onsets)),
                                                       list(zip(distractor_sequence, distractor_onsets)), 4800)




    # Generate the audio signal for the regular sequence
    # audio_signal = generate_audio_from_sequence_simple(regular_sequence, regular_onsets, fs)
    audio_signal = generate_audio_from_sequence_simple(irregular_sequence, regular_onsets, fs)
    # TODO: check that the complex signals are created correctly
    # audio_signal = generate_audio_from_sequence_complex(complex_regular_sequence, complex_regular_onsets, fs)
    # audio_signal = generate_audio_from_sequence_complex(complex_irregular_sequence, complex_irregular_onsets, fs)

    # Play the audio signal
    sd.play(audio_signal, samplerate=fs)
    sd.wait()  # Wait for playback to finish
