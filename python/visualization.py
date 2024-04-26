from __future__ import print_function
from __future__ import division
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import config
import microphone   
import dsp
import led

from collections import deque

# Configuration for delay
DELAY_MS = config.DELAY  # milliseconds of delay
DELAY_SAMPLES = int((config.MIC_RATE / 1000) * DELAY_MS)
audio_delay_buffer = deque(maxlen=DELAY_SAMPLES)

_time_prev = time.time() * 1000.0
"""The previous time that the frames_per_second() function was called"""

_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)
"""The low-pass filter used to estimate frames-per-second"""


def frames_per_second():
    """Return the estimated frames per second

    Returns the current estimate for frames-per-second (FPS).
    FPS is estimated by measured the amount of time that has elapsed since
    this function was previously called. The FPS estimate is low-pass filtered
    to reduce noise.

    This function is intended to be called one time for every iteration of
    the program's main loop.

    Returns
    -------
    fps : float
        Estimated frames-per-second. This value is low-pass filtered
        to reduce noise.
    """
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)


def memoize(function):
    """Provides a decorator for memoizing functions"""
    from functools import wraps
    memo = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper


@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)


def interpolate(y, new_length):
    """Intelligently resizes the array by linearly interpolating the values

    Parameters
    ----------
    y : np.array
        Array that should be resized

    new_length : int
        The length of the new interpolated array

    Returns
    -------
    z : np.array
        New array with length of new_length that contains the interpolated
        values of y.
    """
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z


r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS),
                       alpha_decay=0.2, alpha_rise=0.99)
g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS),
                       alpha_decay=0.05, alpha_rise=0.3)
b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS),
                       alpha_decay=0.1, alpha_rise=0.5)
common_mode = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS),
                       alpha_decay=0.99, alpha_rise=0.01)
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS)),
                       alpha_decay=0.1, alpha_rise=0.99)
p = np.tile(1.0, (3, config.N_PIXELS))
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                     alpha_decay=0.001, alpha_rise=0.99)


def visualize_scroll(y):
    """Effect that originates in the center and scrolls outwards"""
    global p
    y = y**2.0
    gain.update(y)
    y /= gain.value/2
    y *= 255.0
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))*2
    b = int(np.max(y[2 * len(y) // 3:]))*2
    # Scrolling effect window
    p[:, 1:] = p[:, :-1]
    p *= .99
    p = gaussian_filter1d(p, sigma=0.2)
    # Create new color originating at the center
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b
    # Update the LED strip
    return p
    return np.concatenate((p[:, ::-1], p), axis=1)


def visualize_energy(y):
    """Effect that expands from the center with increasing sound energy"""
    global p
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= float((config.N_PIXELS // 2) - 1)
    # Map color channels according to energy in the different freq bands
    scale = 1.1
    r = int(np.mean(y[:len(y) // 3]**(scale+0.15)))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = int(np.mean(y[2 * len(y) // 3:]**(scale-0.1)))
    # Assign color to different frequency regions
    p[0, :r] = 255.0
    p[0, r:] = 0.0
    p[1, :g] = 255.0
    p[1, g:] = 0.0
    p[2, :b] = 255.0
    p[2, b:] = 0.0
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    p[0, :] = gaussian_filter1d(p[0, :], sigma=4.0)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=4.0)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=4.0)
    # Set the new pixel value
    return p



def visualize_spectrum(y):
    global _prev_spectrum  # Ensure _prev_spectrum is defined at the global level

    # Define a threshold below which the LEDs should be black
    threshold = 0.1  # Adjust this value based on experimentation

    # Check if _prev_spectrum needs initialization (e.g., first run)
    if '_prev_spectrum' not in globals():
        _prev_spectrum = np.tile(0.01, config.N_PIXELS)

    r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS),
                           alpha_decay=0.3, alpha_rise=0.8)
    g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS),
                           alpha_decay=0.15, alpha_rise=0.3)
    b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS),
                           alpha_decay=0.1, alpha_rise=0.5)
    common_mode = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS),
                                alpha_decay=0.99, alpha_rise=0.01)

    y = np.copy(interpolate(y, config.N_PIXELS))
    common_mode.update(y)
    diff = y - _prev_spectrum
    _prev_spectrum = np.copy(y)  # Update the previous spectrum

    # Update filter outputs
    r = r_filt.update(y - common_mode.value)
    g = g_filt.update(np.abs(diff))
    b = b_filt.update(np.copy(y))

    # # Apply threshold check
    # r[r < threshold] = 0
    # g[g < threshold] = 0
    # b[b < threshold] = 0

    # Scale to RGB range, no mirroring needed
    r = r * 255 
    g = g * 500
    b = b * 255 

    # Combine into a single array to return
    output = np.array([r, g, b])

    return output


def visualize_spectrum_mirror(y):
    global _prev_spectrum

    # Check if _prev_spectrum needs initialization (e.g., first run)
    if '_prev_spectrum' not in globals():
        _prev_spectrum = np.tile(0.01, config.N_PIXELS // 2)

    r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                        alpha_decay=0.2, alpha_rise=0.99)
    g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                        alpha_decay=0.05, alpha_rise=0.3)
    b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                        alpha_decay=0.1, alpha_rise=0.5)
    common_mode = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                        alpha_decay=0.99, alpha_rise=0.01)
    p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)),
                        alpha_decay=0.1, alpha_rise=0.99)
    p = np.tile(1.0, (3, config.N_PIXELS // 2))
    
    """Effect that maps the Mel filterbank frequencies onto the LED strip"""
    y = np.copy(interpolate(y, config.N_PIXELS // 2))
    common_mode.update(y)
    diff = y - _prev_spectrum
    _prev_spectrum = np.copy(y)
    # Color channel mappings
    r = r_filt.update(y - common_mode.value)
    g = np.abs(diff)
    b = b_filt.update(np.copy(y))
    # Mirror the color channels for symmetric output
    r = np.concatenate((r[::-1], r))
    g = np.concatenate((g[::-1], g))
    b = np.concatenate((b[::-1], b))
    output = np.array([r, g,b]) * 255
    return output

def visualize_spectrum_colorful(y):
    global last_change_time, current_color_index, current_color, next_color
    
    if 'last_change_time' not in globals():
        last_change_time = time.time()
    if 'current_color_index' not in globals():
        current_color_index = 0
    if 'colors' not in globals():
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 0, 128)]  # RGB for Red, Blue, Green, Purple
    if 'current_color' not in globals():
        current_color = np.array(colors[current_color_index])
    if 'next_color' not in globals():
        next_color = np.array(colors[(current_color_index + 1) % len(colors)])

    color_duration = 10  # seconds each color is displayed
    fade_duration = 3    # seconds for color transition

    def update_color():
        global last_change_time, current_color_index, current_color, next_color
        # Calculate elapsed time
        time_now = time.time()
        time_elapsed = time_now - last_change_time

        # Check if it's time to update the current and next colors
        if time_elapsed > color_duration + fade_duration:
            current_color_index = (current_color_index + 1) % len(colors)
            current_color = np.array(colors[current_color_index])
            next_color = np.array(colors[(current_color_index + 1) % len(colors)])
            last_change_time = time_now

        # Calculate current display color based on timing
        if time_elapsed > color_duration:
            fade_progress = (time_elapsed - color_duration) / fade_duration
            current_display_color = (1 - fade_progress) * current_color + fade_progress * next_color
        else:
            current_display_color = current_color

        return current_display_color.flatten()

    current_display_color = update_color() / 255  # Normalize the color for intensity calculations

    # Use the existing spectrum visualization logic here to get the LED values
    spectrum_output = visualize_spectrum_mirror(y) 

    # Apply the current display color to the spectrum output
    colored_output = np.array([spectrum_output[0] * current_display_color[0],
                               spectrum_output[1] * current_display_color[1],
                               spectrum_output[2] * current_display_color[2]])

    return colored_output



fft_plot_filter = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.5, alpha_rise=0.99)
mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.5, alpha_rise=0.99)
volume = dsp.ExpFilter(config.MIN_VOLUME_THRESHOLD,
                       alpha_decay=0.02, alpha_rise=0.02)
fft_window = np.hamming(int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)
prev_fps_update = time.time()


def microphone_update(audio_samples):
    global y_roll, prev_rms, prev_exp, prev_fps_update

    # Add incoming samples to the delay buffer
    audio_delay_buffer.extend(audio_samples)
    if len(audio_delay_buffer) < DELAY_SAMPLES:
        return  # Wait until buffer is full

    # Use only the oldest samples which have been delayed appropriately
    delayed_samples = np.array(audio_delay_buffer)[:samples_per_frame]
    audio_delay_buffer.popleft()  # Remove the processed samples

    # Normalize samples between 0 and 1
    y = delayed_samples / 2.0**15
    # Construct a rolling window of audio samples
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(y)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)
    
    # Rest of your existing processing code follows here
    # For example:
    vol = np.max(np.abs(y_data))
    if vol < config.MIN_VOLUME_THRESHOLD:
        print('No audio input. Volume below threshold. Volume:', vol)
        led.pixels = np.tile(0, (3, config.N_PIXELS))
        led.update()
    else:
        # Transform audio input into the frequency domain
        N = len(y_data)
        N_zeros = 2**int(np.ceil(np.log2(N))) - N
        # Pad with zeros until the next power of two
        y_data *= fft_window
        y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
        YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
        # Construct a Mel filterbank from the FFT data
        mel = np.atleast_2d(YS).T * dsp.mel_y.T
        # Scale data to values more suitable for visualization
        # mel = np.sum(mel, axis=0)
        mel = np.sum(mel, axis=0)
        mel = mel**2.0
        # Gain normalization
        mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        mel /= mel_gain.value
        mel = mel_smoothing.update(mel)
        # Map filterbank output onto LED strip
        output = visualization_effect(mel)
        led.pixels = output
        led.update()
        if config.USE_GUI:
            # Plot filterbank output
            x = np.linspace(config.MIN_FREQUENCY, config.MAX_FREQUENCY, len(mel))
            mel_curve.setData(x=x, y=fft_plot_filter.update(mel))
            # Plot the color channels
            r_curve.setData(y=led.pixels[0])
            g_curve.setData(y=led.pixels[1])
            b_curve.setData(y=led.pixels[2])
    if config.USE_GUI:
        app.processEvents()
    
    if config.DISPLAY_FPS:
        fps = frames_per_second()
        if time.time() - 0.5 > prev_fps_update:
            prev_fps_update = time.time()
            print('FPS {:.0f} / {:.0f}'.format(fps, config.FPS))


# Number of audio samples to read every time frame
samples_per_frame = int(config.MIC_RATE / config.FPS)

# Array containing the rolling audio sample window
y_roll = np.random.rand(config.N_ROLLING_HISTORY, samples_per_frame) / 1e16

visualization_effect = visualize_spectrum
"""Visualization effect to display on the LED strip"""


if __name__ == '__main__':
    if config.USE_GUI:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtGui, QtCore
        # Create GUI window
        app = QtGui.QApplication([])
        view = pg.GraphicsView()
        layout = pg.GraphicsLayout(border=(100,100,100))
        view.setCentralItem(layout)
        view.show()
        view.setWindowTitle('Visualization')
        view.resize(800,600)
        # Mel filterbank plot
        fft_plot = layout.addPlot(title='Filterbank Output', colspan=3)
        fft_plot.setRange(yRange=[-0.1, 1.2])
        fft_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        x_data = np.array(range(1, config.N_FFT_BINS + 1))
        mel_curve = pg.PlotCurveItem()
        mel_curve.setData(x=x_data, y=x_data*0)
        fft_plot.addItem(mel_curve)
        # Visualization plot
        layout.nextRow()
        led_plot = layout.addPlot(title='Visualization Output', colspan=3)
        led_plot.setRange(yRange=[-5, 260])
        led_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        # Pen for each of the color channel curves
        r_pen = pg.mkPen((255, 30, 30, 200), width=4)
        g_pen = pg.mkPen((30, 255, 30, 200), width=4)
        b_pen = pg.mkPen((30, 30, 255, 200), width=4)
        # Color channel curves
        r_curve = pg.PlotCurveItem(pen=r_pen)
        g_curve = pg.PlotCurveItem(pen=g_pen)
        b_curve = pg.PlotCurveItem(pen=b_pen)
        # Define x data
        x_data = np.array(range(1, config.N_PIXELS + 1))
        r_curve.setData(x=x_data, y=x_data*0)
        g_curve.setData(x=x_data, y=x_data*0)
        b_curve.setData(x=x_data, y=x_data*0)
        # Add curves to plot
        led_plot.addItem(r_curve)
        led_plot.addItem(g_curve)
        led_plot.addItem(b_curve)
        # Frequency range label
        freq_label = pg.LabelItem('')
        # Frequency slider
        def freq_slider_change(tick):
            minf = freq_slider.tickValue(0)**2.0 * (config.MIC_RATE / 2.0)
            maxf = freq_slider.tickValue(1)**2.0 * (config.MIC_RATE / 2.0)
            t = 'Frequency range: {:.0f} - {:.0f} Hz'.format(minf, maxf)
            freq_label.setText(t)
            config.MIN_FREQUENCY = minf
            config.MAX_FREQUENCY = maxf
            dsp.create_mel_bank()
        freq_slider = pg.TickSliderItem(orientation='bottom', allowAdd=False)
        freq_slider.tickMoveFinished = freq_slider_change
        freq_slider.addTick((config.MIN_FREQUENCY / (config.MIC_RATE / 2.0))**0.5)
        freq_slider.addTick((config.MAX_FREQUENCY / (config.MIC_RATE / 2.0))**0.5)
        freq_label.setText('Frequency range: {} - {} Hz'.format(
            config.MIN_FREQUENCY,
            config.MAX_FREQUENCY))
        # Brightness label
        bright_label = pg.LabelItem('')
        # Brightness slider
        def bright_slider_change(tick):
            newBrightness = bright_slider.tickValue(0)
            bright_label.setText('Brightness level: {:.0f}%'.format(newBrightness*100))
            config.BRIGHTNESS = newBrightness
        bright_slider = pg.TickSliderItem(orientation='bottom', allowAdd=True)
        bright_slider.addTick(config.BRIGHTNESS, color='#16dbeb' , movable=True)
        bright_slider.tickMoveFinished = bright_slider_change
        bright_label.setText('Brightness: {:.0f}%'.format(bright_slider.tickValue(0)*100))
        # Effect selection
        active_color = '#16dbeb'
        inactive_color = '#FFFFFF'
        def energy_click(x):
            global visualization_effect
            visualization_effect = visualize_energy
            energy_label.setText('Energy', color=active_color)
            scroll_label.setText('Scroll', color=inactive_color)
            spectrum_label.setText('Spectrum', color=inactive_color)
        def scroll_click(x):
            global visualization_effect
            visualization_effect = visualize_scroll
            energy_label.setText('Energy', color=inactive_color)
            scroll_label.setText('Scroll', color=active_color)
            spectrum_label.setText('Spectrum', color=inactive_color)
        def spectrum_click(x):
            global visualization_effect
            visualization_effect = visualize_spectrum_colorful
            energy_label.setText('Energy', color=inactive_color)
            scroll_label.setText('Scroll', color=inactive_color)
            spectrum_label.setText('Spectrum', color=active_color)
        # Create effect "buttons" (labels with click event)
        energy_label = pg.LabelItem('Energy')
        scroll_label = pg.LabelItem('Scroll')
        spectrum_label = pg.LabelItem('Spectrum')
        energy_label.mousePressEvent = energy_click
        scroll_label.mousePressEvent = scroll_click
        spectrum_label.mousePressEvent = spectrum_click
        #energy_click(0)
        spectrum_click(0)
        # Layout
        layout.nextRow()
        layout.addItem(freq_label, colspan=3)
        layout.nextRow()
        layout.addItem(freq_slider, colspan=3)
        layout.nextRow()
        layout.addItem(energy_label)
        layout.addItem(scroll_label)
        layout.addItem(spectrum_label)
        layout.nextRow() 
        layout.addItem(bright_label, colspan=3)
        layout.nextRow() 
        layout.addItem(bright_slider, colspan=3)
    # Initialize LEDs
    else:
        visualization_effect = visualize_energy
    led.update()
    # Start listening to live audio stream

    microphone.start_stream(microphone_update)