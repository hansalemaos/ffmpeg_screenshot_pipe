# Capture Screenshots with Unparalleled Speed and Versatile Features - GDIgrab, DDAgrab, Ctypes, Multiprocessing, GPU, Mouse Capture ...


### pip install ffmpeg-screenshot-pipe (ffmpeg.exe included)

## What is so special about ffmpeg-screenshot-pipe

- 3 methods of capturing screenshots: GDIgrab, DDAgrab or Ctypes
- Faster than all other libraries 
- Efficient mouse capture (with/without)
- Integration with FFmpeg for exceptional speed
- Capture windows running in the background
- Only a few dependencies - all of them pure Python except Numpy 
- Specify windows using HWND and (regex)search for targeted capture
- GPU acceleration for enhanced performance
- High-quality screen recordings and screenshots
- Versatile and powerful functionality
- Seamless integration with numpy for data processing workflows
- Fine-grained control over screen capture settings
- Minimal performance impact
- Suitable for content creators, software developers, and individuals with diverse screen capture needs.
- Multiprocessing for capturing various windows/screens/areas 

FFmpegshot is a versatile and powerful class designed for capturing screen recordings and screenshots. It offers a wide range of options and features that make it suitable for various use cases.

This class is particularly useful for people who require high-quality screen capture capabilities with advanced functionality. It provides efficient and fast screen capturing capabilities, thanks to its integration with 
[FFmpeg - the Gold Standard of Video Encoding/Decoding](https://ffmpeg.org/) , all captured data is conveniently converted to the popular [Numpy](https://numpy.org/doc/stable/index.html#) format, enabling seamless integration with other data processing workflows.

One of the key advantages of FFmpegshot is its exceptional speed, surpassing all other screen capturing methods. By leveraging FFmpeg's optimized encoding and decoding algorithms, it ensures efficient processing and minimal performance impact, resulting in swift and responsive screen captures.

Furthermore, FFmpegshot offers a plethora of options to cater to diverse needs. Users can capture mouse movements, extract frames from background windows, specify individual windows using HWND for capture, and even leverage GPU acceleration for enhanced performance. These features provide users with fine-grained control and flexibility over their screen capture requirements.

Whether it's for content creators seeking professional-grade screen recordings, software developers needing precise visual debugging tools, or individuals wanting efficient and reliable screen capture capabilities, FFmpegshot stands out as a powerful solution. Its combination of speed, versatility, and an array of options makes it an ideal choice for capturing and processing screen data.

## Very important: Find the right frame rate (fps)! 

### Find the optimal frame rate based on performance testing to avoid buffer read/write balance issue.


### Read this part very carefully!

```python

from ffmpeg_screenshot_pipe import FFmpegshot, get_max_framerate

# Use this function to get a rough idea how high you can go! 

mafa = get_max_framerate(
	function="capture_all_screens_gdigrab",
	startframes=45,
	endframes=150,
	timeout=2,
	framedifference=100,
	sleeptimebeforekilling=1,
)

# Frame rate testing results:
# 64 FPS -> 115 frames captured
# 65 FPS -> 115 frames captured
# 66 FPS -> 119 frames captured
# 67 FPS -> 120 frames captured
# 68 FPS -> 121 frames captured
# 69 FPS -> 123 frames captured
# 70 FPS -> 176 frames captured A difference of more than a hundred frames -> (buffer read/write balance issue)



# Based on the results, it seems that going beyond 69 frames per second may cause buffer read/write balance issues
# (on my PC - Intel i5, NVIDIA GeForce RTX 2060 SUPER, 64 GB RAM)
# It is advisable to stay a few frames below the limit to ensure smooth performance.
# Note that setting a very high frame rate (e.g., 240 FPS) may result in significantly lower actual frame rates
# due to buffer constraints.

```

#### Some functions to test the performance of this module


```python
def show_screenshot(screenshot):
    cv2.imshow("test", screenshot)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        return False
    return True
def show_screenshot_bench(
    screenshotiter, stop_at_frame=100, quitkey="q", show_screenshot=True
):
    def show_screenshotx():
        cv2.imshow("test", screenshot)
        if cv2.waitKey(1) & 0xFF == ord(quitkey):
            cv2.destroyAllWindows()
            return False
        return True

    framecounter = 0
    fps = 0
    start_time = time()
    for screenshot in screenshotiter:
        if stop_at_frame:
            if framecounter > stop_at_frame:
                break
            framecounter += 1
        if show_screenshot:
            sho = show_screenshotx()
        else:
            sho = True
        fps += 1
        if not sho:
            break
    print(f"fast_ctypes_screenshots: {fps / (time() - start_time)}")
    cv2.destroyAllWindows()
```

## DDAgrab methods (GPU)

Captures the Windows Desktop via Desktop Duplication API (D3D11) - same that OBS uses.
Note: Nvidia GPU is required to utilize these functions effectively

### Convenient usage with "with" statement

All functions with the suffix _ddagrab/_gdigrab can be called using the "with" statement


### DDAgrab - capture one screen 

[Check out the result - Video](https://github.com/hansalemaos/ffmpeg_screenshot_pipe/raw/main/examples/capture_one_screen_ddagrab_60_frames/video_588.mp4)

```python
	

with FFmpegshot() as sc:
	start_time, fps = time(), 0
	for screenshot in sc.capture_one_screen_ddagrab(
			monitor_index=0,
			frames=60,
			draw_mouse=True,
	):
		cont = show_screenshot(screenshot)
		if not cont:
			break
		fps += 1
	print(f'FPS: {fps / (time() - start_time)}')

#####################################################################		

# Reminder: Ensure to terminate FFmpeg instances if not using "with" statement
ffmpeg_screenshot = FFmpegshot()
# Capture and display screenshots
piciter = ffmpeg_screenshot.capture_one_screen_ddagrab(
	monitor_index=0,
	frames=60,
	draw_mouse=True,
)
# Terminate FFmpeg instances after usage
show_screenshot_bench(
	piciter, stop_at_frame=100, quitkey="q", show_screenshot=True
)
# don't forget to kill the ffmpeg instances after you are done
ffmpeg_screenshot.kill_ffmpeg()
	
```
### DDAgrab - capture all screens

[Check out the result - Video](https://github.com/hansalemaos/ffmpeg_screenshot_pipe/raw/main/examples/capture_all_screens_ddagrab_24_frames/video_228.mp4)

```python
with FFmpegshot() as sc:
	timestamp = timest()
	piciter = sc.capture_all_screens_ddagrab(  # generator
		frames=24, draw_mouse=True, return_empty=False
	)
	# Do something here ....


	# Important Note: The screenshots are taken individually and need to be concatenated for a continuous stream.
	# Traditional methods like cv2.hstack, cv2.vstack, or np.concatenate are too slow for real-time streaming.
	# When setting return_empty=True, the function returns an empty array, and the screenshot can be accessed
	# through temparrays.horizontal / temparrays.vertical (using https://github.com/hansalemaos/fastimgconcat).
	# The values in temparrays.vertical / temparrays.horizontal will be updated in the next iteration to avoid
	# allocating new memory. It is recommended to process the output data immediately after each iteration.
	# If you still need the arrays, make a copy (e.g., temparrays.horizontal.copy()), but keep in mind that copying
	# is an expensive operation.
	# The maximum frame rate achievable depends on the output video. For streaming purposes, up to 45 frames per
	# second (on my Intel i5, NVIDIA GeForce RTX 2060 SUPER, 64 RAM) can be achieved, but it is going to be much
	# lower if you write the file to your HDD..
```

### DDAgrab - capture a screenshot of a certain area

[Check out the result - Video](https://github.com/hansalemaos/ffmpeg_screenshot_pipe/raw/main/examples/capture_box_ddagrab_60_frames_1500x800/video_590.mp4)

```python

with FFmpegshot() as sc:
	timestamp = timest()
	piciter = sc.capture_box_ddagrab(  # generator 
		rectangle=(100, 200, 1600, 1000), # left, top, right, bottom
		frames=60,
		draw_mouse=True,
	)
	

```


### DDAgrab - Capturing a window using GPU acceleration

[Check out the result - Video](https://github.com/hansalemaos/ffmpeg_screenshot_pipe/raw/main/examples/capture_capture_window_ddagrab_notpad_60_client/video_589.mp4)


[Check out the result - Video](https://github.com/hansalemaos/ffmpeg_screenshot_pipe/raw/main/examples/capture_capture_window_ddagrab_notpad_60_window/video_588.mp4)



To capture a window using the GPU, you can leverage the power of FFmpegshot with the following considerations:

Restricting the screen capture to a specific screen: If you have multiple screens, it is recommended to 
capture only the screen where your application is running. For example, if your app is on the 
first screen, you can pass allowed_screens=(0,). By doing so, the capture process becomes 
more efficient. If you don't specify allowed_screens and have multiple screens, the capture process 
may become slower because the images are concatenated before being cropped to the appropriate size.

Limitations with capturing background windows: 
This function is not designed to capture background windows (see gdigrab). 
When you resize the window being captured, 
the resulting screenshot will also be resized accordingly.

In addition, you can utilize the searchdict dictionary to search for specific windows based on various 
criteria such as pid, title, windowtext, hwnd, length, tid, status, coords_client, dim_client, 
coords_win, dim_win, class_name, and path. This allows you to target 
specific windows for capturing. More details about the search criteria 
can be found in the appshwnd GitHub repository.  https://github.com/hansalemaos/appshwnd

```python

# Here is an overview:
searchdict = {
    "pid": 1004,
    "pid_re": "^1.*",
    "title": "IME",
    "title_re": "IM?E",
    "windowtext": "Default IME",
    "windowtext_re": r"Default\s*IME",
    "hwnd": 197666,
    "hwnd_re": r"\d+",
    "length": 12,
    "length_re": "[0-9]+",
    "tid": 6636,
    "tid_re": r"6\d+36",
    "status": "invisible",
    "status_re": "(?:in)?visible",
    "coords_client": (0, 0, 0, 0),
    "coords_client_re": r"\([\d,\s]+\)",
    "dim_client": (0, 0),
    "dim_client_re": "(1?0, 0)",
    "coords_win": (0, 0, 0, 0),
    "coords_win_re": r"\)$",
    "dim_win": (0, 0),
    "dim_win_re": "(1?0, 0)",
    "class_name": "IME",
    "class_name_re": "I?ME$",
    "path": "C:\\Windows\\ImmersiveControlPanel\\SystemSettings.exe",
    "path_re": "SystemSettings.exe",
}


capture_window_ddagrab(
	searchdict={
		"title_re": ".*Notepad.*",
		"status": "visible",
		"path_re": ".*notepad.exe.*",
	},
	frames=30,
	draw_mouse=True,
	allowed_screens=(), # Slow
)
capture_window_ddagrab(
	searchdict={
		"title_re": ".*Notepad.*",
		"status": "visible",
		"path_re": ".*notepad.exe.*",
	},
	frames=30,
	draw_mouse=True,
	allowed_screens=(0,), # Fast - captures only first screen
)
```

## GDIgrab

### Capturing (background) windows using GDIgrab

In addition to capturing windows using GPU acceleration, FFmpegshot also provides the
option to capture windows using gdigrab. Here are some important details about using gdigrab for window capture:

Capturing background windows: With gdigrab, it is possible to capture background windows. 
This means you can capture windows that are not in the foreground or currently active. 
It provides more flexibility in capturing specific windows of interest.

Resizing window capture: Unlike GPU-accelerated window capture, when you resize the window being 
captured using gdigrab, the resulting screenshot will not be automatically resized. 
The screenshot will maintain the original size of the window, regardless of any subsequent resizing.

Handling windows with the same title: One advantage of using this method is that it bypasses the problem of 
capturing windows with the same title. When using the title parameter in gdigrab, 
it creates a unique identification for the window for a short duration (e.g., 2 seconds)
 until the capturing starts. This ensures that the correct window with the specified 
 title is captured, even if multiple windows share the same title.

To utilize gdigrab for window capture and overcome the issue of multiple windows with the same title, 
you can use the capture_window_gdigrab function provided by FFmpegshot. 
By specifying search criteria in the searchdict parameter, including the window title or other properties, you can precisely target the desired window for capture.

By incorporating gdigrab into your screen capture workflow, you can capture background windows, 
handle windows with the same title, and obtain accurate representations of window content.
This offers additional flexibility and reliability for capturing specific windows of interest
in your screen recording or screenshot tasks.

### Capturing a Window 

[Check out the result - Video](https://github.com/hansalemaos/ffmpeg_screenshot_pipe/raw/main/examples/capture_capture_window_gdigrab_notpad_60/video_591.mp4)

```python

with FFmpegshot() as sc:
	timestamp = timest()
	piciter = sc.capture_window_gdigrab( # generator
		searchdict={
			"title_re": ".*Notepad.*",
			"status": "visible",
			"path_re": ".*notepad.exe.*",
		},
		frames=60,
		draw_mouse=True,
	)

```
### Capturing 2 screens (3840x1080)

[Check out the result - Video](https://github.com/hansalemaos/ffmpeg_screenshot_pipe/raw/main/examples/capture_all_screens_gdigrab_60/video_586.mp4)

The fastest methods on my PC for such a big area

```python
with FFmpegshot() as sc:
	timestamp = timest()
	piciter = sc.capture_all_screens_gdigrab( # generator
		frames=60,
		draw_mouse=True,
	)
```

### Capturing 1 screen

[Check out the result - Video](https://github.com/hansalemaos/ffmpeg_screenshot_pipe/raw/main/examples/capture_one_screen_gdigrab_60/video_590.mp4)


```python
with FFmpegshot() as sc:
	piciter = sc.capture_one_screen_gdigrab(
		monitor_index=0,
		frames=60,
		draw_mouse=True,
	)
```


## Ctypes


### Ideal for scenarios with minimal screenshot requirements


In addition to the impressive functionality offered by FFmpegshot, you can also utilize ctypes for capturing screens. FFmpegshot incorporates a supplementary module called [fast_ctypes_screenshots](https://github.com/hansalemaos/fast_ctypes_screenshots), specifically designed to enable screen capturing using ctypes.

Starting FFmpeg incurs a substantial overhead, especially if you only require a small number of screenshots. In such cases, you can employ these ctypes-based methods, which deliver exceptional speed and efficiency.

The following static functions are at your disposal for ctypes-based screen capture:

### capture_all_screens_ctypes(ascontiguousarray=False)

Captures all screens using ctypes, allowing you to specify whether to return an array as a contiguous memory block (ascontiguousarray).


### capture_one_screen_ctypes(monitor=0, ascontiguousarray=False)

Captures a specific screen using ctypes, with an optional parameter to specify the monitor index.


### capture_box_ctypes(rectangle: tuple[int, int, int, int], ascontiguousarray=False)

Captures a region defined by a rectangle using ctypes, with an optional parameter to specify whether to return a contiguous memory block.


### capture_window_window_ctypes(hwnd: int, ascontiguousarray=False)

Captures a window using ctypes based on its window handle (HWND), with an optional parameter to control the memory block.


### capture_window_client_ctypes(hwnd: int, ascontiguousarray=False)

Captures the client area of a window using ctypes based on its window handle (HWND), with an optional parameter to specify the memory block.


Here's an example of capturing all screens using ctypes in combination with FFmpegshot:

In this example, the capture_all_screens_ctypes function is used to capture all screens, with the 
ascontiguousarray parameter set to True to ensure a contiguous memory block. The captured 
frames are then copied to a new list, piciter2_, to address any buffer-related considerations.

By leveraging ctypes-based screen capturing through FFmpegshot, you can benefit from its 
functionality while exploring different capturing methods. This provides additional 
flexibility and options for your screen capture requirements.

```python
with FFmpegshot() as sc:
    piciter_ = sc.capture_all_screens_ctypes(ascontiguousarray=True)
    piciter2_ = []
    for ini, pi in enumerate(piciter_):
        piciter2_.append(pi.copy())
        if ini == 100:
            break
```


## Multiprocessing - capture various windows, screens or areas

[Check out the result - Video](https://github.com/hansalemaos/ffmpeg_screenshot_pipe/raw/main/examples/multicapture/capture_one_screen_ddagrab.mp4)


An example of how to record various screens separately.
It captures screenshots using the ddagrab method.
While there may be a slight delay in capturing the screenshots, 
it's important to note that I was simultaneously recording
two streams, one using Python and the other using OBS.

Here are some important points to consider:

The function works similar to the normal screenshot functions but requires you 
to create dictionaries for each function call.


When using multiprocessing, you can access the latest captured screenshots using procresults.results[index][-1].
The variable "prore" represents the numerical index (e.g., 0 and 1), and [-1] retrieves the newest screenshot.

To enable multiprocessing and handle the results, you can import the necessary modules:
from ffmpeg_screenshot_pipe.ffmpeg_screenshot_pipe_multi import runasync, procresults

The module procresults provides additional settings to control the process:
- procresults.dequebuffer determines the maximum number of screenshots saved in procresults.results[index].

- procresults.asyncsleep specifies the sleep duration (in seconds) between asynchronous function calls.
Setting it to zero can cause the program to freeze.

- procresults.sleeptimeafterkill sets the time (in seconds) to wait after killing the FFmpeg process
before calling sys.exit() to terminate the program gracefully.

- procresults.stop_flag is a Value object that, when set to True, terminates all processes.
It's important to allow sufficient time (e.g., 5 seconds) for the program to exit gracefully to avoid
lingering ffmpeg.exe zombie processes.

The function procresults.sleeptimebeforekill determines the time interval before forcefully 
terminating the processes.
By default, it uses a lambda function to calculate 90% of procresults.sleeptimeafterkill.

It's worth noting that all functions ending with _gdigrab or _ddagrab can be used in multiprocessing.
If you have a weaker CPU (e.g., Intel 5) but a powerful GPU (e.g., GeForce RTX 2060 SUPER), it's recommended
to use the _ddagrab functions for better performance.

### Whole example 

```python

import cv2
import kthread
from time import strftime
from ffmpeg_screenshot_pipe.ffmpeg_screenshot_pipe_multi import runasync, procresults

procresults.dequebuffer = 24
procresults.asyncsleep = 0.001
procresults.sleeptimeafterkill = 5
procresults.sleeptimebeforekill = lambda: procresults.sleeptimeafterkill * 0.9
activethreads = []

timest = lambda: strftime("%Y_%m_%d_%H_%M_%S")


def start_display_thread():
    activethreads.append(kthread.KThread(target=display_thread, name="cv2function"))
    activethreads[-1].daemon = True
    activethreads[-1].start()


def display_thread(stop_at_frame=None):
    co = 0
    goback = False
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    while True:
        try:
            keys = list(procresults.results.keys())
            for prore in keys:
                try:
                    cv2.imshow(str(prore), procresults.results[prore][-1]) # this is how you have to access the results
                except Exception:
                    pass
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.waitKey(0)
                    goback = True
                    break
                if stop_at_frame:
                    co = co + 1
                    if co > stop_at_frame:
                        cv2.destroyAllWindows()
                        goback = True
                    break
            if goback:
                break
        except Exception as e:
            print(e)
            continue

    cv2.destroyAllWindows()
    procresults.stopfunction()


def kill_active_threads():
    for th in activethreads:
        if th.is_alive():
            th.kill()


def run_parallel(dictlist):
    runasync(dictlist)
    kill_active_threads()


if __name__ == "__main__": # don't forget that when using multiprocessing!!
	
	# call the function that is doing something with the screenshots (in this case just showing) 
	# as a thread before you run runasync. 
    start_display_thread()
	
	
    dict0 = dict(
        function="capture_one_screen_ddagrab",
        monitor_index=0,
        frames=25,
        draw_mouse=True,
    )

    dict1 = dict(
        function="capture_one_screen_ddagrab",
        monitor_index=1,
        frames=25,
        draw_mouse=True,
    )
    run_parallel([dict0, dict1])


```
### More examples with multiprocessing

[Check out the result - Video - Capturing 3 Notepad Windows at the same time ](https://github.com/hansalemaos/ffmpeg_screenshot_pipe/raw/main/examples/multicapture/capturing_3_notepad_windows.mp4)

[Check out the result - Video - Capturing 3 areas simultaneously ](https://github.com/hansalemaos/ffmpeg_screenshot_pipe/raw/main/examples/multicapture/capture_3_boxes.mp4)

```python
kill_active_threads()  # killing the function start_display_thread
dict2 = dict(
	function="capture_box_ddagrab",
	rectangle=(100, 100, 500, 500),
	frames=30,
	draw_mouse=True,
)
dict3 = dict(
	function="capture_box_ddagrab",
	rectangle=(500, 500, 1000, 1000),
	frames=30,
	draw_mouse=True,
)
dict4 = dict(
	function="capture_box_ddagrab",
	rectangle=(2011, 900, 3400, 1070),
	frames=30,
	draw_mouse=True,
)

dict5 = dict(
	function="capture_window_ddagrab",
	searchdict={
		"hwnd": 328832,
		"status": "visible",
		"path_re": ".*notepad.exe.*",
	},
	frames=30,
	draw_mouse=True,
	allowed_screens=(0,),
	return_copy=True,
)

# "capturing_3_notepad_windows.mp4"
dict6 = dict(
	function="capture_window_ddagrab",
	searchdict={
		"hwnd": 591080,
		"status": "visible",
		"path_re": ".*notepad.exe.*",
	},
	frames=30,
	draw_mouse=True,
	allowed_screens=(0,),
	return_copy=True,
)
dict7 = dict(
	function="capture_window_ddagrab",
	searchdict={
		"hwnd": 67488,
		"status": "visible",
		"path_re": ".*notepad.exe.*",
	},
	frames=30,
	draw_mouse=True,
	allowed_screens=(0,),
	return_copy=True,
)
```

