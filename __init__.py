from appshwnd import (
    find_window_and_make_best_window_unique,
)
import re
import subprocess
from collections import defaultdict
from ctypes import LibraryLoader, WinDLL, wintypes
import kthread
from time import sleep as sleep_, time
from getfilenuitkapython import get_filepath
from subprocesskiller import kill_process_children_parents
from time import perf_counter
from subprocess_alive import is_process_alive
import ctypes
import os
import sys
import numpy as np
from ctypes.wintypes import HWND
from getmonitorresolution import get_monitors_resolution
from fastimgconcat import temparrays, fastconcat_horizontal, fastconcat_vertical
from threading import Timer
from get_rectangle_infos import get_rectangle_information

from fast_ctypes_screenshots import (
    ScreenshotOfWindow,
    ScreenshotOfRegion,
    ScreenshotOfOneMonitor,
    ScreenshotOfAllMonitors,
)

windll = LibraryLoader(WinDLL)
user32 = windll.user32
# make variables globally available without using "global"
conf = sys.modules[__name__]
conf.pro = defaultdict(list)
# RGB colors
conf.buffmultiply = 3

# no new window for subprocess
startupinfo = subprocess.STARTUPINFO()
creationflags = 0 | subprocess.CREATE_NO_WINDOW
startupinfo.wShowWindow = subprocess.SW_HIDE
invisibledict = {
    "startupinfo": startupinfo,
    "creationflags": creationflags,
}

# get the monitor data right at the beginning
allmoni, gera = get_monitors_resolution()
monwidth = gera["width_all_monitors"]  # width of all monitors together
monheight = gera["height_all_monitors"]  # height of all monitors together
max_monitor_width = gera["max_monitor_width"]
min_monitor_width = gera["min_monitor_width"]
max_monitor_height = gera["max_monitor_height"]
min_monitor_height = gera["min_monitor_height"]

# ffmpeg.exe is located in the modules folder
ffmpegpath_ = os.path.normpath(os.path.join(os.path.dirname(__file__), "ffmpeg.exe"))
if not os.path.exists(ffmpegpath_):
    ffmpegpath_ = get_filepath("ffmpeg.exe")


def kill_ffmpeg(ffmprocproc: subprocess.Popen, t: kthread.KThread) -> None:
    try:
        # let's try to terminate it gracefully
        try:
            ffmprocproc.stdout.close()
        except Exception:
            pass
        try:
            ffmprocproc.stdin.close()
        except Exception:
            pass
        try:
            ffmprocproc.stderr.close()
        except Exception:
            pass
        try:
            ffmprocproc.wait(timeout=0.0001)
        except Exception:
            pass
        try:
            ffmprocproc.terminate()
        except Exception:
            pass
    except Exception:
        pass

    try:
        try:
            # if running threads are still open
            if t.is_alive():
                try:
                    t.kill()
                except Exception:
                    pass
        except Exception:
            pass
        if is_process_alive(ffmprocproc.pid):
            # if it got stuck, we kill it forcefully
            try:
                kill_process_children_parents(
                    pid=ffmprocproc.pid,
                    max_parent_exe="ffmpeg.exe",
                    dontkill=(
                        "Caption",
                        "conhost.exe",
                    ),  # killing conhost.exe is not a good idea
                )
            except Exception:
                pass
    except Exception:
        pass


def get_window_size(hwnd):
    rect = ctypes.wintypes.RECT()

    user32.GetClientRect(hwnd, ctypes.byref(rect))
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    return [rect.left, rect.top, rect.right, rect.bottom, width, height]


# function for capturing the desktop
def win_api_search4(
    myid,
    width=None,
    height=None,
    updatedwindowtext=None,
    ffmpegpath=r"ffmpeg.exe",
    frames=60,
    draw_mouse=True,
    flags=(),
    debug=False,
    revertnamefunction=None,
    change_titel_back_after=3,
    *args,
    **kwargs,
):
    debu = (
        "-v",
        "verbose",
    )
    if not debug:  # disables all unnecessary output
        debu = (
            "-loglevel",
            "-8",
            "-hide_banner",
            "-nostats",
        )

    if updatedwindowtext:
        window2capture = f"title={updatedwindowtext}"
    else:
        window2capture = "desktop"

    offsetlist = []
    if "offset_x" in kwargs:
        offsetlist.extend(
            [
                "-offset_x",
                f"{kwargs['offset_x']}",
            ]
        )
    if "offset_y" in kwargs:
        offsetlist.extend(
            [
                "-offset_y",
                f"{kwargs['offset_y']}",
            ]
        )
    ffmpeg_cmd = [
        ffmpegpath,
        "-y",
        *debu,
        "-nostdin",
        "-draw_mouse",
        str(int(draw_mouse)),
        "-probesize",  # probesize / analyzeduration makes capturing fast
        "32",
        "-analyzeduration",
        "0",
        "-framerate",  # it is not possible to use "-r" -> too slow
        f"{frames}",
        "-f",
        "gdigrab",
        "-framerate",
        f"{frames}",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-video_size",
        f"{width}x{height}",
        *offsetlist,
        "-i",
        window2capture,
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        "format=bgr24",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-an",
        "-sn",
        "-dn",
        *flags,
        "-",
    ]
    if window2capture != "desktop":
        if isinstance(change_titel_back_after, int):
            if change_titel_back_after > 0:
                tim = Timer(change_titel_back_after, revertnamefunction)
                tim.daemon = True
                tim.start()

    stderr = subprocess.DEVNULL
    d2add = invisibledict
    start_new_session = True
    if debug:
        stderr = subprocess.PIPE
        d2add = {}
    conf.pro[myid].append(
        subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            bufsize=height * width * conf.buffmultiply,
            **d2add,
            start_new_session=start_new_session,
        )
    )

    if debug:
        try:
            for data in iter(conf.pro[myid][0].stderr.readline, b""):
                try:
                    print(f"------------{data}")
                except Exception as fe:
                    print(fe)
        except Exception:
            pass


# function for capturing a window
def win_api_search1(
    myid,
    width=None,
    height=None,
    updatedwindowtext=None,
    ffmpegpath=r"ffmpeg.exe",
    frames=60,
    draw_mouse=True,
    flags=(),
    debug=False,
    revertnamefunction=None,
    change_titel_back_after=3,
    *args,
    **kwargs,
):
    debu = (
        "-v",
        "verbose",
    )
    if not debug:
        debu = (
            "-loglevel",
            "-8",
            "-hide_banner",
            "-nostats",
        )

    window2capture = f"title={updatedwindowtext}"

    offsetlist = []
    if "offset_x" in kwargs:
        offsetlist.extend(
            [
                "-offset_x",
                f"{kwargs['offset_x']}",
            ]
        )
    if "offset_y" in kwargs:
        offsetlist.extend(
            [
                "-offset_y",
                f"{kwargs['offset_y']}",
            ]
        )
    ffmpeg_cmd = [
        ffmpegpath,
        "-y",
        *debu,
        "-nostdin",
        "-draw_mouse",
        str(int(draw_mouse)),
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-framerate",
        f"{frames}",
        "-f",
        "gdigrab",
        "-framerate",
        f"{frames}",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        *offsetlist,
        "-i",
        window2capture,
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        "format=bgr24",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-an",
        "-sn",
        "-dn",
        *flags,
        "-",
    ]
    # restore the original window title if it has been changed
    if window2capture != "desktop":
        if isinstance(change_titel_back_after, int):
            if change_titel_back_after > 0:
                tim = Timer(change_titel_back_after, revertnamefunction)
                tim.daemon = True
                tim.start()

    stderr = subprocess.DEVNULL
    d2add = invisibledict
    start_new_session = True
    if debug:
        stderr = subprocess.PIPE
        d2add = {}
    conf.pro[myid].append(
        subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            bufsize=height * width * conf.buffmultiply,
            **d2add,
            start_new_session=start_new_session,
        )
    )

    if debug:
        try:
            for data in iter(conf.pro[myid][0].stderr.readline, b""):
                try:
                    print(f"------------{data}")
                except Exception as fe:
                    print(fe)
        except Exception:
            pass


def win_api_search6(
    myid,
    ffmpegpath=r"ffmpeg.exe",
    IMG_H=allmoni[0]["height"],
    IMG_W=allmoni[0]["width"],
    frames=60,
    offset_x=0,
    offset_y=0,
    draw_mouse=True,
    flags=(),
    debug=False,
    monitor=0,
    *args,
    **kwargs,
):
    fmmp = os.path.normpath(ffmpegpath)
    debu = (
        "-v",
        "verbose",
    )
    if not debug:
        debu = (
            "-loglevel",
            "-8",
            "-hide_banner",
            "-nostats",
        )

    ffmpeg_cmd = [
        fmmp,
        *debu,
        "-y",
        *flags,
        "-nostdin",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-filter_complex",
        f"ddagrab=output_idx={monitor}:draw_mouse={str(int(draw_mouse))}:framerate={frames}:video_size={IMG_W - offset_x}x{IMG_H - offset_y}:offset_x={offset_x}:offset_y={offset_y},hwdownload,format=bgra,format=bgr24",
        "-f",
        "rawvideo",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-",
    ]

    stderr = subprocess.DEVNULL
    d2add = invisibledict
    start_new_session = True
    if debug:
        stderr = subprocess.PIPE
        d2add = {}
    conf.pro[myid].append(
        subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            bufsize=IMG_H * IMG_W * conf.buffmultiply,
            **d2add,
            start_new_session=start_new_session,
        )
    )
    if debug:
        try:
            for data in iter(conf.pro[myid][0].stderr.readline, b""):
                try:
                    print(f"------------{data}")
                except Exception:
                    pass
        except Exception:
            pass


def win_api_search7(
    myid,
    ffmpegpath=r"ffmpeg.exe",
    monitorheight=0,
    monitorwidth=0,
    IMG_H=0,
    IMG_W=0,
    frames=60,
    offset_x=0,
    offset_y=0,
    draw_mouse=True,
    flags=(),
    debug=False,
    monitor=0,
    *args,
    **kwargs,
):
    debu = (
        "-v",
        "verbose",
    )
    if not debug:
        debu = (
            "-loglevel",
            "-8",
            "-hide_banner",
            "-nostats",
        )

    ffmpeg_cmd = [
        ffmpegpath,
        *debu,
        "-y",
        *flags,
        "-nostdin",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-filter_complex",
        f"ddagrab=output_idx={monitor}:draw_mouse={str(int(draw_mouse))}:framerate={frames}:video_size={monitorheight}x{monitorwidth},hwdownload,format=bgra,format=bgr24,crop={IMG_W}:{IMG_H}:{offset_x}:{offset_y}",
        "-f",
        "rawvideo",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-",
    ]

    stderr = subprocess.DEVNULL
    d2add = invisibledict
    start_new_session = True
    if debug:
        stderr = subprocess.PIPE
        d2add = {}
    conf.pro[myid].append(
        subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            bufsize=IMG_W * IMG_H * conf.buffmultiply,
            **d2add,
            start_new_session=start_new_session,
        )
    )
    if debug:
        try:
            for data in iter(conf.pro[myid][0].stderr.readline, b""):
                try:
                    print(f"------------{data}")
                except Exception:
                    pass
        except Exception:
            pass


def adjust_array_size_horizontal() -> None:
    allshapes = monwidth
    allshapesmax = max_monitor_height
    shapeneeded = allshapesmax, allshapes, 3
    if (temparrays.horizontal.shape != shapeneeded) or (
        temparrays.horizontal.dtype.name != np.uint8
    ):
        temparrays.horizontal = np.zeros(shapeneeded, dtype=np.uint8)


def adjust_array_size_vertical() -> None:
    allshapes = monheight
    allshapesmax = max_monitor_width
    shapeneeded = allshapes, allshapesmax, 3
    if (temparrays.vertical.shape != shapeneeded) or (
        temparrays.vertical.dtype.name != np.uint8
    ):
        temparrays.vertical = np.zeros(shapeneeded, dtype=np.uint8)


def intersects(box1, box2):
    return not (
        box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1]
    )


def cropimage(img, coords):
    return img[coords[1] : coords[3], coords[0] : coords[2]]


class FFmpegshot:
    def __init__(self, ffmpegpath: str = ffmpegpath_, ffmpeg_param: tuple = ()):
        if isinstance(ffmpeg_param, tuple):
            self.ffmpeg_param = list(ffmpeg_param)
        else:
            self.ffmpeg_param = ffmpeg_param
        self.t = None
        self.ts = []
        self.playvideo = True
        self.myid = str(perf_counter())
        self.myids = []
        self.coords_moni = []
        self.ffmpegpath = ffmpegpath
        self.additional_instances = []
        self.frame_checker_enabled = False

    def __enter__(self):
        self.playvideo = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.playvideo = False
        try:
            self.kill_ffmpeg()
        except Exception as fa:
            pass

    def kill_ffmpeg(self):
        r"""Terminate all running FFmpeg instances and clean up resources.

        This method stops all running FFmpeg processes associated with the current FFmpegshot instance
        and releases any allocated resources. It is called internally during the cleanup process or
        can be explicitly invoked by the user.

        Note:
            - Killing FFmpeg processes may result in loss of unsaved data.
            - This method does not affect the state of the FFmpegshot instance.

        Raises:
            Exception: If an error occurs while terminating FFmpeg processes.

        """
        self.playvideo = False

        for mi, thr in zip(self.myids, self.ts):
            sleep(1 / 60)

            if conf.pro[mi]:
                kill_ffmpeg(conf.pro[mi][0], thr)
                conf.pro[mi].clear()

        if conf.pro[self.myid]:
            kill_ffmpeg(conf.pro[self.myid][0], self.t)
            conf.pro[self.myid].clear()
        for i in self.additional_instances:
            sleep(1 / 60)

            i.kill_ffmpeg()
        self.myids.clear()
        self.coords_moni.clear()
        self.additional_instances.clear()
        self.ts.clear()
        self.t = None
        self.playvideo = True

    @staticmethod
    def capture_window_client_ctypes(hwnd: int, ascontiguousarray=False):
        """Capture the contents of a specific window using the ctypes module.

        This method captures the contents of a specific window identified by its handle using the ctypes module.
        The captured frames are yielded as a generator.

        Args:
            hwnd (int): The handle of the window to capture.
            ascontiguousarray (bool): Whether to return the frames as contiguous arrays (default: False).

        Yields:
            numpy.ndarray: The captured frames as numpy arrays.

        """
        with ScreenshotOfWindow(
            hwnd=hwnd, client=True, ascontiguousarray=ascontiguousarray
        ) as screenshots_window:
            yield from screenshots_window

    @staticmethod
    def capture_window_window_ctypes(hwnd: int, ascontiguousarray=False):
        """Capture the contents of a specific window using the ctypes module.

        This method captures the contents of a specific window identified by its handle using the ctypes module.
        The captured frames are yielded as a generator.

        Args:
            hwnd (int): The handle of the window to capture.
            ascontiguousarray (bool): Whether to return the frames as contiguous arrays (default: False).

        Yields:
            numpy.ndarray: The captured frames as numpy arrays.

        """
        with ScreenshotOfWindow(
            hwnd=hwnd, client=False, ascontiguousarray=ascontiguousarray
        ) as screenshots_window:
            yield from screenshots_window

    def capture_window_ddagrab(
        self,
        searchdict: dict,
        frames: int = 30,
        draw_mouse: bool = True,
        client_or_window: str = "window",  # "client" or "window"
        quant_screenshots: int | None = None,
        debug: bool = False,
        allowed_screens: tuple = (),
        return_copy: bool = False,
        *args,
        **kwargs,
    ):
        r"""Capture the contents of a specific window using DDAGrab.

        This method captures the contents of a specified window using the DDAGrab library.

        Args:
            searchdict (dict): A dictionary containing search parameters for the target window, all details here: https://github.com/hansalemaos/appshwnd
            frames (int): frames per second (default: 30).
            draw_mouse (bool): Whether to draw the mouse cursor in the captured frames (default: True).
            client_or_window (str): Specifies whether to capture a client or window (default: "window").
                                    Valid values: "client", "window".
            quant_screenshots (int | None): The number of screenshots to take (default: None).
                                            If None, a single screenshot is taken.
            debug (bool): Whether to enable debug mode (default: False).
            allowed_screens (tuple): A tuple of allowed screen IDs to capture (default: empty tuple - all are allowed).
            return_copy (bool): Whether to return a copy of the captured frames (default: False).
            *args: Variable-length arguments.
            **kwargs: Keyword arguments.

        Raises:
            Exception: If an error occurs during the window capture process.

        Yields:
            np.ndarray: the screenshot as a numpy array.

        """
        rect = ctypes.wintypes.RECT()
        window_rect = ctypes.wintypes.RECT()
        client_rect = ctypes.wintypes.RECT()

        def get_window_size_for_window(hwnd):
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            result = [rect.left, rect.top, rect.right, rect.bottom, width, height]
            return result

        def get_window_size_for_client(hwnd):
            user32.GetWindowRect(hwnd, ctypes.byref(window_rect))

            user32.GetClientRect(hwnd, ctypes.byref(client_rect))

            poi = ctypes.wintypes.POINT(window_rect.left, window_rect.top)

            user32.ScreenToClient(hwnd, ctypes.byref(poi))
            client_left, client_top = poi.x, poi.y
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            width = client_rect.right - client_rect.left
            height = client_rect.bottom - client_rect.top
            result = [
                rect.left - client_left,
                rect.top - client_top,
                rect.left - client_left + width,
                rect.top - client_top + height,
                width,
                height,
            ]
            return result

        if client_or_window == "window":
            get_window_size_ = get_window_size_for_window
        else:
            get_window_size_ = get_window_size_for_client

        (
            bestwindows,
            bestwindow,
            hwnd,
            startwindowtext,
            updatedwindowtext,
            revertnamefunction,
        ) = find_window_and_make_best_window_unique(
            searchdict, timeout=5, make_unique=False, flags=re.I
        )
        if not allowed_screens:
            allowed_screens = tuple(allmoni.keys())
        hwndctypes = HWND(hwnd)
        if not quant_screenshots:
            quant_screenshots = 4_000_000_000
        self.playvideo = True
        monicounter_startx = 0
        monicounter_starty = 0
        monicounter_endx = 0
        allmythreads = []
        for monitornumber, monidata in allmoni.items():
            if not self.frame_checker_enabled:
                sleep(1 / frames)

            multiid = f"{self.myid}_____{monitornumber}"
            self.myids.append(multiid)
            monicounter_endx = monicounter_endx + max_monitor_width
            monicounter_endy = monidata["height"]
            self.coords_moni.append(
                [
                    monicounter_startx,
                    monicounter_starty,
                    monicounter_endx,
                    monicounter_endy,
                ]
            )
            monicounter_startx = monicounter_startx + max_monitor_width
            if monitornumber not in allowed_screens:
                allmythreads.append(None)
            else:
                allmythreads.append(monitornumber)

        allmythreadsnotnone = [x for x in allmythreads if isinstance(x, int)]
        if len(allmythreadsnotnone) == 1:
            piciter = self.capture_one_screen_ddagrab(
                monitor_index=allmythreadsnotnone[0],
                frames=frames,
                draw_mouse=draw_mouse,
                quant_screenshots=quant_screenshots,
                debug=debug,
            )
        else:
            piciter = self.capture_all_screens_ddagrab(
                frames=frames,
                draw_mouse=draw_mouse,
                quant_screenshots=quant_screenshots,
                debug=debug,
                vertical_or_horizontal="horizontal",
                return_empty=True,
            )
        coun = 0
        for picture in piciter:
            if not self.frame_checker_enabled:
                sleep(1 / frames)

            if coun >= quant_screenshots:
                self.playvideo = False
                break
            if not self.playvideo:
                break
            allnewcoords = get_window_size_(hwndctypes)
            coords = allnewcoords[:4]
            checkcoords = []
            isthere = True
            allsubstractx = 0
            if debug:
                print(f"{allnewcoords=}")
            for id_, sq_ in zip(self.myids, self.coords_moni):
                if int(id_[-1]) in allowed_screens and intersects(sq_, coords):
                    checkcoords.append(id_)
                    isthere = False

                else:
                    if isthere:
                        allsubstractx = allsubstractx + sq_[2] - sq_[0]

            if not checkcoords:
                continue
            coords2 = (
                coords[0] - allsubstractx,
                coords[1],
                coords[2] - allsubstractx,
                coords[3],
            )
            coords = [x if x >= 0 else 0 for x in coords2]
            coords[0] = coords[0] if coords[0] < monicounter_endx else monicounter_endx
            coords[1] = (
                coords[1] if coords[1] < max_monitor_height else max_monitor_height
            )
            coords[2] = coords[2] if coords[2] < monicounter_endx else monicounter_endx
            coords[3] = (
                coords[3] if coords[3] < max_monitor_height else max_monitor_height
            )

            if len(picture.shape) == 3:
                yield cropimage(picture, coords)
            else:
                if return_copy:
                    yield cropimage(temparrays.horizontal, coords).copy()
                else:
                    yield cropimage(temparrays.horizontal, coords)
            coun = coun + 1
        # self.kill_ffmpeg()

    @staticmethod
    def capture_box_ctypes(
        rectangle: tuple[int, int, int, int], ascontiguousarray=False
    ):
        r"""Capture the contents of a specific rectangular region on the screen using the ctypes module.

        This method captures the contents of a specific rectangular region on the screen using the ctypes module.
        The captured frames are yielded as a generator.

        Args:
            rectangle (tuple): The coordinates of the rectangular region (x0, y0, x1, y1).
            ascontiguousarray (bool): Whether to return the frames as contiguous arrays (default: False).

        Yields:
            numpy.ndarray: The captured frames as numpy arrays.

        """
        x0, y0, x1, y1 = rectangle
        with ScreenshotOfRegion(
            x0=x0, y0=y0, x1=x1, y1=y1, ascontiguousarray=ascontiguousarray
        ) as screenshots_region:
            yield from screenshots_region

    def capture_box_ddagrab(
        self,
        rectangle: tuple[int, int, int, int],
        frames: int = 60,
        draw_mouse: bool = True,
        quant_screenshots: int | None = None,
        return_bytes: bool = False,
        debug: bool = False,
    ):
        r"""Capture the contents of a specific rectangular area on the screen using DDAGrab.

        This method captures the contents of a specified rectangular area on the screen using the DDAGrab library.

        Args:
            rectangle (tuple[int, int, int, int]): A tuple specifying the coordinates of the rectangle in the format
                                                   (left, top, right, bottom).
            frames (int): fps (default: 60).
            draw_mouse (bool): Whether to draw the mouse cursor in the captured frames (default: True).
            quant_screenshots (int | None): The number of screenshots to take (default: None).
                                            If None, a single screenshot is taken.
            return_bytes (bool): Whether to return the captured frames as bytes (default: False).
            debug (bool): Whether to enable debug mode (default: False).

        Raises:
            Exception: If an error occurs during the screen capture process.


        Yields:
            tuple or numpy.ndarray: A tuple representing the size of the captured frame (if `return_bytes` is True),
                                    or a numpy.ndarray representing the captured frame (if `return_bytes` is False).

        """
        rectangleinfo = get_rectangle_information(rect=rectangle)
        startx = rectangleinfo.format_1x4[0]
        starty = rectangleinfo.format_1x4[1]
        IMG_W = rectangleinfo.height
        IMG_H = rectangleinfo.width
        if not quant_screenshots:
            quant_screenshots = 4_000_000_000
        self.playvideo = True

        self.playvideo = True
        monioffsetx = 0
        monioffsety = 0
        monitor = 0
        monitor_height = allmoni[monitor]["height"]
        monitor_width = allmoni[monitor]["width"]

        for kk, ii in allmoni.items():
            monitor_height = allmoni[kk]["height"]
            monitor_width = allmoni[kk]["width"]
            moninfos = (
                monioffsetx,
                monioffsety,
                monitor_width + monioffsetx,
                monitor_height + monioffsety,
            )

            if intersects(moninfos, rectangleinfo.format_1x4):
                monitor = kk

                break
            else:
                monioffsetx = monioffsetx + monitor_width
        startx = startx - monioffsetx
        if IMG_H > monitor_width:
            IMG_H = monitor_width - startx
        if IMG_W > monitor_height:
            IMG_W = monitor_height - starty

        allargs = (
            self.myid,
            self.ffmpegpath,
            monitor_width,
            monitor_height,
            IMG_W,
            IMG_H,
            frames,
            startx,
            starty,
            draw_mouse,
            self.ffmpeg_param,
            debug,
            monitor,
        )
        self.ts.append(
            kthread.KThread(
                target=win_api_search7,
                args=allargs,
                name=self.myid,
            )
        )
        coun = 0
        self.ts[-1].start()
        while self.playvideo:
            if not self.frame_checker_enabled:
                sleep(1 / frames)

            try:
                if coun >= quant_screenshots:
                    self.playvideo = False
                    break
                if return_bytes:
                    yield (IMG_W, IMG_H, 3), conf.pro[self.myid][0].stdout.read(
                        IMG_W * IMG_H * 3
                    )
                else:
                    yield np.frombuffer(
                        conf.pro[self.myid][0].stdout.read(IMG_W * IMG_H * 3),
                        dtype="uint8",
                    ).reshape((IMG_W, IMG_H, 3))

            except Exception as fe:
                continue

    def capture_box_gdigrab(
        self,
        rectangle: tuple[int, int, int, int],
        frames: int = 60,
        draw_mouse: bool = True,
        quant_screenshots: int | None = None,
        return_bytes: bool = False,
        debug: bool = False,
    ):
        """Capture the contents of a specific rectangular area on the screen using GDIGrab.

        This method captures the contents of a specified rectangular area on the screen using the GDIGrab library.

        Args:
            rectangle (tuple[int, int, int, int]): A tuple specifying the coordinates of the rectangle in the format
                                                   (left, top, right, bottom).
            frames (int): fps (default: 60).
            draw_mouse (bool): Whether to draw the mouse cursor in the captured frames (default: True).
            quant_screenshots (int | None): The number of screenshots to take (default: None).
                                            If None, a single screenshot is taken.
            return_bytes (bool): Whether to return the captured frames as bytes (default: False).
            debug (bool): Whether to enable debug mode (default: False).

        Raises:
            Exception: If an error occurs during the screen capture process.

        Yields:
            tuple or numpy.ndarray: A tuple representing the size of the captured frame (if `return_bytes` is True),
                                    or a numpy.ndarray representing the captured frame (if `return_bytes` is False).

        """
        rectangleinfo = get_rectangle_information(rect=rectangle)
        startx = rectangleinfo.format_1x4[0]
        starty = rectangleinfo.format_1x4[1]
        IMG_W = rectangleinfo.height
        IMG_H = rectangleinfo.width
        if not quant_screenshots:
            quant_screenshots = 4_000_000_000
        self.playvideo = True
        self.ts.append(
            kthread.KThread(
                target=win_api_search4,
                args=(
                    self.myid,
                    IMG_H,
                    IMG_W,
                    None,
                    self.ffmpegpath,
                    frames,
                    draw_mouse,
                    self.ffmpeg_param,
                    debug,
                    None,
                    None,
                ),
                kwargs={"offset_x": startx, "offset_y": starty},
                name=self.myid,
            )
        )
        coun = 0
        self.ts[-1].start()
        while self.playvideo:
            if not self.frame_checker_enabled:
                sleep(1 / frames)

            try:
                if coun >= quant_screenshots:
                    self.playvideo = False
                    break
                if return_bytes:
                    yield (IMG_W, IMG_H, 3), conf.pro[self.myid][0].stdout.read(
                        IMG_W * IMG_H * 3
                    )
                else:
                    yield np.frombuffer(
                        conf.pro[self.myid][0].stdout.read(IMG_W * IMG_H * 3),
                        dtype="uint8",
                    ).reshape((IMG_W, IMG_H, 3))

            except Exception as fe:
                continue

    def capture_window_gdigrab(
        self,
        searchdict: dict,
        frames: int = 60,
        draw_mouse: bool = True,
        quant_screenshots: int | None = None,
        return_bytes: bool = False,
        debug: bool = False,
    ):
        """Capture the contents of a specific rectangular area on the screen using GDIGrab.

        This method captures the contents of a specified rectangular area on the screen using the GDIGrab library.

        Args:
    "        rectangle (tuple[int, int, int, int]): A tuple specifying the coordinates of the rectangle in the format
                                                   (left, top, right, bottom).
            frames (int): fps (default: 60).
            draw_mouse (bool): Whether to draw the mouse cursor in the captured frames (default: True).
            quant_screenshots (int | None): The number of screenshots to take (default: None).
                                            If None, a single screenshot is taken.
            return_bytes (bool): Whether to return the captured frames as bytes (default: False).
            debug (bool): Whether to enable debug mode (default: False)."

        Raises:
            Exception: If an error occurs during the screen capture process.

        Yields:
            tuple or numpy.ndarray: A tuple representing the size of the captured frame (if `return_bytes` is True),
                                    or a numpy.ndarray representing the captured frame (if `return_bytes` is False).

        """
        (
            bestwindows,
            bestwindow,
            hwnd,
            startwindowtext,
            updatedwindowtext,
            revertnamefunction,
        ) = find_window_and_make_best_window_unique(
            searchdict, timeout=5, make_unique=True, flags=re.I
        )
        hwndctypes = HWND(hwnd)

        allnewcoords = get_window_size(hwndctypes)
        IMG_W = allnewcoords[-2]
        IMG_H = allnewcoords[-1]
        if not quant_screenshots:
            quant_screenshots = 4_000_000_000
        self.playvideo = True

        self.ts.append(
            kthread.KThread(
                target=win_api_search1,
                args=(
                    self.myid,
                    IMG_W,
                    IMG_H,
                    updatedwindowtext,
                    self.ffmpegpath,
                    frames,
                    draw_mouse,
                    self.ffmpeg_param,
                    debug,
                    revertnamefunction,
                    3,
                ),
                name=self.myid,
            )
        )
        coun = 0
        self.ts[-1].start()
        while self.playvideo:
            if not self.frame_checker_enabled:
                sleep(1 / frames)

            try:
                if coun >= quant_screenshots:
                    self.playvideo = False
                    break
                if return_bytes:
                    yield (IMG_W, IMG_H, 3), conf.pro[self.myid][0].stdout.read(
                        IMG_H * IMG_W * 3
                    )
                else:
                    yield np.frombuffer(
                        conf.pro[self.myid][0].stdout.read(IMG_W * IMG_H * 3),
                        dtype="uint8",
                    ).reshape((IMG_H, IMG_W, 3))

            except Exception as fe:
                continue

    @staticmethod
    def capture_one_screen_ctypes(monitor=0, ascontiguousarray=False):
        r"""Capture the contents of a single monitor using the ctypes module.

           This method captures the contents of a single monitor using the ctypes module.
           The captured frames are yielded as a generator.

           Args:
               monitor (int): The index of the monitor to capture (default: 0).
               ascontiguousarray (bool): Whether to return the frames as contiguous arrays (default: False).

           Yields:
               numpy.ndarray: The captured frames as numpy arrays."""
        with ScreenshotOfOneMonitor(
            monitor=monitor, ascontiguousarray=ascontiguousarray
        ) as screenshots_monitor:
            yield from screenshots_monitor

    def capture_one_screen_gdigrab(
        self,
        monitor_index: int = 0,
        frames: int = 60,
        draw_mouse: bool = True,
        quant_screenshots: int | None = None,
        return_bytes: bool = False,
        debug: bool = False,
    ):
        """Capture the contents of a single screen using GDIGrab.

        This method captures the contents of a single screen identified by the monitor_index using the GDIGrab library.

        Args:
            monitor_index (int): The index of the monitor to capture (default: 0).
            frames (int): The number of frames to capture per second (default: 60).
            draw_mouse (bool): Whether to draw the mouse cursor in the captured frames (default: True).
            quant_screenshots (int | None): The number of screenshots to take (default: None).
                                            If None, a large number (4,000,000,000) is used as the default.
            return_bytes (bool): Whether to return the captured frames as bytes (default: False).
            debug (bool): Whether to enable debug mode (default: False).

        Raises:
            Exception: If an error occurs during the screen capture process.

        Yields:
            tuple or numpy.ndarray: A tuple representing the size of the captured frame (if `return_bytes` is True),
                                    or a numpy.ndarray representing the captured frame (if `return_bytes` is False).

        """
        offset_x = 0
        IMG_W = allmoni[monitor_index]["height"]
        IMG_H = allmoni[monitor_index]["width"]
        if not quant_screenshots:
            quant_screenshots = 4_000_000_000
        self.playvideo = True
        for kk, ii in allmoni.items():
            if kk == monitor_index:
                break
            else:
                offset_x = offset_x + ii["width"]

        self.ts.append(
            kthread.KThread(
                target=win_api_search4,
                args=(
                    self.myid,
                    IMG_H,
                    IMG_W,
                    None,
                    self.ffmpegpath,
                    frames,
                    draw_mouse,
                    self.ffmpeg_param,
                    debug,
                    None,
                    None,
                ),
                kwargs={"offset_x": offset_x, "offset_y": 0},
                name=self.myid,
            )
        )
        coun = 0
        self.ts[-1].start()
        while self.playvideo:
            if not self.frame_checker_enabled:
                sleep(1 / frames)

            try:
                if coun >= quant_screenshots:
                    self.playvideo = False
                    break
                if return_bytes:
                    yield (IMG_W, IMG_H, 3), conf.pro[self.myid][0].stdout.read(
                        IMG_W * IMG_H * 3
                    )
                else:
                    yield np.frombuffer(
                        conf.pro[self.myid][0].stdout.read(IMG_W * IMG_H * 3),
                        dtype="uint8",
                    ).reshape((IMG_W, IMG_H, 3))

            except Exception as fe:
                continue

    def capture_one_screen_ddagrab(
        self,
        monitor_index=0,
        frames: int = 60,
        draw_mouse: bool = True,
        quant_screenshots: int | None = None,
        return_bytes: bool = False,
        debug: bool = False,
    ):
        r"""Capture the contents of a single screen using DDAGrab.

        This method captures the contents of a single screen identified by the monitor_index using the DDAGrab library.

        Args:
            monitor_index (int): The index of the monitor to capture (default: 0).
            frames (int): The number of frames to capture per second (default: 60).
            draw_mouse (bool): Whether to draw the mouse cursor in the captured frames (default: True).
            quant_screenshots (int | None): The number of screenshots to take (default: None).
                                            If None, a large number (4,000,000,000) is used as the default.
            return_bytes (bool): Whether to return the captured frames as bytes (default: False).
            debug (bool): Whether to enable debug mode (default: False).

        Raises:
            Exception: If an error occurs during the screen capture process.

        Yields:
            tuple or numpy.ndarray: A tuple representing the size of the captured frame (if `return_bytes` is True),
                                    or a numpy.ndarray representing the captured frame (if `return_bytes` is False).

        """
        offset_x = 0
        offset_y = 0
        IMG_H = allmoni[monitor_index]["height"]
        IMG_W = allmoni[monitor_index]["width"]
        multiid = f"{self.myid}_____{monitor_index}"
        self.myids.append(multiid)
        if not quant_screenshots:
            quant_screenshots = 4_000_000_000
        self.playvideo = True
        channels = 3
        self.ts.append(
            kthread.KThread(
                target=win_api_search6,
                args=(
                    self.myids[-1],
                    self.ffmpegpath,
                    IMG_H,
                    IMG_W,
                    frames,
                    offset_x,
                    offset_y,
                    draw_mouse,
                    self.ffmpeg_param,
                    debug,
                    monitor_index,
                ),
                name=self.myids[-1],
            )
        )
        coun = 0
        self.ts[-1].start()
        while self.playvideo:
            if not self.frame_checker_enabled:
                sleep(1 / frames)

            try:
                if coun >= quant_screenshots:
                    self.playvideo = False
                    break
                if return_bytes:
                    yield (IMG_H, IMG_W, channels), conf.pro[self.myids[-1]][
                        0
                    ].stdout.read(IMG_W * IMG_H * channels)
                else:
                    yield np.frombuffer(
                        conf.pro[self.myids[-1]][0].stdout.read(
                            IMG_W * IMG_H * channels
                        ),
                        dtype="uint8",
                    ).reshape((IMG_H, IMG_W, channels)).copy()
                coun += 1

            except Exception as fe:
                continue

    @staticmethod
    def capture_all_screens_ctypes(ascontiguousarray=False):
        """Capture the contents of all monitors using the ctypes module.

        This method captures the contents of all monitors using the ctypes module.
        The captured frames are yielded as a generator.

        Args:
            ascontiguousarray (bool): Whether to return the frames as contiguous arrays (default: False).

        Yields:
            numpy.ndarray: The captured frames as numpy arrays.

        """
        with ScreenshotOfAllMonitors(
            ascontiguousarray=ascontiguousarray
        ) as screenshots_all_monitor:
            yield from screenshots_all_monitor

    def capture_all_screens_gdigrab(
        self,
        frames: int = 60,
        draw_mouse: bool = True,
        quant_screenshots: int | None = None,
        return_bytes: bool = False,
        debug: bool = False,
    ):
        """Capture the contents of all screens using GDIGrab.

        This method captures the contents of all screens using the GDIGrab library.

        Args:
            frames (int): The number of frames to capture per second (default: 60).
            draw_mouse (bool): Whether to draw the mouse cursor in the captured frames (default: True).
            quant_screenshots (int | None): The number of screenshots to take (default: None).
                                            If None, a large number (4,000,000,000) is used as the default.
            return_bytes (bool): Whether to return the captured frames as bytes (default: False).
            debug (bool): Whether to enable debug mode (default: False).

        Raises:
            Exception: If an error occurs during the screen capture process.

        Yields:
            tuple or numpy.ndarray: A tuple representing the size of the captured frame (if `return_bytes` is True),
                                    or a numpy.ndarray representing the captured frame (if `return_bytes` is False).

        """

        if not quant_screenshots:
            quant_screenshots = 4_000_000_000
        width = monwidth
        height = max_monitor_height
        win_api_search4(
            myid=self.myid,
            width=width,
            height=height,
            updatedwindowtext=None,
            ffmpegpath=self.ffmpegpath,
            frames=frames,
            draw_mouse=draw_mouse,
            flags=self.ffmpeg_param,
            debug=debug,
            revertnamefunction=None,
            change_titel_back_after=None,
        )
        coun = 0
        while self.playvideo:
            if not self.frame_checker_enabled:
                sleep(1 / frames)
            try:
                if coun >= quant_screenshots:
                    self.playvideo = False
                    break
                if return_bytes:
                    yield (height, width, 3), conf.pro[self.myid][0].stdout.read(
                        height * width * 3
                    )
                else:
                    yield np.frombuffer(
                        conf.pro[self.myid][0].stdout.read(height * width * 3),
                        dtype="uint8",
                    ).reshape((height, width, 3))

            except Exception:
                continue
            coun = coun + 1

    def capture_all_screens_ddagrab(
        self,
        frames: int = 60,
        draw_mouse: bool = True,
        quant_screenshots: int | None = None,
        debug: bool = False,
        vertical_or_horizontal: str = "horizontal",  # "horizontal" or "vertical"
        return_empty: bool = True,
    ):
        r"""Capture the contents of all screens using DDAGrab.

        This method captures the contents of all screens using the DDAGrab library.

        Args:
            frames (int): The number of frames to capture per second (default: 60).
            draw_mouse (bool): Whether to draw the mouse cursor in the captured frames (default: True).
            quant_screenshots (int | None): The number of screenshots to take (default: None).
                                            If None, a large number (4,000,000,000) is used as the default.
            debug (bool): Whether to enable debug mode (default: False).
            vertical_or_horizontal (str): The orientation of the concatenated screens ("horizontal" or "vertical")
                                          (default: "horizontal").
            return_empty (bool): Whether to return an empty array instead of the captured frames (default: True).
                             Results are also available as `temparrays.horizontal` and `temparrays.vertical`.
                             Allocating memory for all screenshots and concatenating them is very expensive.
                             `temparrays.horizontal` and `temparrays.vertical` are used repeatedly.


        Raises:
            Exception: If an error occurs during the screen capture process.

        Yields:
            numpy.ndarray: The captured frames as numpy arrays.

        """
        emptyarray = np.array([], dtype=np.uint8)
        if vertical_or_horizontal == "vertical":
            adjust_array_size_vertical()
        else:
            adjust_array_size_horizontal()

        tuit = tuple(reversed(range(len(allmoni))))[:-1]
        alla = [
            FFmpegshot(
                ffmpegpath=self.ffmpegpath,
                ffmpeg_param=self.ffmpeg_param,
            )
            for _ in tuit
        ]
        for a_ in alla:
            self.additional_instances.append(a_)

        iterallscreens = [
            self.additional_instances[n].capture_one_screen_ddagrab(
                monitor_index=mn,
                frames=frames,
                draw_mouse=draw_mouse,
                quant_screenshots=quant_screenshots,
                debug=debug,
            )
            for n, mn in enumerate(tuit)
        ]
        iterallscreens.insert(
            0,
            self.capture_one_screen_ddagrab(
                monitor_index=0,
                frames=frames,
                draw_mouse=draw_mouse,
                quant_screenshots=quant_screenshots,
                debug=debug,
            ),
        )
        if vertical_or_horizontal == "vertical":
            for screens in zip(*iterallscreens):
                if not self.frame_checker_enabled:
                    sleep(1 / frames)

                fastconcat_vertical(
                    screens,
                    checkarraysize=False,
                )
                if not return_empty:
                    yield temparrays.vertical.copy()
                else:
                    yield emptyarray
        else:
            for screens in zip(*iterallscreens):
                if not self.frame_checker_enabled:
                    sleep(1 / frames)

                fastconcat_horizontal(
                    screens,
                    checkarraysize=False,
                )
                if not return_empty:
                    yield temparrays.horizontal.copy()
                else:
                    yield emptyarray


def sleep(t):
    for _ in range(int(t * 100)):
        sleep_(0.001)


def are_numbers_equal(number1, number2, allowed_difference=10**1):
    return abs(number1 - number2) < allowed_difference


def get_max_framerate(
    function: str,
    startframes: int = 30,
    endframes: int = 60,
    timeout: int = 2,
    sleeptimebeforekilling: int = 1,
    framedifference: int = 100,
    *args,
    **kwargs,
) -> dict:
    ffmpegscreenshotobj = FFmpegshot(ffmpeg_param=())
    ffmpegscreenshotobj.frame_checker_enabled = True

    maxresult = {}
    for nowframes in range(startframes, endframes, 1):
        kwargs["frames"] = nowframes
        piciter = getattr(ffmpegscreenshotobj, function)(*args, **kwargs)
        fps = 0
        timeoutfinal = time() + timeout
        for ini, pic in enumerate(piciter):
            if time() > timeoutfinal:
                ffmpegscreenshotobj.playvideo = False
                sleep(sleeptimebeforekilling)
                ffmpegscreenshotobj.kill_ffmpeg()
                ffmpegscreenshotobj.playvideo = True
                break
            fps += 1
        print(nowframes, fps)
        if are_numbers_equal(fps, nowframes, allowed_difference=framedifference):
            maxresult[nowframes] = fps
            continue
        else:
            break
    return maxresult


def start_multiprocessing(
    sleeptimebeforekilling: int | float,
    procid: int,
    stopflag,
    result_queue,
    function: str,
    *args,
    **kwargs,
):
    ffmpegscreenshotobj = FFmpegshot(
        ffmpeg_param=(),
    )
    doublescreenflag = None
    doublescreen = False
    if function == "capture_all_screens_ddagrab":
        doublescreen = True
        kwargs["return_empty"] = True
        if not "vertical_or_horizontal" in kwargs:
            kwargs["vertical_or_horizontal"] = "horizontal"

        if kwargs["vertical_or_horizontal"] == "vertical":
            doublescreenflag = lambda: temparrays.vertical
        else:
            doublescreenflag = lambda: temparrays.horizontal
    piciter = getattr(ffmpegscreenshotobj, function)(*args, **kwargs)
    for bild in piciter:
        if stopflag.value:
            ffmpegscreenshotobj.playvideo = False
            sleep(sleeptimebeforekilling)
            try:
                ffmpegscreenshotobj.kill_ffmpeg()

                sys.exit(0)
            except Exception:
                os._exit(0)

        if not doublescreen:
            result_queue.put((procid, bild))
        else:
            result_queue.put((procid, doublescreenflag()))

    try:
        ffmpegscreenshotobj.kill_ffmpeg()
        sys.exit(0)
    except Exception:
        os._exit(0)
