import sys
from multiprocessing import Process, SimpleQueue, Value
import asyncio

from kthread_sleep import sleep

from collections import defaultdict, deque

from ffmpeg_screenshot_pipe import start_multiprocessing


procresults = sys.modules[__name__]
procresults.processes = []
procresults.allthreads = []
procresults.dequebuffer = 24
lambdadeq = lambda: deque([], procresults.dequebuffer)
procresults.results = defaultdict(lambdadeq)
procresults.result_queue = SimpleQueue()
procresults.tasks = 2
procresults.function2execute = start_multiprocessing
procresults.asyncsleep=0.001
procresults.stop_flag = Value('b', False)
procresults.stop = None
procresults.stoploop=None
procresults.sleeptimeafterkill = 5
procresults.sleeptimebeforekill = lambda : procresults.sleeptimeafterkill * .9
def stopfunction():

    try:
        try:
            procresults.stop_flag.value = True
            sleep(procresults.sleeptimeafterkill)
        except Exception:
            pass
        try:
            while True:
                procresults.result_queue.close()
                break
        except Exception:
            pass
        try:
            procresults.stoploop.stop()
        except Exception:
            pass
        try:
            procresults.stoploop.close()
        except Exception:
            pass
        for pa in procresults.processes:
            try:
                if pa.is_alive():
                    pa.terminate()
            except Exception:
                continue

    except Exception:
        pass



def runasync(args):
    procresults.tasks =len(args)
    if procresults.tasks == 1:
        procresults.asyncsleep = 0
    async def process_output():
        while True:
            try:
                if procresults.stop_flag.value:
                    break
                chunk = procresults.result_queue.get()
                procresults.results[chunk[0]].append(chunk[-1])
                await asyncio.sleep(procresults.asyncsleep)  # Allow other tasks to run
            except Exception:
                continue

    def fx():
        sleeptimebeforekilling = procresults.sleeptimebeforekill()
        for i,mydi in zip(range(procresults.tasks),args):
            procresults.processes.append(
                Process(target=procresults.function2execute, args=(sleeptimebeforekilling,i, procresults.stop_flag, procresults.result_queue),
                        kwargs=mydi))

            procresults.processes[-1].daemon = True
            procresults.processes[-1].start()

    async def main():
        await asyncio.gather(*[process_output() for _ in range(procresults.tasks)])

    fx()

    loop = asyncio.get_event_loop()
    procresults.stoploop = loop
    procresults.stop=lambda:stopfunction()
    try:
        loop.run_until_complete(main())
    except Exception:
        pass

    finally:
        try:
            try:
                loop.stop()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
        except Exception as fe:
            print(fe)


if __name__ =='__main__':
    pass

