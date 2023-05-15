import multiprocessing
import random
# import string
import time
def task(i):
    print("Hello, I am a task that marked",i)
    time.sleep(random.random()/5)
    print("Goodbye, I am a task that marked",i)



if __name__ == '__main__':
    for i in range(10) :
        pcs = multiprocessing.Process(target=task, kwargs={'i':i}, name='process_'+str(i),)

        pcs.start()
        # pcs.join()
