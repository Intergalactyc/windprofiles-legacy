import os
from datetime import datetime

class Logger:
    def __init__(self, logfile = 'output.log', pid = 0):
        self.is_printer = False
        self.is_void = False
        self.logfile = logfile
        self.pid = pid        
    
    def log(self, string, timestamp = False):
        log_string = f'[{datetime.now()}] {string}' if timestamp else str(string)
        pid = self.pid if self.pid else 'LOGPARENT'
        log_string = f'[[{pid}]] {log_string}'
        with open(self.logfile, 'a') as f:
            f.write(log_string+'\n')
        return

    def sublogger(self, pid = None):
        if pid is None:
            pid = os.getpid()
        self.log(f'Spawned sublogger for pid {pid}', timestamp=True)
        if self.is_printer:
            return Printer(pid = pid)
        if self.is_void:
            return VoidLogger()
        return Logger(logfile = self.logfile, pid = pid)
    
class VoidLogger(Logger):
    def __init__(self):
        Logger.__init__(self)
        self.is_void = True
    
    def log(self, string, timestamp = False):
        return
    
class Printer(Logger):
    def __init__(self, pid = 0):
        Logger.__init__(self, pid = pid)
        self.is_printer = True

    def log(self, string, timestamp = False):
        log_string = f'[{datetime.now()}] {string}' if timestamp else str(string) 
        if self.pid: log_string = f'[[{self.pid}]] {log_string}'
        print(log_string)
        return
