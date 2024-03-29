'''
Logger class - to write messages both to logfile and stdout
'''
import sys


class Logger():
    def __init__(self, logpath):
        self.terminal = sys.stdout
        self.logfile = open(logpath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.write("\n")
        self.logfile.write(message)
        self.logfile.write("\n")
        self.logfile.flush()

    def close(self):
        self.logfile.close()
