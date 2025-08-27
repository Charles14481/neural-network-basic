import matplotlib.pyplot as plt
import random

class ProgressDisplay:
    """
    Create plot that shows model loss over runs
    
    Pass to model train to see
    """
    def __init__(self, interval=0):
        plt.ion()  # Turn on interactive mode
        
        self.losses = []
        self.index = 0
        self.interval = interval
        
        # Create the figure and axis
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], marker='o', linestyle='-', color='b')

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 20)
        self.ax.set_xlabel('runs')
        self.ax.set_ylabel('loss')
        self.ax.set_title('Plot of loss versus time')
        self.ax.grid(True)

        self.shown = False

    def update(self, add, dt):
        """Update function for animation"""
        self.shown = True
        self.losses.append(add)

        # Update line data
        self.line.set_data(list(range(len(self.losses))), self.losses)
        
        # Expand axis if needed
        self.ax.set_xlim(0, max(10, self.index + 1))
        self.ax.set_ylim(0, max(20, max(self.losses) * 1.1))

        self.ax.text(self.index + 0.1, add, f"{dt:.2f}", fontsize=9, color='red')
        
        plt.draw()
        plt.pause(self.interval/1000)

        self.index += 1

    def keep(self):
        if (self.shown):
            plt.ioff()
            plt.show()
