import pandas as pd
import matplotlib.pyplot as plt

class Report:

    '''
    Class to handle all reporting and logging information for an agent
    '''


    def __init__(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        self.log = pd.DataFrame(columns=['Sol', 'Expanded', 'F.Max.Depth', 'Frontier', 'Cur.Depth', 'Cur.Cost'])
        self.__append__(False,0,0,0,0,0)


    def __append__(self, solution=False, nodesExpanded=0, frontierMaxDepth=0, nodesFrontier=0, currentDepth=0, currentCost=0):
        # Note that step iteration number is the index of the dataframe
        line = pd.Series(
            {
                'Sol' : solution,
                'Expanded': nodesExpanded,
                'F.Max.Depth': frontierMaxDepth,
                'Frontier': nodesFrontier,
                'Cur.Depth' : currentDepth,
                'Cur.Cost' : currentCost
            }
        )
        self.log = pd.concat([self.log, pd.DataFrame([line])], ignore_index=True)

    def print_report(self):
        '''
        Prints the report of the reporting
        '''
        self.log.describe().transpose()


    def plotNumberNodesFrontier(self, show=True):
        '''
        Prints iteration vs # nodes in frontier
        '''
        ax = self.log['Frontier'].plot(lw=2, colormap='jet', marker='.', markersize=10, title='Iteration vs # nodes in frontier')
        ax.set_xlabel("# Iteration")
        ax.set_ylabel("# Nodes")

        if show:
            plt.show()

        return ax


    def plotNodesAddedFrontier(self, nbins=20, show=True):
        '''
        Prints histogram of nodes added to frontier per step in bins
        '''

        ax = self.log['Expanded'].plot(kind='hist', colormap='jet', bins=nbins, title='Histogram of nodes added to frontier')
        ax.set_xlabel("Nodes added")
        ax.set_ylabel("Frequency")

        if show:
            plt.show()

        return ax


    def plotFrontierMaxDepth(self, show=True):
        '''
        Print plot of Frontier Maximum Depth
        '''

        ax = self.log['F.Max.Depth'].plot(lw=2, colormap='jet', marker='.', markersize=10, title='Iteration vs Frontier Maximum Depth')
        ax.set_xlabel("# Iteration")
        ax.set_ylabel("Max Depth")

        if show:
            plt.show()

        return ax

    def plotFrontierCost(self, show=True):
        '''
        Print plot of Frontier current node cost
        '''

        ax = self.log['Cur.Cost'].plot(lw=2, colormap='jet', marker='.', markersize=10,
                                          title='Iteration vs Current Node Cost')
        ax.set_xlabel("# Iteration")
        ax.set_ylabel("Cost")

        if show:
            plt.show()

        return ax

    def show(self):
        '''
        show plots
        '''
        plt.show()