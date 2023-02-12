import pandas as pd
import numpy as np
import datetime

class LinearApproximation:
    def __init__(self, max_error, min_segment_length, data=None, target_col=None, date_index=None):
        self.d = None
        self.transformed_d = None
        self.x = None
        self.y = None
        self.len_data = None
        self.max_idx = None

        self.max_error = max_error
        self.min_segment_length = min_segment_length
        
        if data is not None and target_col is not None:
            # Add data if provided info
            if date_index is not None:
                self.add_data(data, target_col, date_index)
            else:
                self.add_data(data, target_col)
    
    def add_data(self, data, target_col, date_index="date_index"):
        """
        Load in data to process with a date index column and target prediction column 
        :param data: Dataframe to use 
        :param target_col: Colum name containing target data 
        :param date_index: Column name containing date index values 
        :return: None
        """
        self.d = data 
        self.x = np.array(self.d[date_index])
        self.y = np.array(self.d[target_col])
        self.len_data = self.d.shape[0]
        self.max_idx = len(self.x) - 1


    def process_data(self):
        """
        Transform original data to pandas dataframe containing information about trends
        trends[i] = [trend_duration[i], trend_slope[i], original data points that make up trends[i]]

        :return: Dataframe of processed data
        """

        # set buffer and lower and upper bounds
        w = self.len_data // 7  # buffer ensures that there is enough data for 5 to 6 segments as specified in original paper
        lower_bound = w // 2
        upper_bound = int(2 * w)

        sequences = []

        i = 0
        while i < self.len_data:
            # bottom_up
            sequences += self.bottom_up(i, w)

            # slide window
            i = w
            w = min(i + int(min(upper_bound, max(lower_bound, self.best_line(i, upper_bound)))), self.len_data)

            print(f"{round(i/self.len_data * 100, 2)}% data processed \n")

        trends = [[None, None, None] for _ in range(len(sequences))]
        for idx, seq in enumerate(sequences):
            trends[idx][0] = seq[1] - seq[0]  # duration
            trends[idx][1] = self.__calculate_slope(self.x[seq[0]:seq[1]+1], self.y[seq[0]:seq[1]+1])
            trends[idx][2] = [self.y[i] for i in range(seq[0], seq[1])]

        #     return torch.tensor(trends, dtype=torch.float)
        self.transformed_d = pd.DataFrame(trends, columns=["trend_duration", "trend_slope", "trend_points"])
        return self.transformed_d


    def best_line(self, i, upper_bound):
        """
        Calculates end index of current window

        :param i: starting index of current window
        :param upper_bound: maximum size of window
        :return: ending index of current window
        """
        error = 0
        j = i
        while error <= self.max_error and j < i + upper_bound:
            j += self.min_segment_length
            curr_x, curr_y = self.x[i:j], self.y[i:j]
            error = self.__calculate_error(curr_x, curr_y)

        return j


    def bottom_up(self, i, j):
        """
        Performs bottom up algorithm on data[i:j] as described in: http://www.cs.ucr.edu/~eamonn/icdm-01.pdf
        and returns list of segments represented by indices

        :param i: starting index of current window
        :param j: ending index of current window
        :return: segments (2-D list)
                 segments[i] = [starting index of segments[i], ending index of segments[i]]
        """
        # print(f"Performing bottom_up with i={i}, j={j}, max_error={self.max_error}")
        # segment_increase = self.min_segment_length//4
        segments = [[k, k + 2] for k in range(i, j, 2)]

        fully_merged = False
        while not fully_merged:

            min_merge_error, min_merge_idx = float("inf"), None
            min_seg_length, min_seg_idx = float("inf"), None

            # get min error
            for idx in range(0, len(segments) - 1, 2):
                sub_i, sub_j = segments[idx][0], segments[idx+1][1]+1
                curr_x, curr_y = self.x[sub_i:sub_j], self.y[sub_i:sub_j]
                curr_error = self.__calculate_error(curr_x, curr_y)

                if curr_error < min_merge_error:
                    min_merge_error = curr_error
                    min_merge_idx = idx

            # get min segment length
            for idx in range(len(segments)):
                if segments[idx][1] - segments[idx][0] < min_seg_length:
                    min_seg_length = segments[idx][1] - segments[idx][0]
                    min_seg_idx = idx

            # find spots to merge if necessary
            replace, first_half, second_half = None, None, None
            if min_merge_error < self.max_error:
                segments[min_merge_idx] = [segments[min_merge_idx][0], min(self.max_idx, segments[min_merge_idx+1][1])]
                segments.pop(min_merge_idx+1)

            elif min_seg_length < self.min_segment_length:
                if min_seg_idx == len(segments) - 1:
                    min_seg_idx -= 1
                    
                segments[min_seg_idx] = [segments[min_seg_idx][0], min(self.max_idx, segments[min_seg_idx+1][1])]
                segments.pop(min_seg_idx+1)
                      
            else: 
                fully_merged = True

        return segments


    def __calculate_error(self, x, y):
        """
        Calculates least squared error of linear approximation of x, y

        :param x: inputs to linear approximation
        :param y: targets of linear approximation
        :return: error
        """
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            error = np.linalg.lstsq(A, y, rcond=None)[1][0]
        except IndexError:
            error = 0

        return error


    def __calculate_slope(self, x, y):
        """
        Calculates slope of linear approximation of x, y

        :param x: inputs to linear approximation
        :param y: targets of linear approximation
        :return: slope
        """
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        except IndexError:
            m = 0

        return m


    def __calculate_intercept(self, x, y):
        """
        Calculates intercept of linear approximation of x, y

        :param x: inputs to linear approximation
        :param y: targets of linear approximation
        :return: intercept
        """
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        except IndexError:
            c = 0

        return c


    def save_to_csv(self, file_path):
        """
        Saves transformed data to csv file for use

        :return: boolean
        """
        try:
            self.transformed_d.to_csv(file_path)
            return True

        except AttributeError:
            print("Transformed data does not exist: failed to save to csv file")
            return False
