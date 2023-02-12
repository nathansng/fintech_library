import pandas as pd

class DataLoader:
    def __init__(self, path, file_type=None, index=None, index_col=None, date_col=None, date_format=None): 
        self.path = path
        self.data = None
    
        # Load in data 
        if self.path is not None and file_type is not None: 
            self.load_data(file_type)
        
        # Filter by index if required
        if index is not None: 
            if index_col is not None: 
                self.filter_index(index, index_col)
            else: 
                self.filter_index(index)
        
        # Format date if required 
        if date_col is not None:
            if date_format is not None: 
                self.date_index(date_col, date_format)
            else:
                self.date_index(date_col)
        
    def load_data(self, file_type="csv"):
        """
        Loads data into dataframe object based on file type
        :param file_type: File type of data file 
        :return: None
        """
        if file_type == "csv":
            self.data = pd.read_csv(self.path)
        elif file_type == "json":
            self.data = pd.read_json(self.path)
            
    def filter_index(self, index, index_col="Index"):
        """
        Filters for only certain indices of data 
        :param index: String of index name to filter for 
        :return: True if index column exists and filter is applied, False otherwise 
        """
        if self.data is not None and index_col in self.data.columns: 
            self.data = self.data[self.data[index_col] == index]
            return True
        return False
    
    def date_index(self, date_col, date_format="%Y-%m-%d"):
        """
        Turns a string date column into a date object and calculates an index based on date
        :param date_col: Name of date column 
        :param date_format: String format of string date 
        :return: True if date column exists and data processing is applied, False otherwise 
        """
        if self.data is not None and date_col in self.data.columns:
            self.data['numeric_date'] = self.data[date_col].apply(lambda date: datetime.datetime.strptime(date, date_format))
            self.data['date_index'] = (self.data["numeric_date"] - self.data["numeric_date"].min()).dt.days
            return True
        return False