import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import math
import time
from typing import List, Any
from pydantic import BaseModel, Field, ValidationError
from functools import wraps
from contextlib import contextmanager


# ========================== 1. Pydantic Models for Validation ========================
class GroupingConfig(BaseModel):
    pack_size: int = Field(..., gt=0, description="Number of cells in one pack/string")
    input_file: str = Field(..., min_length=1)
    output_file: str = Field(..., min_length=1)


# =================================== 2. Decorators ===================================
def log_execution(func):
    """Decorator to log execution time of methods."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.time() - start_time
            print(f"[LOG] {func.__name__} completed in {elapsed:.4f} seconds.")

    return wrapper


# ================================ 3. Context Managers ================================
@contextmanager
def handle_pandas_exceptions():
    """Context manager for safe pandas I/O operations."""
    try:
        yield
    except pd.errors.EmptyDataError:
        print("[ERROR] The provided CSV file is empty.")
        raise
    except FileNotFoundError:
        print("[ERROR] The specified CSV file could not be found.")
        raise
    except Exception as e:
        print(f"[ERROR] An unexpected pandas error occurred: {e}")
        raise


# ============================ 4. Object-Oriented Programming (OOP) ============================
class BatteryCellGrouper:
    CAP_COL: str = 'Latest Cycle N1 Discharge Capacity (Ah)'
    CAP_COL_N: str = 'Latest Cycle N1 Discharge Capacity (Ah)_n'
    IR_COL: str = 'Latest Cycle N1 DCIR (Ohm-cm2)'
    IR_COL_N: str = 'Latest Cycle N1 DCIR (Ohm-cm2)_n'
    SKIP_COLUMNS: set = {"Related Battery Pack", "Cell ID", "dist1", "dist2"}

    def __init__(self, config: GroupingConfig) -> None:
        self.config: GroupingConfig = config
        self.df: pd.DataFrame = pd.DataFrame()
        self.df_n: pd.DataFrame = pd.DataFrame()
        self.res_packID: List[int] = []
        self.res_cellID: List[Any] = []
        self.res_Cap: List[float] = []
        self.res_IR: List[float] = []
        self.pack_id: int = 0

    @log_execution
    def load_data(self) -> None:
        """Loads data from CSV using context-managed error handling."""
        with handle_pandas_exceptions():
            self.df = pd.read_csv(self.config.input_file, engine='c')
            print(f"Total # cells = {self.df.shape[0]}")

    def normalize(self) -> None:
        """Normalizes columns using list comprehensions to filter columns."""
        self.df_n = self.df.copy()

        # 5. List Comprehension
        cols_to_normalize = [col for col in self.df.columns if col not in self.SKIP_COLUMNS]

        for feature in cols_to_normalize:
            max_val = self.df[feature].max()
            min_val = self.df[feature].min()
            if max_val != min_val:
                self.df_n[f"{feature}_n"] = (self.df[feature] - min_val) / (max_val - min_val)
            else:
                self.df_n[f"{feature}_n"] = 0.0

    def check_range(self, df_subset: pd.DataFrame, col: str, range_limit: float) -> int:
        """Calculates if variation range is outside allowed limits with Error Handling."""
        try:
            values = df_subset[col].tolist()
            avg = sum(values) / self.config.pack_size
            if avg == 0: return 0

            range_u = (max(values) - avg) / avg
            range_l = (avg - min(values)) / avg

            return 1 if max(range_u, range_l) > (range_limit / 100.0) else 0
        except ZeroDivisionError:
            return 0

    def get_user_input(self, prompt: str, type_func: type = float) -> Any:
        """Helper to get and validate user input recursively."""
        while True:
            try:
                return type_func(input(prompt))
            except ValueError:
                print(f"[ERROR] Invalid input. Please enter a valid {type_func.__name__}.")

    @log_execution
    def process_capacity_outliers(self, allow_c_range: float, check_upper: bool) -> None:
        """Groups points iteratively by eliminating/segmenting capacity outliers."""
        sort_ascending = not check_upper
        self.df_n.sort_values(by=[self.CAP_COL], ascending=sort_ascending, inplace=True)
        extreme_cap = self.df_n.iloc[:self.config.pack_size]

        if self.check_range(extreme_cap, self.CAP_COL, allow_c_range) == 1:
            outlier_type = "Upper" if check_upper else "Lower"
            remove_outlier = self.get_user_input(
                f'{outlier_type} capacity outliers found. Enter "1" to remove or "0" to skip: ', int
            )
        else:
            remove_outlier = 0

        while self.check_range(extreme_cap, self.CAP_COL, allow_c_range) == 1:
            if remove_outlier == 1:
                if self.df_n.shape[0] > 1:
                    self.df_n = self.df_n.tail(-1)
                    extreme_cap = self.df_n.iloc[:self.config.pack_size]
                else:
                    print('All cells removed! Increase allowable range.')
                    break
            else:
                if self.df_n.shape[0] >= self.config.pack_size:
                    self.pack_id += 1
                    if check_upper:
                        print(f"   pack ID = {self.pack_id}")
                    p1 = [self.df_n.iloc[0][self.CAP_COL_N], self.df_n.iloc[0][self.IR_COL_N]]
                    temp_np = self.df_n[[self.CAP_COL_N, self.IR_COL_N]].to_numpy()
                    dist = [math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) for p2 in temp_np]
                    self.df_n['Distance'] = dist
                    self.df_n.sort_values(by=["Distance"], ascending=True, inplace=True)

                    for i in range(self.config.pack_size):
                        row = self.df_n.iloc[i]
                        self.res_cellID.append(row['Cell ID'])
                        self.res_Cap.append(row[self.CAP_COL])
                        self.res_IR.append(row[self.IR_COL])
                        self.res_packID.append(self.pack_id)

                    self.df_n = self.df_n.tail(-self.config.pack_size)
                    self.df_n.drop(columns=['Distance'], inplace=True)
                    self.df_n.sort_values(by=[self.CAP_COL], ascending=sort_ascending, inplace=True)

                    if check_upper:
                        max_cap = max(self.df_n[self.CAP_COL]) if not self.df_n.empty else 0
                        print(f"{self.df_n.shape[0]} cells remain with max capacity = {max_cap}")
                    extreme_cap = self.df_n.head(self.config.pack_size)
                else:
                    break

    @log_execution
    def process_ir_outliers(self, allow_ir_range: float) -> None:
        """Removes IR outliers based on user preferences."""
        print("Removing IR outliers is NOT recommended.")
        remove_ir_u = self.get_user_input('Enter "1" to continue with UPPER IR outliers removal or "0" to skip: ', int)

        if remove_ir_u == 1:
            print("Removing upper IR outliers...")
            self.df_n.sort_values(by=[self.IR_COL], ascending=False, inplace=True)
            max_ir = self.df_n.iloc[:self.config.pack_size]
            while self.check_range(max_ir, self.IR_COL, allow_ir_range) == 1 and not self.df_n.empty:
                self.df_n = self.df_n.tail(-1)
                max_ir = self.df_n.iloc[:self.config.pack_size]
            print(f"# cells after removal of UPPER IR outliers = {self.df_n.shape[0]}")

        remove_ir_l = self.get_user_input('Enter "1" to continue with LOWER IR outliers removal or "0" to skip: ', int)
        if remove_ir_l == 1:
            print("Removing lower IR outliers...")
            self.df_n.sort_values(by=[self.IR_COL], ascending=True, inplace=True)
            min_ir = self.df_n.iloc[:self.config.pack_size]
            while self.check_range(min_ir, self.IR_COL, allow_ir_range) == 1 and not self.df_n.empty:
                self.df_n = self.df_n.tail(-1)
                min_ir = self.df_n.iloc[:self.config.pack_size]
            print(f"# cells after removal of LOWER IR outliers = {self.df_n.shape[0]}")

    @log_execution
    def group_remaining_gmm(self) -> None:
        """Groups the remaining standard cells using Gaussian Mixture Models."""
        df_g = self.df_n.copy()
        pack_id_gmm = 0

        while df_g.shape[0] >= self.config.pack_size:
            n_components = int(df_g.shape[0] / self.config.pack_size)
            try:
                gm = GaussianMixture(n_components=n_components).fit(df_g[[self.CAP_COL_N, self.IR_COL_N]])
                centers = gm.means_
                centers = sorted([c for c in centers], key=lambda k: [k[0], k[1]])
                pack_id_gmm = pack_id_gmm + self.pack_id + 1
                p1 = centers[0]
                temp_np = df_g[[self.CAP_COL_N, self.IR_COL_N]].to_numpy()
                dist = [math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) for p2 in temp_np]
                df_g['Distance'] = dist
                df_g.sort_values(by=['Distance'], ascending=True, inplace=True)

                for i in range(self.config.pack_size):
                    row = df_g.iloc[i]
                    self.res_cellID.append(row['Cell ID'])
                    self.res_Cap.append(row[self.CAP_COL])
                    self.res_IR.append(row[self.IR_COL])
                    self.res_packID.append(pack_id_gmm)

                df_g = df_g.tail(-self.config.pack_size)
                df_g.drop(columns=['Distance'], inplace=True)

            except Exception as e:
                print(f"[ERROR] GMM fitting failed: {e}")
                break

    @log_execution
    def save_results(self) -> None:
        """Consolidates arrays and saves the grouped configuration to CSV."""
        try:
            res_df = pd.DataFrame({
                'Cell ID': self.res_cellID,
                'Related Battery Pack': self.res_packID,
                'Cap': self.res_Cap,
                'IR': self.res_IR
            })
            res_df.to_csv(self.config.output_file, index=False)
            print(f"Results successfully saved to {self.config.output_file}")
        except IOError as e:
            print(f"[ERROR] Failed to write to {self.config.output_file}: {e}")

    def run(self) -> None:
        """Orchestrates the script logic."""
        self.load_data()
        if self.df.empty:
            print("[ERROR] No data loaded. Exiting.")
            return

        self.normalize()

        allow_c_range = self.get_user_input('Input capacity range, +/- %: ', float)
        allow_ir_range = self.get_user_input('Input IR range, +/- %: ', float)

        self.process_capacity_outliers(allow_c_range, check_upper=True)
        self.process_capacity_outliers(allow_c_range, check_upper=False)
        self.process_ir_outliers(allow_ir_range)
        self.group_remaining_gmm()
        self.save_results()


# ============================= Script Execution =================================
if __name__ == "__main__":
    try:
        pack_size_input = int(input('Input # cells in 1 pack/string: '))
        filename_input = input('Input csv file name with the list of test and cell IDs (without .csv): ')

        # Pydantic validation on startup config
        config = GroupingConfig(
            pack_size=pack_size_input,
            input_file=f"{filename_input}.csv",
            output_file=f"grouping_result_{filename_input}.csv"
        )

        grouper = BatteryCellGrouper(config)
        grouper.run()

    except ValidationError as e:
        print(f"[ERROR] Configuration validation failed:\n{e}")
    except ValueError:
        print("[ERROR] Invalid input type. Please enter numerical values where expected.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")