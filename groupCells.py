"""
CKM: Constrained K-Means:
 - spatial grouping as standard clustering but allows you to set a strict size_min and size_max for every single cluster.
 - optimizes all groups simultaneously rather than greedily
PCA (Principal Component Analysis) creates a "blended" score of Capacity and IR.
While sorting by PCA does start at the extremes, it mixes the variables together.
Because it blends them, it doesn't strictly prioritize Capacity over IR.
"""

import pandas as pd
from sklearn.decomposition import PCA
import math
import time
from typing import List, Any
from pydantic import BaseModel, Field, ValidationError
from functools import wraps
from contextlib import contextmanager
import requests


# ============================== 1. Pydantic Models for Validation ==============================
class GroupingConfig(BaseModel):
    pack_size: int = Field(..., gt=0, description="Number of cells in one pack/string")
    input_file: str = Field(..., min_length=1)
    output_file: str = Field(..., min_length=1)

class QBUpdateRecord(BaseModel):
    """Ensures each record update sent to QuickBase match the required JSON structure."""
    cell_id: int
    pack_id: int
    def to_qb_format(self) -> dict:
        return {
            "3": {"value": self.cell_id},
            "129": {"value": self.pack_id}
        }

# ======================================= 2. Decorators =========================================
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


# ======================================== 3. Context Managers ==========================================
@contextmanager
def handle_api_errors():
    """Context manager to catch and cleanly log QuickBase API network and HTTP errors."""
    try:
        yield
    except requests.exceptions.HTTPError as e:
        # Catches 4xx and 5xx errors (e.g., Bad Request, Unauthorized, Server Error)
        print(f"[ERROR] QuickBase API rejected the request. HTTP {e.response.status_code}: {e.response.text}")
    except requests.exceptions.ConnectionError:
        # Catches DNS failures, refused connections, etc.
        print("[ERROR] Connection failed. Could not reach QuickBase. Check your network.")
    except requests.exceptions.Timeout:
        # Catches requests that hang indefinitely
        print("[ERROR] The QuickBase API request timed out.")
    except requests.exceptions.RequestException as e:
        # Catch-all for any other requests-related issues
        print(f"[ERROR] An unexpected network error occurred: {e}")


# ================================= 4. Object-Oriented Programming (OOP) =================================
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
        self.df = pd.read_csv(self.config.input_file, engine='c')
        print(f"Total # cells = {self.df.shape[0]}")
        self.df_n = self.df.copy()
        cols_to_normalize = [col for col in self.df.columns if col not in self.SKIP_COLUMNS]
        for feature in cols_to_normalize:
            max_val = self.df[feature].max()
            min_val = self.df[feature].min()
            if max_val != min_val:
                self.df_n[f"{feature}_n"] = (self.df[feature] - min_val) / (max_val - min_val)
            else:
                self.df_n[f"{feature}_n"] = 0.0

    def check_range(self, df_subset: pd.DataFrame, col: str, range_limit: float) -> int:
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
        while True:
            try:
                return type_func(input(prompt))
            except ValueError:
                print(f"[ERROR] Invalid input. Please enter a valid {type_func.__name__}.")

    @log_execution
    def process_capacity_outliers(self, allow_c_range: float, check_upper: bool) -> None:
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
                    if check_upper: print(f"   pack ID = {self.pack_id}")

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
    def group_remaining_pca(self) -> None:
        """Groups the remaining standard cells using PCA and Array Slicing."""
        df_g = self.df_n.copy()
        if df_g.shape[0] < self.config.pack_size:
            print("[INFO] Not enough cells remaining to form a full pack.")
            return

        # 1. PCA Reduction: Scales easily to 3 or 4 variables in the future
        pca = PCA(n_components=1)
        features = df_g[[self.CAP_COL_N, self.IR_COL_N]]
        df_g['PCA_Score'] = pca.fit_transform(features)

        # 2. Sort by the 1D principal component score
        df_g.sort_values(by=['PCA_Score'], ascending=True, inplace=True)

        pack_id_pca = self.pack_id

        # 3. Slice array into exact pack sizes
        for i in range(0, df_g.shape[0], self.config.pack_size):
            chunk = df_g.iloc[i: i + self.config.pack_size]

            if chunk.shape[0] == self.config.pack_size:
                pack_id_pca += 1
                for _, row in chunk.iterrows():
                    self.res_cellID.append(row['Cell ID'])
                    self.res_Cap.append(row[self.CAP_COL])
                    self.res_IR.append(row[self.IR_COL])
                    self.res_packID.append(pack_id_pca)
            else:
                print(
                    f"[INFO] {chunk.shape[0]} leftover cells excluded (not enough for a pack of {self.config.pack_size}).")

        self.pack_id = pack_id_pca

    @log_execution
    def save_results(self) -> None:
        """Save results to dataframe to pass to Validator.
                Need to modify to save res_df only, bring to_csv to Validator"""
        self.res_df = pd.DataFrame({
            'Cell ID': self.res_cellID,
            'Related Cell Set': self.res_packID,
            'Cap': self.res_Cap,
            'IR': self.res_IR
        })

    def run(self) -> None:
        self.load_data()
        if self.df.empty:
            print("[ERROR] No data loaded. Exiting.")
            return
        self.c_range = self.get_user_input('Input capacity range, +/- %: ', float)
        self.ir_range = self.get_user_input('Input IR range, +/- %: ', float)
        self.process_capacity_outliers(self.c_range, check_upper=True)
        self.process_capacity_outliers(self.c_range, check_upper=False)
        self.process_ir_outliers(self.ir_range)
        self.group_remaining_pca()
        self.save_results()


# ===================================== Pack Validator =====================================
class PackValidator:
    def __init__(self, grouping: BatteryCellGrouper, grouping_output: str, validation_output: str) -> None:
        self.df = grouping.res_df
        self.out = grouping_output
        self.vf = validation_output
        self.range = grouping.c_range/100

    def validate_grouping(self, df: pd.DataFrame) -> pd.DataFrame:
        validation = []
        for pack in self.df['Related Cell Set'].unique():
            min_cap = df.loc[df['Related Cell Set'] == pack,'Cap'].min()
            max_cap = df.loc[df['Related Cell Set'] == pack, 'Cap'].max()
            avg_cap = df.loc[df['Related Cell Set'] == pack,'Cap'].mean()
            try:
                max_dev = max(max_cap - avg_cap, avg_cap - min_cap)/avg_cap
                validation.append(("pass" if max_dev <= self.range else "fail"))
            except ZeroDivisionError:
                validation.append("N/A")
        validation_df = pd.DataFrame({
            'Cell Set': self.df['Related Cell Set'].unique(),
            'Capacity Range Check': validation
        })
        return validation_df

    def print_to_csv(self) -> None:
        self.df.to_csv(self.out, index=False)
        self.validation_res.to_csv(self.vf, index=False)

    def run(self):
        self.validation_res = self.validate_grouping(self.df)
        if (self.validation_res["Capacity Range Check"] == "fail").any():
            print("Grouping validation failed, no print-to-csv.")
            self.res = "fail"
        else:
            print("Grouping validation passed, printing grouping result and validation result to csv files.")
            self.res = "pass"
            self.print_to_csv()

# ===================================== QuickBase Client =====================================
class QuickBaseClient:
    """Object-oriented wrapper for QuickBase API using requests."""

    def __init__(self, updates, url: str, database: str, user_token: str):
        self.url = url
        self.table_id = database
        self.headers = {
            'QB-Realm-Hostname': self.url,
            'Authorization': f'QB-USER-TOKEN {user_token}',
            'Content-Type': 'application/json'
        }
        self.api_base = "https://api.quickbase.com/v1/records"
        self.updates = updates

    def push_updates(self, updates: list[QBUpdateRecord]):
        if not updates:
            print("No updates needed.")
            return
        payload = {
            "to": self.table_id,
            "data": [u.to_qb_format() for u in updates],
            "mergeFieldId": "3"
        }
        with handle_api_errors():
            r = requests.post(self.api_base, headers=self.headers, json=payload, timeout=10)
            r.raise_for_status()  # triggers the context manager if QuickBase returns an error
            print(f"[INFO] Successfully updated {len(updates)} records in QuickBase.")

    def run(self):
        self.push_updates(self.updates)


# ===================================== Script Execution =====================================
if __name__ == "__main__":
    try:
        pack_size_input = int(input('Input # cells in 1 pack/string: '))
        filename_input = input('Input csv file name with the list of test and cell IDs (without .csv): ')
        config = GroupingConfig(
            pack_size=pack_size_input,
            input_file=f"{filename_input}.csv",
            output_file=f"grouping_result_{filename_input}.csv"
        )
        grouper = BatteryCellGrouper(config)
        grouper.run()
        validation_file = f"grouping_validation_result_{filename_input}.csv"
        validator = PackValidator(grouper, config.output_file, validation_file)
        validator.run()
        if validator.res == "pass":
            raw_updates = grouper.res_df[['Cell ID', 'Related Cell Set']].values.tolist()
            qb_records = [QBUpdateRecord(cell_id=row[0], pack_id=row[1]) for row in raw_updates]
            QB = QuickBaseClient(
                qb_records,
                "https://company.quickbase.com/v1/records",
                "table_id",
                "user_token"  # dbrobot user token
            )
            QB.run()

    except ValidationError as e:
        print(f"[ERROR] Configuration validation failed:\n{e}")
    except ValueError:
        print("[ERROR] Invalid input type. Please enter numerical values where expected.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
