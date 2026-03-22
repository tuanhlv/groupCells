"""
Mimic manual grouping, remove process_capacity_outliers and process_ir_outliers
"""
import pandas as pd
import time
from typing import List, Any
from pydantic import BaseModel, Field, ValidationError
from functools import wraps
from contextlib import contextmanager
import requests


# ============================== Pydantic Models for Validation ==============================
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

# ======================================= Decorators =========================================
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


# ====================================== Context Managers ======================================
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


# ======================================== BatteryCellGrouper ===================================
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
    def group_outside_in(self, allow_c_range: float) -> None:
        """
        Mimics manual 'greedy' grouping: Starts at extreme highs/lows of Capacity.
        If the extreme group fails the range check, the single most extreme cell
        is discarded as an outlier. If it passes, it becomes a pack.
        """
        # Sort by Capacity to prioritize it
        self.df_n.sort_values(by=[self.CAP_COL], ascending=True, inplace=True)

        dropped_outliers = 0

        # Alternate between checking the bottom (lowest) and top (highest)
        check_bottom = True

        while self.df_n.shape[0] >= self.config.pack_size:
            if check_bottom:
                # Look at the lowest capacity cells
                target_group = self.df_n.head(self.config.pack_size)
            else:
                # Look at the highest capacity cells
                target_group = self.df_n.tail(self.config.pack_size)

            # Check if this group meets the capacity range limit
            if self.check_range(target_group, self.CAP_COL, allow_c_range) == 1:
                # Fails range check: Spread is too wide.
                # Drop the single most extreme cell (either the absolute lowest or absolute highest)
                if check_bottom:
                    self.df_n = self.df_n.tail(-1)  # Drop lowest
                else:
                    self.df_n = self.df_n.head(-1)  # Drop highest

                dropped_outliers += 1
                # Don't switch sides; re-evaluate this side with the new extreme
                continue

            else:
                # Passes range check: We are comfortable with this distribution. Group them!
                self.pack_id += 1
                for _, row in target_group.iterrows():
                    self.res_cellID.append(row['Cell ID'])
                    self.res_Cap.append(row[self.CAP_COL])
                    self.res_IR.append(row[self.IR_COL])
                    self.res_packID.append(self.pack_id)

                # Remove grouped cells from the pool
                if check_bottom:
                    self.df_n = self.df_n.tail(-self.config.pack_size)
                else:
                    self.df_n = self.df_n.head(-self.config.pack_size)

                # Switch sides to balance the grouping from both extremes inward
                check_bottom = not check_bottom

        print(f"[INFO] Grouping complete. {dropped_outliers} cells were excluded as extreme outliers.")
        if self.df_n.shape[0] > 0:
            print(f"[INFO] {self.df_n.shape[0]} leftover cells excluded (not enough to form a final pack).")


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
            return
        self.c_range = self.get_user_input('Input capacity range, +/- %: ', float)
        self.group_outside_in(self.c_range)
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
            'QB-Realm-Hostname': self.url.replace('https://', '').split('/')[0],
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
                "user_token"
            )
            QB.run()

    except ValidationError as e:
        print(f"[ERROR] Configuration validation failed:\n{e}")
    except ValueError:
        print("[ERROR] Invalid input type. Please enter numerical values where expected.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
