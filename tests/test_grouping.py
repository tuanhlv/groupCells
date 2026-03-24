import pytest
import pandas as pd
from pydantic import ValidationError
from unittest.mock import MagicMock, patch
import requests

# Import the classes the main script
from groupCells import (
    BatteryCellGrouper,
    PackValidator,
    GroupingConfig,
    QBUpdateRecord,
    QuickBaseClient,
    handle_api_errors
)


# ==========================================
# 1. Fixtures: Reusable Setup Data
# ==========================================
@pytest.fixture
def mock_config():
    """Provides a valid GroupingConfig object."""
    return GroupingConfig(
        pack_size=2,
        input_file="dummy.csv",
        output_file="output.csv"
    )


@pytest.fixture
def sample_df():
    """Provides a clean, standardized dataframe for unit testing."""
    data = {
        'Cell ID': [101, 102, 103, 104],
        'Latest Cycle N1 Discharge Capacity (Ah)': [4.5, 4.6, 4.4, 4.55],
        'Latest Cycle N1 Discharge Capacity (Ah)_n': [0.5, 1.0, 0.0, 0.75],
        'Latest Cycle N1 DCIR (Ohm-cm2)': [0.012, 0.013, 0.011, 0.0125],
        'Latest Cycle N1 DCIR (Ohm-cm2)_n': [0.5, 1.0, 0.0, 0.75]
    }
    return pd.DataFrame(data)


# ==========================================
# 2. Schema & Validation Enforcement
# ==========================================
def test_grouping_config_guards():
    """Ensures GroupingConfig rejects invalid inputs."""
    # Test pack_size <= 0
    with pytest.raises(ValidationError):
        GroupingConfig(pack_size=0, input_file="test.csv", output_file="out.csv")

    # Test empty file names
    with pytest.raises(ValidationError):
        GroupingConfig(pack_size=2, input_file="", output_file="out.csv")


def test_qb_update_record_guards():
    """Ensures QBUpdateRecord requires all fields and correct types."""
    # Missing pack_id
    with pytest.raises(ValidationError):
        QBUpdateRecord(cell_id=101)

    # Valid instantiation should pass and format correctly
    record = QBUpdateRecord(cell_id=101, pack_id=5)
    formatted = record.to_qb_format()
    assert formatted["3"]["value"] == 101
    assert formatted["129"]["value"] == 5


# ==========================================
# 3. Mathematical Edge Cases
# ==========================================
def test_zero_division_handling(mock_config):
    """Ensures a pack with 0.0 values doesn't crash the script."""
    grouper = BatteryCellGrouper(mock_config)
    df_zero = pd.DataFrame({'Cap': [0.0, 0.0]})

    # check_range should safely catch the ZeroDivisionError and return 0
    assert grouper.check_range(df_zero, 'Cap', 5.0) == 0


def test_insufficient_data(mock_config, capsys):
    """Verifies behavior when remaining cells are fewer than pack_size."""
    grouper = BatteryCellGrouper(mock_config)
    # Give it only 1 cell, but pack_size is 2
    grouper.df_n = pd.DataFrame({
        'Cell ID': [1],
        'Latest Cycle N1 Discharge Capacity (Ah)_n': [1.0],
        'Latest Cycle N1 DCIR (Ohm-cm2)_n': [1.0]
    })

    grouper.group_remaining_pca()

    # Check that it printed the correct INFO log instead of failing
    captured = capsys.readouterr()
    assert "Not enough cells remaining" in captured.out


# ==========================================
# 4. Boundary (Tolerance) Testing
# ==========================================
def test_exact_tolerance_boundary(mock_config):
    """Tests the absolute edge of the allowable range percentage."""
    grouper = BatteryCellGrouper(mock_config)

    # Average is 100. Max deviation is exactly 5.0 (5.0%)
    df_bound = pd.DataFrame({'Cap': [105.0, 95.0]})

    # Given fixed data, if tolerance is exactly 5.0%, it should pass (return 0)
    assert grouper.check_range(df_bound, 'Cap', 5.0) == 0

    # Given fixed data, if tolerance is 4.99%, it must fail (return 1)
    assert grouper.check_range(df_bound, 'Cap', 4.99) == 1


# ==========================================
# 5. Data Integrity & Shape
# ==========================================
def test_data_integrity_after_grouping(mock_config, sample_df):
    """Verifies no rows are lost and cell IDs stay unique during PCA grouping."""
    grouper = BatteryCellGrouper(mock_config)
    grouper.df_n = sample_df

    grouper.group_remaining_pca()
    grouper.save_results()

    res = grouper.res_df

    # Count Consistency: 4 input cells should yield 4 output cells
    assert len(res) == 4

    # Column Preservation
    assert 'Cell ID' in res.columns
    assert 'Related Cell Set' in res.columns

    # Unique Grouping: All 4 cell IDs must be unique
    assert len(res['Cell ID'].unique()) == 4

    # With pack_size 2, there should be exactly 2 distinct packs
    assert len(res['Related Cell Set'].unique()) == 2


# ==========================================
# 6. Mocking External Dependencies
# ==========================================
@patch('requests.post')
def test_quickbase_api_success(mock_post):
    """Verifies successful payload transmission to QuickBase."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    records = [QBUpdateRecord(cell_id=999, pack_id=10)]
    client = QuickBaseClient(records, "https://api.quickbase.com/v1/records", "dbid", "token")
    client.run()

    # Verify requests.post was called with the correct JSON structure
    assert mock_post.called
    args, kwargs = mock_post.call_args
    payload = kwargs['json']
    assert payload['to'] == "dbid"
    assert payload['data'][0]['3']['value'] == 999
    assert payload['data'][0]['129']['value'] == 10


@patch('requests.post')
def test_quickbase_api_error_handling(mock_post, capsys):
    """Verifies the context manager catches HTTP 401s without crashing."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized Token"

    # Simulate the raise_for_status() throwing an HTTPError
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mock_post.return_value = mock_response

    records = [QBUpdateRecord(cell_id=999, pack_id=10)]
    client = QuickBaseClient(records, "https://api.quickbase.com/v1/records", "dbid", "token")

    # This should execute fully without raising an unhandled exception
    client.run()

    # Verify the context manager printed the correct safety log
    captured = capsys.readouterr()
    assert "[ERROR] QuickBase API rejected the request. HTTP 401" in captured.out
