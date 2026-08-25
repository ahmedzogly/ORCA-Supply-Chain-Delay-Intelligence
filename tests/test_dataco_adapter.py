from pathlib import Path
import pytest
from delay_intelligence.data.adapters.dataco import DataCoAdapter

DATA = Path('data/external/dataco/DataCoSupplyChainDataset.csv')
pytestmark = pytest.mark.skipif(not DATA.exists(), reason='DataCo is not bundled; external validation is NOT VALIDATED')


def test_dataco_adapter_loads_and_maps_ontology():
    adapter = DataCoAdapter()
    df = adapter.load_and_transform(nrows=100)
    assert not df.empty
    assert 'T_pred' in df.columns
    assert 'Delay_Flag' in df.columns
    assert 'Shipment Mode' in df.columns


def test_dataco_eligibility():
    adapter = DataCoAdapter()
    df = adapter.load_and_transform(nrows=100)
    assert 'Canceled' not in df['Delivery Status'].values
